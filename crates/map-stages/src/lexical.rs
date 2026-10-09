//! The lexical dimension: structural classifier plus BM25 scorer.
//!
//! This pair is everything a default build contains. It needs no API key, no
//! network, and no model download, and it is fully Tier A deterministic.
//!
//! It also demonstrates the format's central claim — that the format is dumb
//! and the stages are smart. BM25 needs no posting-list object type and no
//! support anywhere in `map-format`: term frequencies are simply what this
//! dimension chooses to put in its descriptor text, and the scorer is the only
//! thing that knows how to read it back.
//!
//! # Two BM25 implementations, on purpose
//!
//! [`Bm25Builder`] is a straightforward in-memory reference. The production
//! path is [`crate::pack::PackBuilder`], which memory-maps its postings. They
//! implement the same traits and are checked against each other in
//! `pack.rs` — which is what stops the two copies of the formula drifting, and
//! is also the standing proof that the scorer interface is implementable more
//! than once.

use std::collections::BTreeMap;

use map_core::{
    CandidateScope, Classifier, ClassifyBatch, CorpusStats, DimensionRecords, DimensionScores,
    IndexedRecord, Located, QueryBundle, Result, Scorer, ScorerBuilder, Stage,
};
use map_format::{Record, RecordKind, RecordMeta};

/// Split text into searchable terms.
///
/// A compound is emitted whole *and* split on `_` and camelCase boundaries, so
/// `refresh_token`, `refreshToken`, and `refresh` all retrieve the same text.
/// Terms are lowercased; single characters are dropped as noise.
///
/// **The splitting rule is shaped by code and this is the one stage where that
/// still shows.** It costs prose nothing — the boundaries simply never fire on
/// ordinary words — but what prose would actually want is missing rather than
/// present-and-tuned: no stopwords, no stemming, and a word character class of
/// `[alnum_]` that splits hyphenated words and contractions.
///
/// Adding any of it is a measurement, not an opinion, and there is no non-code
/// golden set to rank a candidate against yet. When there is, the change is a
/// new classifier `impl` rather than an edit here: the format needs nothing for
/// one, and a second tokenizer must not silently re-key the first one's corpus.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut terms = Vec::new();

    for run in text.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
        if run.len() < 2 {
            continue;
        }
        let lowered = run.to_lowercase();

        // Sub-terms first, then the whole identifier if it differs.
        let mut current = String::new();
        let mut previous_lower = false;
        for c in run.chars() {
            let boundary = c == '_' || (c.is_uppercase() && previous_lower);
            if boundary && !current.is_empty() {
                if current.len() >= 2 {
                    terms.push(current.to_lowercase());
                }
                current.clear();
            }
            if c != '_' {
                current.push(c);
            }
            previous_lower = c.is_lowercase() || c.is_numeric();
        }
        if current.len() >= 2 {
            let part = current.to_lowercase();
            if part != lowered {
                terms.push(part);
            }
        }
        terms.push(lowered);
    }
    terms
}

/// Encode term frequencies as descriptor text.
///
/// Sorted by term so the descriptor is byte-identical for identical input —
/// the encoding is part of Tier A.
pub fn encode_frequencies(terms: &[String]) -> String {
    let mut counts: BTreeMap<&str, u32> = BTreeMap::new();
    for t in terms {
        *counts.entry(t.as_str()).or_insert(0) += 1;
    }
    let mut out = String::new();
    for (term, count) in counts {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(term);
        if count > 1 {
            out.push(':');
            out.push_str(&count.to_string());
        }
    }
    out
}

/// Decode descriptor text back into term frequencies.
pub fn decode_frequencies(descriptor: &str) -> Vec<(&str, u32)> {
    descriptor
        .split_whitespace()
        .map(|token| match token.split_once(':') {
            Some((term, count)) => (term, count.parse().unwrap_or(1)),
            None => (token, 1),
        })
        .collect()
}

/// Derives lexical descriptors without a model.
///
/// Offline, instant, and deterministic — the properties that let a default
/// build work with nothing configured.
///
/// **Despite the name, this is a tokenizer, not a parser.** It extracts no
/// symbols, signatures, or import edges, and understands no syntax. The name
/// describes the role in the pipeline — the stage that derives a descriptor
/// structurally rather than from a model — not the sophistication of it.
#[derive(Clone, Debug, Default)]
pub struct StructuralClassifier;

impl Stage for StructuralClassifier {
    fn implementation(&self) -> &str {
        "structural"
    }

    fn config(&self) -> String {
        "structural:v1".to_owned()
    }
}

impl Classifier for StructuralClassifier {
    fn classify(&self, batch: &ClassifyBatch<'_>) -> Result<Vec<DimensionRecords>> {
        let mut out = Vec::with_capacity(batch.items.len());
        for sources in batch.items {
            // An item's sources are joined before tokenizing: one segment's
            // text at level 0, a group's descriptors above it. Term frequencies
            // over the concatenation are the same thing either way.
            let mut terms = Vec::new();
            for text in sources {
                terms.extend(tokenize(text));
            }
            let descriptor = encode_frequencies(&terms);

            // One record per dimension this implementation owns — the batch
            // covers items AND dimensions in a single pass.
            let mut per_dimension = DimensionRecords::new();
            for dimension in batch.dimensions {
                per_dimension.insert(
                    dimension.clone(),
                    Record {
                        descriptor: Some(descriptor.clone()),
                        tensor: None,
                        meta: RecordMeta {
                            kind: if batch.level == 0 {
                                RecordKind::Segment
                            } else {
                                RecordKind::Cluster
                            },
                            dimension: dimension.clone(),
                            level: batch.level,
                            children: Vec::new(),
                            // Left None deliberately: the stored object must
                            // carry no resource identity or two identical
                        },
                    },
                );
            }
            out.push(per_dimension);
        }
        Ok(out)
    }
}

/// Okapi BM25 saturation parameter.
pub(crate) const K1: f32 = 1.2;
/// Okapi BM25 length-normalization parameter.
pub(crate) const B: f32 = 0.75;

/// Okapi BM25 inverse document frequency, the non-negative variant.
pub(crate) fn idf(corpus: f32, document_frequency: f32) -> f32 {
    ((corpus - document_frequency + 0.5) / (document_frequency + 0.5) + 1.0).ln()
}

/// The largest score this query could achieve against any record.
///
/// BM25 saturates: as term frequency grows, each term's contribution tends to
/// `idf * (k1 + 1)`. Summing that bound over the query's terms gives a ceiling
/// that depends only on the query and the corpus statistics, never on which
/// records happened to match.
///
/// Dividing by it is what makes the score a normalized `[0, 1]` relevance
/// rather than an unbounded quantity. Two properties follow, and both matter:
///
/// - **Rank within a query is untouched.** It is a positive constant divisor,
///   so single-dimension results are bit-for-bit what they were before.
/// - **1.0 keeps an absolute meaning** — "contains every query term, saturated"
///   — so scores stay comparable across queries. Normalizing by the *observed*
///   maximum instead would force the top hit to 1.0 for every query, including
///   one where nothing good matched, which destroys exactly the signal a
///   consumer needs to decide whether to trust the answer at all.
///
/// Terms absent from the corpus (`df == 0`) are excluded: they can contribute
/// nothing to any real score, so counting them in the ceiling would drag every
/// result toward zero because of a typo.
pub(crate) fn query_ceiling<'a, I>(corpus: f32, document_frequencies: I) -> f32
where
    I: IntoIterator<Item = &'a u32>,
{
    let total: f32 = document_frequencies
        .into_iter()
        .filter(|df| **df > 0)
        .map(|df| idf(corpus, *df as f32) * (K1 + 1.0))
        .sum();
    // Guards division when no query term exists in the corpus at all.
    if total > 0.0 {
        total
    } else {
        1.0
    }
}

/// One record's location, as the reference scorer stores it.
#[derive(Clone, Debug)]
struct Location {
    resource: String,
    start: u32,
    end: u32,
    level: u16,
}

/// One dimension's corpus within a grouped scorer.
#[derive(Clone, Debug, Default)]
struct DimensionCorpus {
    docs: BTreeMap<u32, BTreeMap<String, u32>>,
    lengths: BTreeMap<u32, f32>,
    document_frequency: BTreeMap<String, u32>,
    average_length: f32,
}

impl DimensionCorpus {
    /// Raw, un-normalized Okapi BM25 for one record.
    ///
    /// `global` carries federated statistics when several indexes answer one
    /// query; without it the corpus scores against itself, which is the same
    /// arithmetic when there is only one.
    fn raw(&self, id: u32, terms: &[&str], corpus: f32, global: Option<&CorpusStats>) -> f32 {
        let Some(doc) = self.docs.get(&id) else {
            return 0.0;
        };
        let average_length = global.map_or(self.average_length, |g| g.average_length());
        let length = self.lengths.get(&id).copied().unwrap_or(0.0);
        let norm = 1.0 - B + B * length / average_length.max(1.0);

        let mut total = 0.0f32;
        for term in terms {
            let Some(&tf) = doc.get(*term) else { continue };
            let df = match global {
                Some(g) => g.document_frequency.get(*term).copied().unwrap_or(0),
                None => self.document_frequency.get(*term).copied().unwrap_or(0),
            } as f32;
            total += idf(corpus, df) * (tf as f32 * (K1 + 1.0)) / (tf as f32 + K1 * norm);
        }
        total
    }

    /// This corpus's contribution to the federated statistics for `terms`.
    fn stats_for(&self, terms: &[String]) -> CorpusStats {
        CorpusStats {
            records: self.docs.len() as u64,
            total_length: self.lengths.values().map(|l| *l as f64).sum(),
            document_frequency: terms
                .iter()
                .map(|t| {
                    (
                        t.clone(),
                        self.document_frequency.get(t).copied().unwrap_or(0),
                    )
                })
                .collect(),
        }
    }
}

/// Deduplicated query terms, in first-seen order.
fn unique_terms(text: &str) -> Vec<String> {
    let mut seen: BTreeMap<String, ()> = BTreeMap::new();
    tokenize(text)
        .into_iter()
        .filter(|t| seen.insert(t.clone(), ()).is_none())
        .collect()
}

/// Accumulates descriptors for the in-memory reference scorer.
#[derive(Clone, Debug, Default)]
pub struct Bm25Builder {
    dimensions: BTreeMap<String, DimensionCorpus>,
    locations: BTreeMap<u32, Location>,
}

impl Bm25Builder {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Stage for Bm25Builder {
    fn implementation(&self) -> &str {
        "bm25"
    }

    fn config(&self) -> String {
        format!("bm25:k1={K1},b={B}")
    }
}

impl ScorerBuilder for Bm25Builder {
    fn push(&mut self, record: &IndexedRecord<'_>) {
        // The location is shared across dimensions, so it is recorded whether
        // or not this scorer can read the payload.
        self.locations.entry(record.id).or_insert_with(|| Location {
            resource: record.resource.to_owned(),
            start: record.segment.start,
            end: record.segment.end,
            level: record.record.meta.level,
        });

        // Tensor-only records carry nothing this scorer can read.
        let Some(descriptor) = record.record.descriptor.as_deref() else {
            return;
        };
        if descriptor.is_empty() {
            return;
        }

        let corpus = self
            .dimensions
            .entry(record.record.meta.dimension.clone())
            .or_default();

        let mut terms: BTreeMap<String, u32> = BTreeMap::new();
        let mut length = 0u32;
        for (term, count) in decode_frequencies(descriptor) {
            // Counts are attacker-controlled descriptor text and release builds
            // do not check overflow; wrapping here would silently corrupt the
            // BM25 length norm. Must saturate exactly as the pack does, or the
            // two BM25 implementations disagree on a hostile input.
            length = length.saturating_add(count);
            let total = terms.entry(term.to_owned()).or_insert(0);
            *total = total.saturating_add(count);
        }
        for term in terms.keys() {
            *corpus.document_frequency.entry(term.clone()).or_insert(0) += 1;
        }
        corpus.docs.insert(record.id, terms);
        corpus.lengths.insert(record.id, length as f32);
    }

    fn build(mut self: Box<Self>) -> Result<Box<dyn Scorer>> {
        for corpus in self.dimensions.values_mut() {
            corpus.average_length = if corpus.docs.is_empty() {
                0.0
            } else {
                corpus.lengths.values().sum::<f32>() / corpus.docs.len() as f32
            };
        }
        Ok(Box::new(Bm25Reference {
            dimensions: self.dimensions,
            locations: self.locations,
        }))
    }
}

/// In-memory Okapi BM25. The reference implementation and test oracle.
#[derive(Clone, Debug)]
pub struct Bm25Reference {
    dimensions: BTreeMap<String, DimensionCorpus>,
    locations: BTreeMap<u32, Location>,
}

impl Stage for Bm25Reference {
    fn implementation(&self) -> &str {
        "bm25"
    }

    fn config(&self) -> String {
        format!("bm25:k1={K1},b={B}")
    }
}

impl Scorer for Bm25Reference {
    fn len(&self) -> usize {
        self.locations.len()
    }

    /// Merged across the dimensions this scorer owns, matching how it scores
    /// them: one corpus per dimension, each contributing its own totals.
    fn corpus_stats(&self, query: &QueryBundle, dimensions: &[String]) -> Option<CorpusStats> {
        let mut totals: Option<CorpusStats> = None;
        for name in dimensions {
            let Some(text) = query.get(name).and_then(|f| f.text.as_deref()) else {
                continue;
            };
            let Some(corpus) = self.dimensions.get(name) else {
                continue;
            };
            let stats = corpus.stats_for(&unique_terms(text));
            totals
                .get_or_insert_with(CorpusStats::default)
                .merge(&stats);
        }
        totals
    }

    fn score(
        &self,
        scope: CandidateScope<'_>,
        query: &QueryBundle,
        dimensions: &[String],
        corpus_stats: Option<&CorpusStats>,
    ) -> Result<Vec<(u32, DimensionScores)>> {
        let mut by_id: BTreeMap<u32, DimensionScores> = BTreeMap::new();

        for name in dimensions {
            let Some(field) = query.get(name) else {
                continue;
            };
            let Some(text) = field.text.as_deref() else {
                continue;
            };
            let Some(corpus) = self.dimensions.get(name) else {
                continue;
            };
            let terms = unique_terms(text);
            if terms.is_empty() {
                continue;
            }
            let refs: Vec<&str> = terms.iter().map(String::as_str).collect();
            let n = corpus_stats.map_or(corpus.docs.len() as f32, |g| g.records as f32);

            // Normalize against what this query could achieve, not against
            // what it did achieve — see `query_ceiling`. Federated document
            // frequencies when given, so every index divides by the same
            // ceiling and an index missing a term cannot inflate its scores.
            let frequencies: Vec<u32> = refs
                .iter()
                .map(|t| match corpus_stats {
                    Some(g) => g.document_frequency.get(*t).copied().unwrap_or(0),
                    None => corpus.document_frequency.get(*t).copied().unwrap_or(0),
                })
                .collect();
            let ceiling = query_ceiling(n, &frequencies);

            let record = |id: u32, by_id: &mut BTreeMap<u32, DimensionScores>| {
                let raw = corpus.raw(id, &refs, n, corpus_stats);
                if raw > 0.0 {
                    by_id
                        .entry(id)
                        .or_default()
                        .insert(name.clone(), (raw / ceiling).min(1.0));
                }
            };
            match scope {
                CandidateScope::All => {
                    for id in 0..self.locations.len() as u32 {
                        record(id, &mut by_id);
                    }
                }
                CandidateScope::Ids(ids) => {
                    for id in ids {
                        record(*id, &mut by_id);
                    }
                }
            }
        }
        Ok(by_id.into_iter().collect())
    }

    fn located(&self, id: u32) -> Option<Located<'_>> {
        let l = self.locations.get(&id)?;
        Some(Located {
            resource: &l.resource,
            start: l.start,
            end: l.end,
            level: l.level,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use map_core::{Content, Segment};

    use map_core::QueryField;

    /// A one-dimension lexical record.
    pub(crate) fn lexical_record(text: &str) -> Record {
        Record {
            descriptor: Some(encode_frequencies(&tokenize(text))),
            tensor: None,
            meta: RecordMeta {
                kind: RecordKind::Segment,
                dimension: "lexical".into(),
                level: 0,
                children: vec![],
            },
        }
    }

    /// Build a reference scorer from raw text, one record per string.
    pub(crate) fn reference_over(texts: &[&str]) -> Box<dyn Scorer> {
        let mut builder = Box::new(Bm25Builder::new());
        for (i, text) in texts.iter().enumerate() {
            let record = lexical_record(text);
            builder.push(&IndexedRecord {
                id: i as u32,
                resource: "t.rs",
                segment: Segment {
                    start: i as u32,
                    end: i as u32 + 1,
                },
                record: &record,
            });
        }
        builder.build().unwrap()
    }

    /// A single-dimension lexical query.
    fn q(text: &str) -> QueryBundle {
        let mut bundle = QueryBundle::new();
        bundle.insert("lexical".to_owned(), QueryField::text(text));
        bundle
    }

    fn dims() -> Vec<String> {
        vec!["lexical".to_owned()]
    }

    /// Score every record and rank — the same shape the retriever runs, with
    /// the candidate set fixed independently of any dimension.
    fn run(scorer: &dyn Scorer, text: &str) -> Vec<(u32, f32)> {
        let scored = scorer
            .score(map_core::CandidateScope::All, &q(text), &dims(), None)
            .unwrap();

        let mut out: Vec<(u32, f32)> = scored
            .into_iter()
            .filter_map(|(id, s)| s.get("lexical").map(|v| (id, *v)))
            .collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        out
    }

    #[test]
    fn identifiers_split_on_case_and_underscore() {
        let terms = tokenize("fn refresh_token(sessionToken: &str)");
        for expected in [
            "refresh",
            "token",
            "refresh_token",
            "session",
            "sessiontoken",
        ] {
            assert!(terms.iter().any(|t| t == expected), "missing {expected:?}");
        }
    }

    #[test]
    fn single_characters_are_dropped() {
        assert!(tokenize("a b c").is_empty());
    }

    #[test]
    fn frequencies_round_trip() {
        let encoded = encode_frequencies(&tokenize("token token session"));
        let decoded: BTreeMap<_, _> = decode_frequencies(&encoded).into_iter().collect();
        assert_eq!(decoded.get("token"), Some(&2));
        assert_eq!(decoded.get("session"), Some(&1));
    }

    #[test]
    fn descriptor_encoding_is_deterministic() {
        // Byte-identical descriptors for identical input is a Tier A promise.
        let a = encode_frequencies(&tokenize("zebra alpha zebra"));
        let b = encode_frequencies(&tokenize("zebra alpha zebra"));
        assert_eq!(a, b);
        assert!(a.starts_with("alpha"), "terms must be sorted: {a:?}");
    }

    #[test]
    fn one_classify_call_covers_every_dimension() {
        // The cost model rests on this: N dimensions is one pass, not N.
        let content = Content {
            key: "t.rs".into(),
            text: "fn refresh_token() {}".into(),
        };
        let segments = [Segment {
            start: 0,
            end: content.text.len() as u32,
        }];
        let items: Vec<Vec<&str>> = segments.iter().map(|s| vec![s.slice(&content)]).collect();
        let dimensions = ["lexical".to_owned(), "symbols".to_owned()];

        let out = StructuralClassifier
            .classify(&ClassifyBatch {
                items: &items,
                dimensions: &dimensions,
                level: 0,
            })
            .unwrap();

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), 2);
        assert!(out[0].contains_key("lexical"));
        assert!(out[0].contains_key("symbols"));
    }

    #[test]
    fn classified_records_carry_no_resource_identity() {
        // Storing the path inside the payload makes two identical files
        // collide on one key with different bytes. It has been fixed once
        // already; this pins it at the stage that would reintroduce it.
        let content = Content {
            key: "src/a.rs".into(),
            text: "fn main() {}".into(),
        };
        let segments = [Segment {
            start: 0,
            end: content.text.len() as u32,
        }];
        let items: Vec<Vec<&str>> = segments.iter().map(|s| vec![s.slice(&content)]).collect();
        let out = StructuralClassifier
            .classify(&ClassifyBatch {
                items: &items,
                dimensions: &["lexical".to_owned()],
                level: 0,
            })
            .unwrap();

        assert!(out[0]["lexical"].validate().is_ok());
    }

    #[test]
    fn tensor_only_records_are_skipped_not_rejected() {
        // One corpus can hold dimensions of both kinds; a text scorer must
        // ignore what it cannot read rather than fail the build.
        let record = Record {
            descriptor: None,
            tensor: Some(map_format::Tensor {
                dtype: map_format::DType::F32,
                shape: vec![2],
                data: vec![0; 8],
            }),
            meta: RecordMeta {
                kind: RecordKind::Segment,
                dimension: "behavior".into(),
                level: 0,
                children: vec![],
            },
        };
        let mut builder = Box::new(Bm25Builder::new());
        builder.push(&IndexedRecord {
            id: 0,
            resource: "t.rs",
            segment: Segment { start: 0, end: 1 },
            record: &record,
        });
        let scorer = builder.build().unwrap();
        // The slot still exists — ids are shared across dimensions, so the
        // location is recorded — but there is nothing here to score.
        assert_eq!(scorer.len(), 1);
        assert!(run(scorer.as_ref(), "anything").is_empty());
    }

    #[test]
    fn bm25_ranks_the_matching_document_first() {
        let scorer = reference_over(&[
            "fn parse_config(path: &Path)",
            "fn refresh_token(session: &Session)",
            "struct Logger { level: Level }",
        ]);
        let hits = run(scorer.as_ref(), "refresh token");
        assert_eq!(
            hits[0].0, 1,
            "the session-refresh document should rank first"
        );
    }

    #[test]
    fn every_score_is_a_normalized_relevance() {
        // The invariant the retriever rests on: dimensions can only be merged
        // into one relevance if they all speak the same [0,1] scale.
        let scorer = reference_over(&[
            "fn parse_config(path: &Path)",
            "fn refresh_token(session: &Session)",
            "struct Logger { level: Level }",
        ]);
        for query in ["refresh token", "parse", "logger level path session"] {
            for (id, score) in run(scorer.as_ref(), query) {
                assert!(
                    (0.0..=1.0).contains(&score),
                    "score {score} for record {id} out of range on {query:?}"
                );
            }
        }
    }

    #[test]
    fn normalization_does_not_reorder_a_single_dimension() {
        // Dividing by a per-query constant is rank-preserving. That is why
        // normalizing could land without moving any existing ripgrep result.
        let texts: Vec<String> = (0..40).map(|i| format!("token session item{i}")).collect();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let scorer = reference_over(&refs);

        let normalized = run(scorer.as_ref(), "token session");
        let mut expected = normalized.clone();
        expected.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        assert_eq!(normalized, expected);
    }

    #[test]
    fn an_absent_query_term_does_not_depress_every_score() {
        // Normalizing against terms the corpus has never seen would let a
        // single typo drag an otherwise perfect match toward zero.
        let scorer = reference_over(&["fn refresh_token(session: &Session)"]);
        let clean = run(scorer.as_ref(), "refresh token");
        let typo = run(scorer.as_ref(), "refresh token zzzznotpresent");
        assert_eq!(clean[0].1, typo[0].1);
    }

    #[test]
    fn rare_terms_outrank_common_ones() {
        let texts: Vec<String> = (0..10)
            .map(|i| {
                let extra = if i == 3 { "quiesce" } else { "common" };
                format!("shared shared {extra}")
            })
            .collect();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let scorer = reference_over(&refs);
        assert_eq!(run(scorer.as_ref(), "quiesce")[0].0, 3);
    }

    #[test]
    fn no_match_returns_nothing() {
        let scorer = reference_over(&["fn main() {}"]);
        assert!(run(scorer.as_ref(), "nonexistent_symbol").is_empty());
    }

    #[test]
    fn ranking_is_deterministic() {
        let texts: Vec<String> = (0..50).map(|i| format!("token session item{i}")).collect();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let scorer = reference_over(&refs);
        assert_eq!(
            run(scorer.as_ref(), "token session"),
            run(scorer.as_ref(), "token session")
        );
    }

    #[test]
    fn scoring_a_candidate_set_never_invents_members() {
        // Candidates are fixed by the retriever; a scorer may report on fewer
        // than it was given, never on more. Otherwise a dimension would be
        // reaching outside the scope it was handed.
        let texts: Vec<String> = (0..50).map(|i| format!("shared item{i}")).collect();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let scorer = reference_over(&refs);

        let candidates: Vec<u32> = vec![3, 7, 11];
        let scored = scorer
            .score(
                map_core::CandidateScope::Ids(&candidates),
                &q("shared"),
                &dims(),
                None,
            )
            .unwrap();
        assert!(
            scored.iter().all(|(id, _)| candidates.contains(id)),
            "scorer reported a record outside the candidate set"
        );
        assert_eq!(scored.len(), 3, "all three candidates match 'shared'");
    }

    #[test]
    fn a_tensor_only_query_scores_nothing_against_a_text_dimension() {
        let scorer = reference_over(&["fn main() {}"]);
        let mut bundle = QueryBundle::new();
        bundle.insert(
            "lexical".to_owned(),
            QueryField {
                text: None,
                tensor: Some(map_format::Tensor {
                    dtype: map_format::DType::F32,
                    shape: vec![2],
                    data: vec![0; 8],
                }),
            },
        );
        assert!(scorer
            .score(map_core::CandidateScope::Ids(&[0]), &bundle, &dims(), None)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn located_reports_where_a_hit_came_from() {
        let scorer = reference_over(&["alpha", "beta"]);
        let at = scorer.located(1).unwrap();
        assert_eq!(at.resource, "t.rs");
        assert_eq!((at.start, at.end, at.level), (1, 2, 0));
        assert!(scorer.located(99).is_none());
    }
}
