//! The in-memory cosine scorer — the reference implementation for the tensor
//! half of the record model.
//!
//! [`CosineBuilder`] scores by walking every record it holds, O(corpus) per
//! query. That is the honest cost of a reference; the memory-mapped
//! [`crate::densepack::DensePack`] is the production path, and this is the
//! oracle it is checked against, so a bug in the packed encoding cannot hide
//! behind a shortcut.

use map_core::{
    CandidateScope, CorpusStats, DimensionScores, IndexedRecord, Located, QueryBundle, Result,
    Scorer, ScorerBuilder, Stage,
};
use map_format::{DType, Tensor};

/// Decode little-endian f32 tensor bytes, or `None` if the dtype is not f32.
fn from_tensor(tensor: &Tensor) -> Option<Vec<f32>> {
    if tensor.dtype != DType::F32 || tensor.validate().is_err() {
        return None;
    }
    Some(
        tensor
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

/// Accumulates tensor records for [`CosineScorer`].
#[derive(Clone, Debug, Default)]
pub struct CosineBuilder {
    dimensions: std::collections::BTreeMap<String, std::collections::BTreeMap<u32, Vec<f32>>>,
    locations: std::collections::BTreeMap<u32, (String, u32, u32, u16)>,
}

impl CosineBuilder {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Stage for CosineBuilder {
    fn implementation(&self) -> &str {
        "cosine"
    }

    fn config(&self) -> String {
        "cosine:v1".to_owned()
    }
}

impl ScorerBuilder for CosineBuilder {
    fn push(&mut self, record: &IndexedRecord<'_>) {
        self.locations.entry(record.id).or_insert_with(|| {
            (
                record.resource.to_owned(),
                record.segment.start,
                record.segment.end,
                record.record.meta.level,
            )
        });

        // Descriptor-only records carry nothing this scorer reads — the mirror
        // image of what the BM25 pack does with tensor-only records.
        let Some(tensor) = record.record.tensor.as_ref() else {
            return;
        };
        let Some(values) = from_tensor(tensor) else {
            return;
        };
        self.dimensions
            .entry(record.record.meta.dimension.clone())
            .or_default()
            .insert(record.id, values);
    }

    fn build(self: Box<Self>) -> Result<Box<dyn Scorer>> {
        Ok(Box::new(CosineScorer {
            dimensions: self.dimensions,
            locations: self.locations,
        }))
    }
}

/// Cosine similarity over stored tensors.
#[derive(Clone, Debug)]
pub struct CosineScorer {
    dimensions: std::collections::BTreeMap<String, std::collections::BTreeMap<u32, Vec<f32>>>,
    locations: std::collections::BTreeMap<u32, (String, u32, u32, u16)>,
}

impl Stage for CosineScorer {
    fn implementation(&self) -> &str {
        "cosine"
    }

    fn config(&self) -> String {
        "cosine:v1".to_owned()
    }
}

impl Scorer for CosineScorer {
    fn len(&self) -> usize {
        self.locations.len()
    }

    /// Visits every record in scope, since there is no index to narrow them.
    ///
    /// Under [`CandidateScope::Ids`] the cost is bounded by the scope; under
    /// `All` it is every stored vector — dense scoring is inherently O(corpus),
    /// as there is no postings list to skip the non-matches. Only an
    /// approximate-nearest-neighbour index would make this sublinear.
    fn score(
        &self,
        scope: CandidateScope<'_>,
        query: &QueryBundle,
        dimensions: &[String],
        _corpus: Option<&CorpusStats>,
    ) -> Result<Vec<(u32, DimensionScores)>> {
        let mut by_id: std::collections::BTreeMap<u32, DimensionScores> = Default::default();

        for name in dimensions {
            let Some(field) = query.get(name) else {
                continue;
            };
            let Some(tensor) = field.tensor.as_ref() else {
                continue;
            };
            let Some(q) = from_tensor(tensor) else {
                continue;
            };
            let Some(vectors) = self.dimensions.get(name) else {
                continue;
            };

            let consider =
                |id: u32,
                 v: &[f32],
                 by_id: &mut std::collections::BTreeMap<u32, DimensionScores>| {
                    if v.len() != q.len() {
                        return;
                    }
                    // Both sides are L2-normalized at construction, so the dot
                    // product is already the cosine — and already in [0,1] once
                    // negatives are dropped. No further normalization needed.
                    let dot: f32 = v.iter().zip(&q).map(|(a, b)| a * b).sum();
                    if dot > 0.0 {
                        by_id
                            .entry(id)
                            .or_default()
                            .insert(name.clone(), dot.min(1.0));
                    }
                };
            match scope {
                CandidateScope::All => {
                    for (id, v) in vectors {
                        consider(*id, v, &mut by_id);
                    }
                }
                CandidateScope::Ids(ids) => {
                    for id in ids {
                        if let Some(v) = vectors.get(id) {
                            consider(*id, v, &mut by_id);
                        }
                    }
                }
            }
        }
        Ok(by_id.into_iter().collect())
    }

    fn located(&self, id: u32) -> Option<Located<'_>> {
        let (resource, start, end, level) = self.locations.get(&id)?;
        Some(Located {
            resource,
            start: *start,
            end: *end,
            level: *level,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::tokenize;
    use map_core::Segment;
    use map_format::{Record, RecordKind, RecordMeta};

    /// Component count for the test fixture below.
    const DIMS: u32 = 256;

    /// FNV-1a, written out rather than taken from `std`, whose `DefaultHasher`
    /// is explicitly not stable across Rust releases — a fixture that shifted
    /// with the toolchain would make these tests mysteriously flaky.
    fn fnv1a(bytes: &[u8]) -> u64 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for b in bytes {
            hash ^= *b as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash
    }

    /// A deterministic text-to-unit-vector projection (the hashing trick), here
    /// only to manufacture tensors for the scorer under test.
    ///
    /// It is emphatically **not** semantic — `authenticate` and `log in` land in
    /// unrelated components — so nothing here is evidence about retrieval
    /// quality. The real embedder is `map-embed`'s distilled one.
    fn project(text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; DIMS as usize];
        for term in tokenize(text) {
            let h = fnv1a(term.as_bytes());
            let index = (h % DIMS as u64) as usize;
            // A sign bit drawn from a different part of the hash keeps unrelated
            // tokens from all pushing the same direction.
            let sign = if (h >> 63) & 1 == 1 { -1.0 } else { 1.0 };
            v[index] += sign;
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }

    fn tensor_of(text: &str) -> Tensor {
        let values = project(text);
        let mut data = Vec::with_capacity(values.len() * 4);
        for x in &values {
            data.extend_from_slice(&x.to_le_bytes());
        }
        Tensor {
            dtype: DType::F32,
            shape: vec![values.len() as u32],
            data,
        }
    }

    fn record_with(tensor: Tensor) -> Record {
        Record {
            descriptor: None,
            tensor: Some(tensor),
            meta: RecordMeta {
                kind: RecordKind::Segment,
                dimension: "descriptive".into(),
                level: 0,
                children: vec![],
            },
        }
    }

    fn scorer_over(texts: &[&str]) -> Box<dyn Scorer> {
        let mut builder = Box::new(CosineBuilder::new());
        for (i, text) in texts.iter().enumerate() {
            let record = record_with(tensor_of(text));
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

    fn dims() -> Vec<String> {
        vec!["descriptive".to_owned()]
    }

    /// A dense query bundle for the `descriptive` dimension.
    fn q(text: &str) -> QueryBundle {
        let mut bundle = QueryBundle::new();
        bundle.insert(
            "descriptive".to_owned(),
            map_core::QueryField {
                text: None,
                tensor: Some(tensor_of(text)),
            },
        );
        bundle
    }

    /// Score every record and rank, as the retriever does.
    fn run(scorer: &dyn Scorer, text: &str) -> Vec<(u32, f32)> {
        let scored = scorer
            .score(CandidateScope::All, &q(text), &dims(), None)
            .unwrap();
        let mut out: Vec<(u32, f32)> = scored
            .into_iter()
            .filter_map(|(id, s)| s.get("descriptive").map(|v| (id, *v)))
            .collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        out
    }

    #[test]
    fn a_query_retrieves_the_record_sharing_its_tokens() {
        let scorer = scorer_over(&[
            "fn parse_config(path: &Path)",
            "fn refresh_token(session: &Session)",
            "struct Logger { level: Level }",
        ]);
        let hits = run(scorer.as_ref(), "refresh token session");
        assert_eq!(hits[0].0, 1, "the session-refresh record should rank first");
    }

    #[test]
    fn every_score_is_a_normalized_relevance() {
        // Same invariant the BM25 side asserts, reached a different way: this
        // scorer is bounded by construction and needs no ceiling.
        let scorer = scorer_over(&[
            "fn refresh_token(session: &Session)",
            "fn parse_config(path: &Path)",
        ]);
        for (_, score) in run(scorer.as_ref(), "refresh token session") {
            assert!((0.0..=1.0).contains(&score), "cosine out of range: {score}");
        }
    }

    #[test]
    fn work_is_bounded_by_the_candidate_set_not_the_corpus() {
        // A brute-force scorer has no index, so the candidate set is the only
        // thing bounding its cost. It must honour that bound rather than
        // scanning everything it holds.
        let texts: Vec<String> = (0..100).map(|i| format!("item{i} shared")).collect();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let scorer = scorer_over(&refs);

        let scored = scorer
            .score(CandidateScope::Ids(&[5, 42]), &q("shared"), &dims(), None)
            .unwrap();
        assert!(
            scored.iter().all(|(id, _)| *id == 5 || *id == 42),
            "scorer reported a record outside the candidate set"
        );
    }

    #[test]
    fn descriptor_only_records_are_skipped_not_rejected() {
        let record = Record {
            descriptor: Some("alpha beta".into()),
            tensor: None,
            meta: RecordMeta {
                kind: RecordKind::Segment,
                dimension: "lexical".into(),
                level: 0,
                children: vec![],
            },
        };
        let mut builder = Box::new(CosineBuilder::new());
        builder.push(&IndexedRecord {
            id: 0,
            resource: "t.rs",
            segment: Segment { start: 0, end: 1 },
            record: &record,
        });
        let scorer = builder.build().unwrap();
        // The location slot survives; there is simply no vector to score.
        assert_eq!(scorer.len(), 1);
        assert!(scorer
            .score(CandidateScope::Ids(&[0]), &q("alpha"), &dims(), None)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn a_text_only_query_scores_nothing_against_a_tensor_scorer() {
        let scorer = scorer_over(&["fn main() {}"]);
        let mut bundle = QueryBundle::new();
        bundle.insert("descriptive".to_owned(), map_core::QueryField::text("main"));
        assert!(scorer
            .score(CandidateScope::Ids(&[0]), &bundle, &dims(), None)
            .unwrap()
            .is_empty());
    }
}
