//! The lexical dimension's acceleration structure: an mmap-able inverted
//! index.
//!
//! # Why this exists
//!
//! Loading an index by reading every object, verifying it, and parsing JSON is
//! O(records) *per query*, and scoring by walking every record is O(records)
//! *per query* again. Measured on ripgrep: 152 ms to load 2,481 records and
//! ~6 ms to score them. At 100k records that load is seconds, and the scan
//! dominates even when twelve records match.
//!
//! A resident daemon would have hidden the load behind a warm process and left
//! the linear scan exactly where it was. This fixes both:
//!
//! | | before | with a pack |
//! |---|---|---|
//! | load | O(records) — open, verify, parse each object | O(1) — one `mmap` |
//! | query | O(records) — score everything | O(matching) — walk postings |
//!
//! # Ownership
//!
//! The **scorer owns its acceleration structure**. This file is the BM25
//! scorer's format and nothing in `map-format` knows it exists; a dense
//! scorer would own an ANN graph file the same way. That is the same
//! "dumb format, smart stages" split that lets BM25 term frequencies live in
//! opaque descriptor text.
//!
//! The pack is **derived** — it lives under `.map/cache/`, is gitignored, and
//! can be rebuilt from committed objects at any time. Deleting it costs time,
//! never information.
//!
//! # Layout
//!
//! Little-endian throughout, sections 8-byte aligned, no pointer chasing:
//!
//! ```text
//! header    64 B     magic, version, counts, section offsets, avgdl
//! records   n × 24 B resource span into the string pool, byte span, doc length
//! terms     n × 16 B term span into the string pool, postings span
//! postings  n × 8 B  (record id, term frequency)
//! strings            resource keys and terms, concatenated
//! ```
//!
//! Terms are sorted so lookup is a binary search over fixed-size entries;
//! records keep insertion order so ids stay stable.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

// `Result` is deliberately not imported: this module has its own
// `Result<T, PackError>` in scope and importing map_core's one-parameter alias
// silently shadows it.
use map_core::{
    CandidateScope, CorpusStats, DimensionScores, IndexedRecord, Located, QueryBundle, Scorer,
    ScorerBuilder, Stage,
};

// The scoring constants, the IDF formula and the normalization ceiling are
// imported rather than restated. They were duplicated here once, which left two
// copies of Okapi BM25 that happened to agree; nothing would have caught them
// drifting apart.
use crate::lexical::{decode_frequencies, idf, query_ceiling, tokenize, B, K1};

const MAGIC: &[u8; 8] = b"MAPPACK\0";
const VERSION: u32 = 1;
const HEADER_LEN: usize = 64;
const RECORD_LEN: usize = 24;
const TERM_LEN: usize = 16;
const POSTING_LEN: usize = 8;

/// Errors from reading a pack.
#[derive(Debug, thiserror::Error)]
pub enum PackError {
    #[error("io error at {path}: {source}")]
    Io {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("malformed pack: {0}")]
    Malformed(&'static str),
    /// Written by a different version of this format.
    ///
    /// Distinct from [`Malformed`](PackError::Malformed) because the caller's
    /// response differs: a pack is gitignored derived cache, so an old one is
    /// rebuilt rather than reported. Folding this into the generic malformed
    /// case froze the layout by accident — bumping `VERSION` broke every
    /// existing cache with a hard error instead of costing one rebuild.
    #[error("pack was written by format version {found}; this build writes {expected}")]
    UnsupportedVersion { found: u32, expected: u32 },
}

impl PackError {
    /// Whether this pack can simply be rebuilt from the objects.
    pub fn is_rebuildable(&self) -> bool {
        matches!(self, PackError::UnsupportedVersion { .. })
    }
}

/// Accumulates records and emits pack bytes.
#[derive(Debug, Default)]
pub struct PackBuilder {
    /// Learned from the first record pushed.
    dimension: String,
    records: Vec<BuilderRecord>,
    /// term -> (record id, frequency)
    postings: BTreeMap<String, Vec<(u32, u32)>>,
}

#[derive(Debug)]
struct BuilderRecord {
    resource: String,
    start: u32,
    end: u32,
    length: u32,
    level: u16,
}

impl PackBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn finish(self) -> Vec<u8> {
        let mut strings: Vec<u8> = Vec::new();
        // Resource keys repeat heavily (one per segment), so intern them.
        let mut interned: BTreeMap<&str, (u32, u32)> = BTreeMap::new();

        let mut record_bytes = Vec::with_capacity(self.records.len() * RECORD_LEN);
        let mut total_length = 0u64;
        for record in &self.records {
            let (offset, len) = *interned.entry(&record.resource).or_insert_with(|| {
                let offset = strings.len() as u32;
                strings.extend_from_slice(record.resource.as_bytes());
                (offset, record.resource.len() as u32)
            });
            record_bytes.extend_from_slice(&offset.to_le_bytes());
            record_bytes.extend_from_slice(&len.to_le_bytes());
            record_bytes.extend_from_slice(&record.start.to_le_bytes());
            record_bytes.extend_from_slice(&record.end.to_le_bytes());
            record_bytes.extend_from_slice(&record.length.to_le_bytes());
            // Fabric height, in what was a reserved word. Retrieval zoom is a
            // filter on this, so it has to survive into the mapped form.
            record_bytes.extend_from_slice(&(record.level as u32).to_le_bytes());
            total_length += record.length as u64;
        }

        let mut term_bytes = Vec::with_capacity(self.postings.len() * TERM_LEN);
        let mut posting_bytes = Vec::new();
        for (term, entries) in &self.postings {
            let term_offset = strings.len() as u32;
            strings.extend_from_slice(term.as_bytes());

            let postings_offset = (posting_bytes.len() / POSTING_LEN) as u32;
            for (id, frequency) in entries {
                posting_bytes.extend_from_slice(&id.to_le_bytes());
                posting_bytes.extend_from_slice(&frequency.to_le_bytes());
            }

            term_bytes.extend_from_slice(&term_offset.to_le_bytes());
            term_bytes.extend_from_slice(&(term.len() as u32).to_le_bytes());
            term_bytes.extend_from_slice(&postings_offset.to_le_bytes());
            term_bytes.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        }

        // Averaged over records that actually carry text. Empty slots exist so
        // ids stay positional; counting them would drag `avgdl` down and
        // distort every length normalization in the corpus.
        let scoreable = self.records.iter().filter(|r| r.length > 0).count();
        let average_length = if scoreable == 0 {
            0.0f32
        } else {
            total_length as f32 / scoreable as f32
        };

        let off_records = HEADER_LEN as u64;
        let off_terms = off_records + record_bytes.len() as u64;
        let off_postings = off_terms + term_bytes.len() as u64;
        let off_strings = off_postings + posting_bytes.len() as u64;

        let mut out = Vec::with_capacity(off_strings as usize + strings.len());
        out.extend_from_slice(MAGIC); //                          0..8
        out.extend_from_slice(&VERSION.to_le_bytes()); //          8..12
        out.extend_from_slice(&(self.records.len() as u32).to_le_bytes()); // 12..16
        out.extend_from_slice(&(self.postings.len() as u32).to_le_bytes()); // 16..20
        out.extend_from_slice(&average_length.to_le_bytes()); //  20..24
        out.extend_from_slice(&off_records.to_le_bytes()); //     24..32
        out.extend_from_slice(&off_terms.to_le_bytes()); //       32..40
        out.extend_from_slice(&off_postings.to_le_bytes()); //    40..48
        out.extend_from_slice(&off_strings.to_le_bytes()); //     48..56
        out.extend_from_slice(&[0u8; 8]); // reserved             56..64
        debug_assert_eq!(out.len(), HEADER_LEN);

        out.extend_from_slice(&record_bytes);
        out.extend_from_slice(&term_bytes);
        out.extend_from_slice(&posting_bytes);
        out.extend_from_slice(&strings);
        out
    }
}

impl Stage for PackBuilder {
    fn implementation(&self) -> &str {
        "bm25"
    }

    fn config(&self) -> String {
        format!("bm25:k1={K1},b={B}")
    }
}

impl ScorerBuilder for PackBuilder {
    /// Record ids are **positional**, which is only sound because this method
    /// allocates a slot for every record it is offered — including a tensor-only
    /// one it cannot read. Skipping would renumber every later record, so a
    /// candidate list from another dimension would address the wrong spans here.
    ///
    /// Chosen over widening the on-disk record to carry an explicit id: an
    /// unreadable record's slot costs 24 bytes and no postings, cheaper than a
    /// format change.
    fn push(&mut self, record: &IndexedRecord<'_>) {
        debug_assert_eq!(
            record.id as usize,
            self.records.len(),
            "pack ids are positional; the driver must push every record of a \
             dimension in id order"
        );
        if self.dimension.is_empty() {
            self.dimension = record.record.meta.dimension.clone();
        }

        let mut length = 0u32;
        if let Some(descriptor) = record.record.descriptor.as_deref() {
            let id = self.records.len() as u32;
            let mut counts: BTreeMap<&str, u32> = BTreeMap::new();
            for (term, count) in decode_frequencies(descriptor) {
                // The counts come out of committed descriptor text, which a
                // hostile committer writes. Release builds do not check
                // overflow, so a plain `+=` would wrap and hand BM25 a tiny
                // length norm for an enormous document. Saturating keeps the
                // norm monotone in the claimed length.
                length = length.saturating_add(count);
                let total = counts.entry(term).or_insert(0);
                *total = total.saturating_add(count);
            }
            for (term, count) in counts {
                self.postings
                    .entry(term.to_owned())
                    .or_default()
                    .push((id, count));
            }
        }

        self.records.push(BuilderRecord {
            resource: record.resource.to_owned(),
            start: record.segment.start,
            end: record.segment.end,
            length,
            level: record.record.meta.level,
        });
    }

    fn build(self: Box<Self>) -> map_core::Result<Box<dyn Scorer>> {
        // Round-trips through the serialized form on purpose: the in-memory
        // rebuild path and the mmap'd path then exercise identical code, so a
        // bug in the encoding cannot hide behind a shortcut.
        let dimension = self.dimension.clone();
        let pack = Pack::from_bytes(self.finish(), &dimension).map_err(|e| {
            map_core::Error::io(
                std::path::PathBuf::from("<in-memory pack>"),
                std::io::Error::other(e),
            )
        })?;
        Ok(Box::new(pack))
    }
}

/// A memory-mapped pack.
///
/// Backed by an `mmap` where available, so opening it costs no parsing and no
/// copying — the kernel pages in only the bytes a query actually touches, and
/// concurrent processes share those pages through the page cache.
pub struct Pack {
    /// Which dimension this pack scores.
    ///
    /// Supplied by the caller rather than stored in the file: a pack lives at
    /// `cache/<dimension>.pack`, so the name is already carried by the path,
    /// and putting it in the header would be a second source of truth that
    /// could disagree with the first.
    dimension: String,
    bytes: Backing,
    records: usize,
    terms: usize,
    average_length: f32,
    off_records: usize,
    off_terms: usize,
    off_postings: usize,
    off_strings: usize,
}

pub(crate) enum Backing {
    Mapped(memmap2::Mmap),
    Owned(Vec<u8>),
}

impl std::ops::Deref for Backing {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Backing::Mapped(m) => m,
            Backing::Owned(v) => v,
        }
    }
}

/// A pack file mapped into memory, not yet parsed.
///
/// The split exists so a caller can decide whether to *trust* the bytes before
/// it interprets them. Parsing first meant bytes nothing vouched for still
/// reached the parser, and an unparseable substitute errored out instead of
/// being rebuilt. Both pack formats map through here, so there is one `mmap`
/// path and one fallback.
pub struct PackFile {
    bytes: Backing,
}

impl PackFile {
    /// Memory-map a pack file, without looking at its contents.
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;

        // SAFETY: mapping a file is unsound only if another process mutates it
        // underneath us. Packs are written atomically (temp file plus rename),
        // never mutated in place, so an open mapping always refers to a
        // complete, immutable file. Every field read from it later is
        // bounds-checked against the mapping length by `parse`.
        #[allow(unsafe_code)]
        let mapped = unsafe { memmap2::Mmap::map(&file) };

        Ok(PackFile {
            bytes: match mapped {
                Ok(m) => Backing::Mapped(m),
                // Some filesystems refuse mmap; reading is slower but correct.
                Err(_) => Backing::Owned(std::fs::read(path)?),
            },
        })
    }

    /// The whole pack image, mapped or read.
    ///
    /// Lets a caller hash exactly the bytes it is about to parse — reading the
    /// file a second time would leave a window in which the two differ.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub(crate) fn into_backing(self) -> Backing {
        self.bytes
    }
}

impl Pack {
    /// Memory-map a pack from disk, scoring `dimension`.
    pub fn open(path: &Path, dimension: &str) -> Result<Self, PackError> {
        let file = PackFile::open(path).map_err(|e| PackError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        Self::from_file(file, dimension)
    }

    /// Parse an already-mapped pack file, scoring `dimension`.
    pub fn from_file(file: PackFile, dimension: &str) -> Result<Self, PackError> {
        Self::parse(file.into_backing(), dimension)
    }

    /// Read a pack from bytes already in memory, scoring `dimension`.
    pub fn from_bytes(bytes: Vec<u8>, dimension: &str) -> Result<Self, PackError> {
        Self::parse(Backing::Owned(bytes), dimension)
    }

    fn parse(bytes: Backing, dimension: &str) -> Result<Self, PackError> {
        if bytes.len() < HEADER_LEN {
            return Err(PackError::Malformed("truncated header"));
        }
        if &bytes[0..8] != MAGIC {
            return Err(PackError::Malformed("bad magic"));
        }
        let found = u32(&bytes[8..12]);
        if found != VERSION {
            return Err(PackError::UnsupportedVersion {
                found,
                expected: VERSION,
            });
        }

        let records = u32(&bytes[12..16]) as usize;
        let terms = u32(&bytes[16..20]) as usize;
        let average_length = f32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
        let off_records = u64(&bytes[24..32]) as usize;
        let off_terms = u64(&bytes[32..40]) as usize;
        let off_postings = u64(&bytes[40..48]) as usize;
        let off_strings = u64(&bytes[48..56]) as usize;

        let pack = Pack {
            dimension: dimension.to_owned(),
            records,
            terms,
            average_length,
            off_records,
            off_terms,
            off_postings,
            off_strings,
            bytes,
        };

        // Bounds-check every section before any query can index into it.
        let record_end = off_records
            .checked_add(records.checked_mul(RECORD_LEN).ok_or(overflow())?)
            .ok_or(overflow())?;
        let term_end = off_terms
            .checked_add(terms.checked_mul(TERM_LEN).ok_or(overflow())?)
            .ok_or(overflow())?;
        if record_end > pack.bytes.len()
            || term_end > pack.bytes.len()
            || off_postings > pack.bytes.len()
            || off_strings > pack.bytes.len()
            || off_records > off_terms
            || off_terms > off_postings
            || off_postings > off_strings
        {
            return Err(PackError::Malformed("section out of bounds"));
        }
        Ok(pack)
    }

    /// Number of distinct terms.
    pub fn term_count(&self) -> usize {
        self.terms
    }

    fn record_length(&self, id: usize) -> f32 {
        let at = self.off_records + id * RECORD_LEN;
        u32(&self.bytes[at + 16..at + 20]) as f32
    }

    fn string(&self, offset: usize, len: usize) -> Option<&[u8]> {
        let start = self.off_strings.checked_add(offset)?;
        let end = start.checked_add(len)?;
        self.bytes.get(start..end)
    }

    fn term_at(&self, i: usize) -> Option<(&[u8], usize, usize)> {
        let at = self.off_terms + i * TERM_LEN;
        let entry = self.bytes.get(at..at + TERM_LEN)?;
        let term = self.string(u32(&entry[0..4]) as usize, u32(&entry[4..8]) as usize)?;
        Some((
            term,
            u32(&entry[8..12]) as usize,
            u32(&entry[12..16]) as usize,
        ))
    }

    /// Postings for a term, as `(record id, frequency)`.
    fn postings(&self, term: &str) -> Option<(usize, usize)> {
        // Terms are stored sorted, so this is a binary search over fixed-size
        // entries — no dictionary to build at load time.
        let (mut lo, mut hi) = (0usize, self.terms);
        while lo < hi {
            let mid = (lo + hi) / 2;
            let (candidate, offset, count) = self.term_at(mid)?;
            match candidate.cmp(term.as_bytes()) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some((offset, count)),
            }
        }
        None
    }
}

impl Stage for Pack {
    fn implementation(&self) -> &str {
        "bm25"
    }

    fn config(&self) -> String {
        format!("bm25:k1={K1},b={B}")
    }
}

impl Pack {
    /// Raw BM25 for every record containing a query term, plus the ceiling
    /// that normalizes it.
    ///
    /// Touches only records that contain a query term — the scalability
    /// property. Cost is proportional to the number of *matching* records, not
    /// to corpus size.
    fn raw_scores(
        &self,
        text: &str,
        corpus_stats: Option<&CorpusStats>,
    ) -> (BTreeMap<u32, f32>, f32) {
        let mut totals: BTreeMap<u32, f32> = BTreeMap::new();
        if self.records == 0 {
            return (totals, 1.0);
        }

        // Federated statistics when given, this pack's own otherwise. A single
        // index is the degenerate case where they are the same numbers, so the
        // two paths cannot disagree.
        let corpus = corpus_stats.map_or(self.records as f32, |g| g.records as f32);
        let average_length = corpus_stats.map_or(self.average_length, |g| g.average_length());

        let terms = deduplicated_terms(text);

        // The ceiling is over every query term the *federation* has, not just
        // the ones this pack holds. Computing it locally is what lets an index
        // missing a term return inflated scores, because `query_ceiling` drops
        // `df == 0` terms and a smaller divisor means a bigger quotient.
        let ceiling = match corpus_stats {
            Some(global) => {
                let dfs: Vec<u32> = terms
                    .iter()
                    .map(|t| global.document_frequency.get(t).copied().unwrap_or(0))
                    .collect();
                query_ceiling(corpus, &dfs)
            }
            None => {
                let dfs: Vec<u32> = terms
                    .iter()
                    .filter_map(|t| self.postings(t).map(|(_, count)| count as u32))
                    .collect();
                query_ceiling(corpus, &dfs)
            }
        };

        for term in &terms {
            let Some((offset, count)) = self.postings(term) else {
                continue;
            };
            // Weight by the federation's document frequency, so the same term
            // is worth the same everywhere. Falls back to the local count when
            // scoring alone.
            let df = corpus_stats
                .and_then(|g| g.document_frequency.get(term).copied())
                .unwrap_or(count as u32);
            let weight = idf(corpus, df as f32);

            for i in 0..count {
                let at = self.off_postings + (offset + i) * POSTING_LEN;
                let Some(entry) = self.bytes.get(at..at + POSTING_LEN) else {
                    break;
                };
                let id = u32(&entry[0..4]);
                let tf = u32(&entry[4..8]) as f32;
                if id as usize >= self.records {
                    continue;
                }

                let norm = 1.0 - B + B * self.record_length(id as usize) / average_length.max(1.0);
                *totals.entry(id).or_insert(0.0) += weight * (tf * (K1 + 1.0)) / (tf + K1 * norm);
            }
        }
        (totals, ceiling)
    }

    /// This pack's contribution to the federated statistics for `text`.
    fn stats_for(&self, text: &str) -> CorpusStats {
        let mut document_frequency = BTreeMap::new();
        for term in deduplicated_terms(text) {
            let df = self.postings(&term).map_or(0, |(_, count)| count as u32);
            document_frequency.insert(term, df);
        }
        CorpusStats {
            records: self.records as u64,
            total_length: self.average_length as f64 * self.records as f64,
            document_frequency,
        }
    }

    /// The query text this pack's dimension was asked for, if any.
    fn text_for<'a>(&self, query: &'a QueryBundle, dimensions: &[String]) -> Option<&'a str> {
        dimensions
            .iter()
            .find(|d| **d == self.dimension)
            .and_then(|d| query.get(d))
            .and_then(|f| f.text.as_deref())
    }
}

/// Query terms, each once, in first-seen order.
///
/// A repeated term would otherwise be weighted twice, in the score and in the
/// ceiling.
fn deduplicated_terms(text: &str) -> Vec<String> {
    let mut seen = BTreeSet::new();
    tokenize(text)
        .into_iter()
        .filter(|term| seen.insert(term.clone()))
        .collect()
}

impl Scorer for Pack {
    fn len(&self) -> usize {
        self.records
    }

    /// Walks postings, not records.
    ///
    /// `raw_scores` visits only records containing a query term, so the work is
    /// O(matching). Under [`CandidateScope::All`] that is the whole cost —
    /// every match is emitted. Under `Ids` a set is built once from the scope
    /// and each match is checked against it, O(matching·log|scope|); only that
    /// path pays for the scope, and the common `All` path pays nothing.
    fn corpus_stats(&self, query: &QueryBundle, dimensions: &[String]) -> Option<CorpusStats> {
        self.text_for(query, dimensions)
            .map(|text| self.stats_for(text))
    }

    fn score(
        &self,
        scope: CandidateScope<'_>,
        query: &QueryBundle,
        dimensions: &[String],
        corpus: Option<&CorpusStats>,
    ) -> map_core::Result<Vec<(u32, DimensionScores)>> {
        let Some(text) = self.text_for(query, dimensions) else {
            return Ok(Vec::new());
        };
        let (totals, ceiling) = self.raw_scores(text, corpus);

        // `None` = every match is eligible (All); `Some(set)` = restrict to it.
        let eligible: Option<BTreeSet<u32>> = match scope {
            CandidateScope::All => None,
            CandidateScope::Ids(ids) => Some(ids.iter().copied().collect()),
        };
        let mut out = Vec::new();
        for (id, raw) in totals {
            if eligible.as_ref().is_some_and(|set| !set.contains(&id)) {
                continue;
            }
            let mut scores = DimensionScores::new();
            scores.insert(self.dimension.clone(), (raw / ceiling).min(1.0));
            out.push((id, scores));
        }
        Ok(out)
    }

    fn located(&self, id: u32) -> Option<Located<'_>> {
        let id = id as usize;
        if id >= self.records {
            return None;
        }
        let at = self.off_records + id * RECORD_LEN;
        let entry = &self.bytes[at..at + RECORD_LEN];
        let offset = u32(&entry[0..4]) as usize;
        let len = u32(&entry[4..8]) as usize;
        let resource = std::str::from_utf8(self.string(offset, len)?).ok()?;
        Some(Located {
            resource,
            start: u32(&entry[8..12]),
            end: u32(&entry[12..16]),
            level: u32(&entry[20..24]) as u16,
        })
    }
}

fn overflow() -> PackError {
    PackError::Malformed("offset overflow")
}

fn u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn u64(bytes: &[u8]) -> u64 {
    u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::{encode_frequencies, Bm25Builder, StructuralClassifier};
    use map_core::{Classifier, ClassifyBatch, Content, Segment};
    use map_format::{Record, RecordKind, RecordMeta};

    /// A segment record as the structural classifier would emit it.
    fn record_of(text: &str) -> Record {
        let content = Content {
            key: "t.rs".into(),
            text: text.into(),
        };
        let segments = [Segment {
            start: 0,
            end: content.text.len() as u32,
        }];
        let items: Vec<Vec<&str>> = segments.iter().map(|s| vec![s.slice(&content)]).collect();
        let dims = ["lexical".to_owned()];
        StructuralClassifier
            .classify(&ClassifyBatch {
                items: &items,
                dimensions: &dims,
                level: 0,
            })
            .unwrap()[0]["lexical"]
            .clone()
    }

    fn leveled(descriptor: &str, level: u16) -> Record {
        Record {
            descriptor: Some(encode_frequencies(&tokenize(descriptor))),
            tensor: None,
            meta: RecordMeta {
                kind: if level == 0 {
                    RecordKind::Segment
                } else {
                    RecordKind::Cluster
                },
                dimension: "lexical".into(),
                level,
                children: vec![],
            },
        }
    }

    fn corpus() -> Vec<(&'static str, Record)> {
        vec![
            ("src/config.rs", record_of("fn parse_config(path: &Path)")),
            (
                "src/session.rs",
                record_of("fn refresh_token(session: &Session)"),
            ),
            ("src/log.rs", record_of("struct Logger { level: Level }")),
        ]
    }

    /// Generic over the builder so the pack and the reference are filled by
    /// literally the same code — otherwise "they agree" could just mean the
    /// two test harnesses agree.
    fn fill<S: ScorerBuilder + ?Sized>(
        builder: &mut S,
        entries: &[(&str, Record)],
        span: impl Fn(usize) -> u32,
    ) {
        for (i, (resource, record)) in entries.iter().enumerate() {
            builder.push(&IndexedRecord {
                id: i as u32,
                resource,
                segment: Segment {
                    start: 0,
                    end: span(i),
                },
                record,
            });
        }
    }

    fn dims() -> Vec<String> {
        vec!["lexical".to_owned()]
    }

    fn q(text: &str) -> QueryBundle {
        let mut bundle = QueryBundle::new();
        bundle.insert("lexical".to_owned(), map_core::QueryField::text(text));
        bundle
    }

    /// Score every record and rank, as the retriever does.
    fn run(scorer: &dyn Scorer, text: &str) -> Vec<(u32, f32)> {
        let scored = scorer
            .score(CandidateScope::All, &q(text), &dims(), None)
            .unwrap();
        let mut out: Vec<(u32, f32)> = scored
            .into_iter()
            .filter_map(|(id, s)| s.get("lexical").map(|v| (id, *v)))
            .collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        out
    }

    fn build() -> Pack {
        let mut builder = PackBuilder::new();
        fill(&mut builder, &corpus(), |i| 100 + i as u32);
        Pack::from_bytes(builder.finish(), "lexical").unwrap()
    }

    fn build_bytes() -> Vec<u8> {
        let mut b = PackBuilder::new();
        fill(&mut b, &corpus(), |_| 10);
        b.finish()
    }

    /// The in-memory reference over the same corpus, as an oracle.
    fn reference() -> Box<dyn Scorer> {
        let mut builder = Box::new(Bm25Builder::new());
        fill(&mut *builder, &corpus(), |i| 100 + i as u32);
        builder.build().unwrap()
    }

    #[test]
    fn roundtrips_records() {
        let pack = build();
        assert_eq!(pack.len(), 3);
        assert_eq!(pack.located(1).unwrap().resource, "src/session.rs");
        assert_eq!(pack.located(1).unwrap().end, 101);
        assert!(pack.located(9).is_none());
    }

    #[test]
    fn ranks_the_same_as_the_unpacked_scorer() {
        // The pack is an acceleration structure, not a different algorithm.
        // Two implementations of Okapi BM25 exist; this is what stops them
        // drifting apart unnoticed.
        let pack = build();
        let reference = reference();

        for query in ["refresh token", "parse config", "logger level"] {
            let packed: Vec<u32> = run(&pack, query).into_iter().map(|(i, _)| i).collect();
            let plain: Vec<u32> = run(reference.as_ref(), query)
                .into_iter()
                .map(|(i, _)| i)
                .collect();
            assert_eq!(packed, plain, "ranking diverged for {query:?}");
        }
    }

    #[test]
    fn scores_match_the_unpacked_scorer() {
        let pack = build();
        let reference = reference();

        let packed = run(&pack, "refresh token");
        let plain = run(reference.as_ref(), "refresh token");
        assert_eq!(packed.len(), plain.len());
        for ((_, a), (_, b)) in packed.iter().zip(plain.iter()) {
            assert!((a - b).abs() < 1e-4, "score drift: {a} vs {b}");
        }
    }

    #[test]
    fn every_score_is_a_normalized_relevance() {
        let pack = build();
        for query in ["refresh token", "parse", "logger level path session"] {
            for (id, score) in run(&pack, query) {
                assert!(
                    (0.0..=1.0).contains(&score),
                    "score {score} for record {id} out of range on {query:?}"
                );
            }
        }
    }

    #[test]
    fn only_matching_records_are_touched() {
        // The scalability claim: a term in one document must not walk the
        // whole corpus.
        let mut builder = PackBuilder::new();
        for i in 0..1000 {
            let record = if i == 417 {
                record_of("fn quiesce_the_reactor()")
            } else {
                record_of("fn ordinary_helper(value: u32)")
            };
            builder.push(&IndexedRecord {
                id: i,
                resource: "src/big.rs",
                segment: Segment {
                    start: i,
                    end: i + 1,
                },
                record: &record,
            });
        }
        let pack = Pack::from_bytes(builder.finish(), "lexical").unwrap();

        let hits = run(&pack, "quiesce");
        assert_eq!(hits.len(), 1, "only the one matching record should score");
        assert_eq!(hits[0].0, 417);
    }

    #[test]
    fn unknown_terms_return_nothing() {
        assert!(run(&build(), "nonexistent_symbol").is_empty());
    }

    #[test]
    fn empty_pack_is_valid() {
        let pack = Pack::from_bytes(PackBuilder::new().finish(), "lexical").unwrap();
        assert!(pack.is_empty());
        assert!(run(&pack, "anything").is_empty());
    }

    #[test]
    fn unreadable_records_still_consume_an_id() {
        // Pack ids are positional, so a skipped record would renumber every
        // later one and a candidate list from another dimension would then
        // address the wrong spans.
        let mut builder = PackBuilder::new();
        let semantic = Record {
            descriptor: None,
            tensor: Some(map_format::Tensor {
                dtype: map_format::DType::F32,
                shape: vec![2],
                data: vec![0; 8],
            }),
            meta: RecordMeta {
                kind: RecordKind::Segment,
                dimension: "descriptive".into(),
                level: 0,
                children: vec![],
            },
        };
        builder.push(&IndexedRecord {
            id: 0,
            resource: "src/a.rs",
            segment: Segment { start: 0, end: 1 },
            record: &semantic,
        });
        builder.push(&IndexedRecord {
            id: 1,
            resource: "src/a.rs",
            segment: Segment { start: 1, end: 2 },
            record: &record_of("fn quiesce_the_reactor()"),
        });
        let pack = Pack::from_bytes(builder.finish(), "lexical").unwrap();

        assert_eq!(pack.len(), 2, "the unreadable record must keep its slot");
        let hits = run(&pack, "quiesce");
        assert_eq!(
            hits[0].0, 1,
            "the readable record must keep the id the driver gave it"
        );
    }

    #[test]
    fn fabric_level_survives_the_round_trip() {
        // Retrieval zoom is a filter on level, so losing it in the mapped form
        // would silently disable the whole overview path.
        let mut builder = PackBuilder::new();
        for (i, level) in [0u16, 1, 7].iter().enumerate() {
            let record = leveled("cluster label alpha", *level);
            builder.push(&IndexedRecord {
                id: i as u32,
                resource: "src/a.rs",
                segment: Segment {
                    start: i as u32,
                    end: i as u32 + 1,
                },
                record: &record,
            });
        }
        let pack = Pack::from_bytes(builder.finish(), "lexical").unwrap();
        assert_eq!(pack.located(0).unwrap().level, 0);
        assert_eq!(pack.located(1).unwrap().level, 1);
        assert_eq!(pack.located(2).unwrap().level, 7);
    }

    #[test]
    fn building_is_deterministic() {
        assert_eq!(build_bytes(), build_bytes());
    }

    #[test]
    fn an_absurd_term_count_cannot_wrap_the_length_norm() {
        // Descriptor text is committed data a hostile clone writes. Counts that
        // sum past u32 must saturate rather than wrap: a wrapped length means a
        // tiny norm, which is an arbitrarily inflated BM25 score. Debug builds
        // check arithmetic, so this also pins the release behaviour.
        let hostile = Record {
            descriptor: Some("a:4294967295 b:1".to_owned()),
            tensor: None,
            meta: RecordMeta {
                kind: RecordKind::Segment,
                dimension: "lexical".into(),
                level: 0,
                children: vec![],
            },
        };
        let entries = vec![
            ("src/hostile.rs", hostile),
            ("src/config.rs", record_of("fn parse_config(path: &Path)")),
        ];

        let mut builder = PackBuilder::new();
        fill(&mut builder, &entries, |_| 10);
        let pack = Pack::from_bytes(builder.finish(), "lexical").unwrap();

        let mut plain = Box::new(Bm25Builder::new());
        fill(&mut *plain, &entries, |_| 10);
        let reference = plain.build().unwrap();

        for query in ["a", "b", "parse config"] {
            let packed = run(&pack, query);
            let oracle = run(reference.as_ref(), query);
            assert_eq!(
                packed.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
                oracle.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
                "pack and reference diverged for {query:?}"
            );
            for ((id, a), (_, b)) in packed.iter().zip(oracle.iter()) {
                assert!(a.is_finite(), "record {id} scored {a} on {query:?}");
                assert!(
                    (0.0..=1.0).contains(a),
                    "score {a} for record {id} out of range on {query:?}"
                );
                assert!((a - b).abs() < 1e-4, "score drift: {a} vs {b}");
            }
        }
    }

    #[test]
    fn resource_keys_are_interned() {
        // One key per segment would dominate the pack on a large file.
        let plain = leveled("term", 0);
        let mut many = PackBuilder::new();
        let mut few = PackBuilder::new();
        for i in 0..200u32 {
            let span = Segment {
                start: i,
                end: i + 1,
            };
            many.push(&IndexedRecord {
                id: i,
                resource: "src/very/long/path/to/a/file.rs",
                segment: span,
                record: &plain,
            });
            few.push(&IndexedRecord {
                id: i,
                resource: "x",
                segment: span,
                record: &plain,
            });
        }
        let overhead = many.finish().len() - few.finish().len();
        assert!(
            overhead < 200,
            "resource key appears to be stored per record ({overhead} bytes over 200 records)"
        );
    }

    #[test]
    fn rejects_corrupt_packs() {
        assert!(Pack::from_bytes(vec![], "lexical").is_err());
        assert!(Pack::from_bytes(b"NOPENOPE".to_vec(), "lexical").is_err());

        let mut bytes = build_bytes();
        bytes[8] = 99; // version
        assert!(Pack::from_bytes(bytes, "lexical").is_err());

        let mut bytes = build_bytes();
        bytes[12] = 0xff; // absurd record count
        bytes[13] = 0xff;
        bytes[14] = 0xff;
        assert!(Pack::from_bytes(bytes, "lexical").is_err());
    }
}
