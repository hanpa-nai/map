//! The dense dimension's acceleration structure: an mmap-able vector store.
//!
//! The exact analog of [`crate::pack`] for tensors instead of postings. The
//! committed form of a dense dimension is its per-segment tensor objects
//! (portable, verifiable); this pack is the **derived cache** — gitignored,
//! rebuildable, and memory-mapped so a query re-embeds nothing and parses no
//! JSON. It is what makes a dense dimension load like the lexical one instead
//! of re-running the embedder over the whole corpus every time.
//!
//! # Layout
//!
//! Little-endian, sections 8-byte aligned:
//!
//! ```text
//! header    64 B     magic, version, record count, dim, section offsets
//! records   n × 24 B resource span into strings, byte span, fabric level
//! vectors   n × dim × 4 B   row-major f32, one row per record
//! strings            interned resource keys
//! ```
//!
//! Records keep insertion order so ids stay positional, matching the shared
//! segmentation — the same invariant the lexical pack relies on.
//!
//! # Scoring
//!
//! Vectors are stored as the embedder produced them: L2-normalized. So cosine
//! similarity is just the dot product, and a query is scored by one pass of
//! dot products over the candidate rows — O(candidates · dim), no matrix
//! library. Negative similarities are dropped, matching the reference
//! `CosineScorer`, so a matched score is always in `[0, 1]`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use map_core::{
    CandidateScope, CorpusStats, DimensionScores, IndexedRecord, Located, QueryBundle, Scorer,
    ScorerBuilder, Stage,
};

// The mapped-file backing is shared with the lexical pack so both formats have
// exactly one `mmap` path and one non-mmap fallback between them.
use crate::pack::{Backing, PackFile};

const MAGIC: &[u8; 8] = b"MAPDENS\0";
const VERSION: u32 = 1;
const HEADER_LEN: usize = 64;
const RECORD_LEN: usize = 24;

/// Errors from reading a dense pack.
#[derive(Debug, thiserror::Error)]
pub enum DensePackError {
    #[error("io error at {path}: {source}")]
    Io {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("malformed dense pack: {0}")]
    Malformed(&'static str),
    /// Written by a different version of this format; see
    /// [`PackError::UnsupportedVersion`](crate::pack::PackError::UnsupportedVersion).
    #[error("dense pack was written by format version {found}; this build writes {expected}")]
    UnsupportedVersion { found: u32, expected: u32 },
}

impl DensePackError {
    /// Whether this pack can simply be rebuilt from the objects.
    pub fn is_rebuildable(&self) -> bool {
        matches!(self, DensePackError::UnsupportedVersion { .. })
    }
}

#[derive(Debug)]
struct BuilderRecord {
    resource: String,
    start: u32,
    end: u32,
    level: u16,
    /// The stored vector, or `None` for a record this dimension cannot embed.
    vector: Option<Vec<f32>>,
}

/// Accumulates tensor records and emits dense-pack bytes.
#[derive(Debug, Default)]
pub struct DensePackBuilder {
    dimension: String,
    dim: Option<usize>,
    records: Vec<BuilderRecord>,
}

impl DensePackBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    fn vector_of(record: &IndexedRecord<'_>) -> Option<Vec<f32>> {
        let tensor = record.record.tensor.as_ref()?;
        if tensor.dtype != map_format::DType::F32 || tensor.validate().is_err() {
            return None;
        }
        Some(
            tensor
                .data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
        )
    }

    /// Serialize the pack.
    ///
    /// Records without a vector are written as a zero row so ids stay
    /// positional; a zero vector scores 0 against any query, which is the
    /// correct "unmatchable" behaviour.
    pub fn finish(self) -> Vec<u8> {
        let dim = self.dim.unwrap_or(0);

        let mut strings: Vec<u8> = Vec::new();
        let mut interned: BTreeMap<&str, (u32, u32)> = BTreeMap::new();

        let mut record_bytes = Vec::with_capacity(self.records.len() * RECORD_LEN);
        let mut vector_bytes = Vec::with_capacity(self.records.len() * dim * 4);
        let zeros = vec![0.0f32; dim];

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
            record_bytes.extend_from_slice(&(record.level as u32).to_le_bytes());
            record_bytes.extend_from_slice(&0u32.to_le_bytes()); // reserved

            let row = record.vector.as_deref().unwrap_or(&zeros);
            // A vector of the wrong width would corrupt the matrix stride; pad
            // or truncate to `dim` so the section stays rectangular.
            for i in 0..dim {
                let x = row.get(i).copied().unwrap_or(0.0);
                vector_bytes.extend_from_slice(&x.to_le_bytes());
            }
        }

        let off_records = HEADER_LEN as u64;
        let off_vectors = off_records + record_bytes.len() as u64;
        let off_strings = off_vectors + vector_bytes.len() as u64;

        let mut out = Vec::with_capacity(off_strings as usize + strings.len());
        out.extend_from_slice(MAGIC); //                                   0..8
        out.extend_from_slice(&VERSION.to_le_bytes()); //                  8..12
        out.extend_from_slice(&(self.records.len() as u32).to_le_bytes()); // 12..16
        out.extend_from_slice(&(dim as u32).to_le_bytes()); //            16..20
        out.extend_from_slice(&[0u8; 4]); // reserved                     20..24
        out.extend_from_slice(&off_records.to_le_bytes()); //             24..32
        out.extend_from_slice(&off_vectors.to_le_bytes()); //             32..40
        out.extend_from_slice(&off_strings.to_le_bytes()); //             40..48
        out.extend_from_slice(&[0u8; 16]); // reserved                    48..64
        debug_assert_eq!(out.len(), HEADER_LEN);

        out.extend_from_slice(&record_bytes);
        out.extend_from_slice(&vector_bytes);
        out.extend_from_slice(&strings);
        out
    }
}

impl Stage for DensePackBuilder {
    fn implementation(&self) -> &str {
        "cosine"
    }

    fn config(&self) -> String {
        "cosine:v1".to_owned()
    }
}

impl ScorerBuilder for DensePackBuilder {
    fn push(&mut self, record: &IndexedRecord<'_>) {
        debug_assert_eq!(
            record.id as usize,
            self.records.len(),
            "dense pack ids are positional; push every record of a dimension in id order"
        );
        if self.dimension.is_empty() {
            self.dimension = record.record.meta.dimension.clone();
        }

        let vector = Self::vector_of(record);
        if let Some(v) = &vector {
            // The first vector fixes the width for the whole matrix.
            self.dim.get_or_insert(v.len());
        }

        self.records.push(BuilderRecord {
            resource: record.resource.to_owned(),
            start: record.segment.start,
            end: record.segment.end,
            level: record.record.meta.level,
            vector,
        });
    }

    fn build(self: Box<Self>) -> map_core::Result<Box<dyn Scorer>> {
        let dimension = self.dimension.clone();
        let pack = DensePack::from_bytes(self.finish(), &dimension).map_err(|e| {
            map_core::Error::io(
                std::path::PathBuf::from("<in-memory dense pack>"),
                std::io::Error::other(e),
            )
        })?;
        Ok(Box::new(pack))
    }
}

/// A memory-mapped dense pack.
pub struct DensePack {
    dimension: String,
    bytes: Backing,
    records: usize,
    dim: usize,
    off_records: usize,
    off_vectors: usize,
    off_strings: usize,
}

impl DensePack {
    /// Memory-map a dense pack from disk, scoring `dimension`.
    pub fn open(path: &Path, dimension: &str) -> Result<Self, DensePackError> {
        let file = PackFile::open(path).map_err(|e| DensePackError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        Self::from_file(file, dimension)
    }

    /// Parse an already-mapped pack file, scoring `dimension`.
    pub fn from_file(file: PackFile, dimension: &str) -> Result<Self, DensePackError> {
        Self::parse(file.into_backing(), dimension)
    }

    /// Read a dense pack from in-memory bytes, scoring `dimension`.
    pub fn from_bytes(bytes: Vec<u8>, dimension: &str) -> Result<Self, DensePackError> {
        Self::parse(Backing::Owned(bytes), dimension)
    }

    fn parse(bytes: Backing, dimension: &str) -> Result<Self, DensePackError> {
        if bytes.len() < HEADER_LEN {
            return Err(DensePackError::Malformed("truncated header"));
        }
        if &bytes[0..8] != MAGIC {
            return Err(DensePackError::Malformed("bad magic"));
        }
        let found = u32(&bytes[8..12]);
        if found != VERSION {
            return Err(DensePackError::UnsupportedVersion {
                found,
                expected: VERSION,
            });
        }
        let records = u32(&bytes[12..16]) as usize;
        let dim = u32(&bytes[16..20]) as usize;
        let off_records = u64(&bytes[24..32]) as usize;
        let off_vectors = u64(&bytes[32..40]) as usize;
        let off_strings = u64(&bytes[40..48]) as usize;

        let record_end = records
            .checked_mul(RECORD_LEN)
            .and_then(|n| n.checked_add(off_records))
            .ok_or(DensePackError::Malformed("record section overflow"))?;
        let vector_end = records
            .checked_mul(dim)
            .and_then(|n| n.checked_mul(4))
            .and_then(|n| n.checked_add(off_vectors))
            .ok_or(DensePackError::Malformed("vector section overflow"))?;

        if off_records > off_vectors
            || off_vectors > off_strings
            || record_end > off_vectors
            || vector_end > off_strings
            || off_strings > bytes.len()
        {
            return Err(DensePackError::Malformed("section bounds inconsistent"));
        }

        let pack = DensePack {
            dimension: dimension.to_owned(),
            bytes,
            records,
            dim,
            off_records,
            off_vectors,
            off_strings,
        };
        Ok(pack)
    }

    /// Embedding width.
    pub fn dim(&self) -> usize {
        self.dim
    }

    fn string(&self, offset: usize, len: usize) -> Option<&[u8]> {
        let start = self.off_strings.checked_add(offset)?;
        let end = start.checked_add(len)?;
        self.bytes.get(start..end)
    }

    /// The stored vector for one record id.
    fn vector(&self, id: usize) -> Option<&[u8]> {
        let row = id.checked_mul(self.dim)?.checked_mul(4)?;
        let at = self.off_vectors.checked_add(row)?;
        self.bytes.get(at..at + self.dim * 4)
    }

    /// The query text this pack's dimension was asked for, decoded to f32.
    fn query_vector(&self, query: &QueryBundle, dimensions: &[String]) -> Option<Vec<f32>> {
        let field = dimensions
            .iter()
            .find(|d| **d == self.dimension)
            .and_then(|d| query.get(d))?;
        let tensor = field.tensor.as_ref()?;
        if tensor.dtype != map_format::DType::F32 {
            return None;
        }
        let v: Vec<f32> = tensor
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        (v.len() == self.dim).then_some(v)
    }

    /// Score one record's stored vector against the query, pushing a hit if the
    /// cosine is positive. Shared by both scopes so the math lives in one place.
    fn score_one(&self, id: u32, q: &[f32], out: &mut Vec<(u32, DimensionScores)>) {
        if id as usize >= self.records {
            return;
        }
        let Some(row) = self.vector(id as usize) else {
            return;
        };
        // Both sides L2-normalized, so the dot product is the cosine.
        let dot: f32 = row
            .as_chunks::<4>()
            .0
            .iter()
            .zip(q)
            .map(|(c, qi)| f32::from_le_bytes(*c) * qi)
            .sum();
        if dot > 0.0 {
            let mut scores = DimensionScores::new();
            scores.insert(self.dimension.clone(), dot.min(1.0));
            out.push((id, scores));
        }
    }
}

impl Stage for DensePack {
    fn implementation(&self) -> &str {
        "cosine"
    }

    fn config(&self) -> String {
        "cosine:v1".to_owned()
    }
}

impl Scorer for DensePack {
    fn len(&self) -> usize {
        self.records
    }

    fn score(
        &self,
        scope: CandidateScope<'_>,
        query: &QueryBundle,
        dimensions: &[String],
        _corpus: Option<&CorpusStats>,
    ) -> map_core::Result<Vec<(u32, DimensionScores)>> {
        let Some(q) = self.query_vector(query, dimensions) else {
            return Ok(Vec::new());
        };
        if self.dim == 0 {
            return Ok(Vec::new());
        }

        // Dense scoring is inherently O(records) — every stored vector must be
        // compared to the query — so `All` iterates the range directly, and only
        // `Ids` pays to materialize a set (deduped and ascending, as the output
        // contract requires).
        let mut out = Vec::new();
        match scope {
            CandidateScope::All => {
                for id in 0..self.records as u32 {
                    self.score_one(id, &q, &mut out);
                }
            }
            CandidateScope::Ids(ids) => {
                for id in ids.iter().copied().collect::<BTreeSet<u32>>() {
                    self.score_one(id, &q, &mut out);
                }
            }
        }
        Ok(out)
    }

    fn located(&self, id: u32) -> Option<Located<'_>> {
        let id = id as usize;
        if id >= self.records {
            return None;
        }
        let at = self.off_records + id * RECORD_LEN;
        let entry = self.bytes.get(at..at + RECORD_LEN)?;
        let offset = u32(&entry[0..4]) as usize;
        let len = u32(&entry[4..8]) as usize;
        let resource = std::str::from_utf8(self.string(offset, len)?).ok()?;
        Some(Located {
            resource,
            start: u32(&entry[8..12]),
            end: u32(&entry[12..16]),
            level: u32(&entry[16..20]) as u16,
        })
    }
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
    use crate::CosineBuilder;
    use map_core::{QueryField, Segment};
    use map_format::{DType, Record, RecordKind, RecordMeta, Tensor};

    /// A unit-length record vector from raw components.
    fn record(values: &[f32]) -> Record {
        let norm = values.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        let mut data = Vec::new();
        for x in values {
            data.extend_from_slice(&(x / norm).to_le_bytes());
        }
        Record {
            descriptor: None,
            tensor: Some(Tensor {
                dtype: DType::F32,
                shape: vec![values.len() as u32],
                data,
            }),
            meta: RecordMeta {
                kind: RecordKind::Segment,
                dimension: "semantic".into(),
                level: 0,
                children: vec![],
            },
        }
    }

    fn query_bundle(values: &[f32]) -> QueryBundle {
        let norm = values.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        let mut data = Vec::new();
        for x in values {
            data.extend_from_slice(&(x / norm).to_le_bytes());
        }
        let mut b = QueryBundle::new();
        b.insert(
            "semantic".to_owned(),
            QueryField {
                text: None,
                tensor: Some(Tensor {
                    dtype: DType::F32,
                    shape: vec![values.len() as u32],
                    data,
                }),
            },
        );
        b
    }

    fn fill<S: ScorerBuilder + ?Sized>(builder: &mut S, rows: &[Record]) {
        for (i, r) in rows.iter().enumerate() {
            builder.push(&IndexedRecord {
                id: i as u32,
                resource: "t.rs",
                segment: Segment {
                    start: i as u32,
                    end: i as u32 + 1,
                },
                record: r,
            });
        }
    }

    fn corpus() -> Vec<Record> {
        vec![
            record(&[1.0, 0.0, 0.0]),
            record(&[0.0, 1.0, 0.0]),
            record(&[0.9, 0.1, 0.0]),
        ]
    }

    fn dims() -> Vec<String> {
        vec!["semantic".to_owned()]
    }

    fn run(scorer: &dyn Scorer, q: &QueryBundle) -> Vec<(u32, f32)> {
        let mut out: Vec<(u32, f32)> = scorer
            .score(CandidateScope::All, q, &dims(), None)
            .unwrap()
            .into_iter()
            .filter_map(|(id, m)| m.get("semantic").map(|s| (id, *s)))
            .collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        out
    }

    fn pack() -> DensePack {
        let mut b = DensePackBuilder::new();
        fill(&mut b, &corpus());
        DensePack::from_bytes(b.finish(), "semantic").unwrap()
    }

    fn reference() -> Box<dyn Scorer> {
        let mut b = Box::new(CosineBuilder::new());
        fill(&mut *b, &corpus());
        b.build().unwrap()
    }

    #[test]
    fn ranks_the_same_as_the_reference_cosine_scorer() {
        // Two implementations of cosine over the same vectors; this is what
        // keeps the mmap'd pack honest against the in-memory oracle.
        let pack = pack();
        let reference = reference();
        for q in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.7, 0.7, 0.0]] {
            let qb = query_bundle(&q);
            let packed: Vec<u32> = run(&pack, &qb).into_iter().map(|(i, _)| i).collect();
            let plain: Vec<u32> = run(reference.as_ref(), &qb)
                .into_iter()
                .map(|(i, _)| i)
                .collect();
            assert_eq!(packed, plain, "ranking diverged for {q:?}");
        }
    }

    #[test]
    fn scores_match_the_reference_within_tolerance() {
        let pack = pack();
        let reference = reference();
        let qb = query_bundle(&[1.0, 0.0, 0.0]);
        let packed = run(&pack, &qb);
        let plain = run(reference.as_ref(), &qb);
        assert_eq!(packed.len(), plain.len());
        for ((_, a), (_, b)) in packed.iter().zip(plain.iter()) {
            assert!((a - b).abs() < 1e-5, "score drift {a} vs {b}");
        }
    }

    #[test]
    fn the_aligned_vector_scores_highest() {
        let hits = run(&pack(), &query_bundle(&[1.0, 0.0, 0.0]));
        assert_eq!(hits[0].0, 0, "the (1,0,0) record should win");
        assert!((hits[0].1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn every_score_is_in_unit_range() {
        for q in [[1.0, 0.0, 0.0], [0.3, 0.4, 0.9], [-1.0, 2.0, 0.5]] {
            for (_, s) in run(&pack(), &query_bundle(&q)) {
                assert!((0.0..=1.0).contains(&s), "score {s} out of range");
            }
        }
    }

    #[test]
    fn a_missing_tensor_keeps_its_id_and_scores_zero() {
        // Positional ids: a record with no vector still occupies a slot, so a
        // candidate list from another dimension addresses the right rows.
        let mut b = DensePackBuilder::new();
        let mut rows = corpus();
        rows.insert(
            1,
            Record {
                descriptor: Some("no tensor here".into()),
                tensor: None,
                meta: RecordMeta {
                    kind: RecordKind::Segment,
                    dimension: "semantic".into(),
                    level: 0,
                    children: vec![],
                },
            },
        );
        fill(&mut b, &rows);
        let pack = DensePack::from_bytes(b.finish(), "semantic").unwrap();
        assert_eq!(pack.len(), 4);
        // id 1 (the tensorless record) must never appear in results.
        let hits = run(&pack, &query_bundle(&[0.0, 1.0, 0.0]));
        assert!(hits.iter().all(|(id, _)| *id != 1));
    }

    #[test]
    fn round_trips_locations_and_dim() {
        let pack = pack();
        assert_eq!(pack.dim(), 3);
        assert_eq!(pack.located(2).unwrap().resource, "t.rs");
        assert_eq!(pack.located(2).unwrap().end, 3);
        assert!(pack.located(9).is_none());
    }

    #[test]
    fn fabric_level_survives_the_round_trip() {
        // Cluster records ride the dense pack alongside segments; zoom is a
        // filter on this level, so losing it would silently disable overview.
        let mut b = DensePackBuilder::new();
        for (i, level) in [0u16, 1, 5].iter().enumerate() {
            let mut r = record(&[1.0, 0.0, 0.0]);
            r.meta.level = *level;
            r.meta.kind = if *level == 0 {
                RecordKind::Segment
            } else {
                RecordKind::Cluster
            };
            b.push(&IndexedRecord {
                id: i as u32,
                resource: "t.rs",
                segment: Segment {
                    start: i as u32,
                    end: i as u32 + 1,
                },
                record: &r,
            });
        }
        let pack = DensePack::from_bytes(b.finish(), "semantic").unwrap();
        assert_eq!(pack.located(0).unwrap().level, 0);
        assert_eq!(pack.located(1).unwrap().level, 1);
        assert_eq!(pack.located(2).unwrap().level, 5);
    }

    #[test]
    fn building_is_deterministic() {
        let bytes = || {
            let mut b = DensePackBuilder::new();
            fill(&mut b, &corpus());
            b.finish()
        };
        assert_eq!(bytes(), bytes());
    }

    #[test]
    fn rejects_corrupt_packs() {
        let good = || {
            let mut b = DensePackBuilder::new();
            fill(&mut b, &corpus());
            b.finish()
        };
        assert!(DensePack::from_bytes(vec![], "semantic").is_err());
        assert!(DensePack::from_bytes(b"NOPENOPE".to_vec(), "semantic").is_err());
        let mut bytes = good();
        bytes[8] = 99; // version
        assert!(DensePack::from_bytes(bytes, "semantic").is_err());
        let mut bytes = good();
        bytes[12] = 0xff; // absurd record count
        bytes[13] = 0xff;
        bytes[14] = 0xff;
        assert!(DensePack::from_bytes(bytes, "semantic").is_err());
    }
}
