//! Stage traits, the dimension bundle, and pipeline orchestration.
//!
//! # One record type, everywhere
//!
//! Every stage that produces something searchable produces a
//! [`map_format::Record`]. The fabricator's clusters are therefore the *same*
//! type as segments, so one uniform search space with one scoring path is a
//! property the types enforce rather than a claim the prose makes. A stage
//! consumes whichever payload it understands and ignores the rest.
//!
//! # The bundle, not a fan-out
//!
//! Dimensions are not parallel pipeline branches. Every stage receives the full
//! set of dimensions it applies to and handles them together, because the
//! classifier is the weightiest stage and classifying one segment across N
//! dimensions must be **one call, not N**. [`group_by_implementation`] is what
//! makes that concrete.
//!
//! ```text
//! discover → preprocess → segment → classify → embed → fabricate → retrieve
//! ```
//!
//! `discover` and `preprocess` carry no dimension data: they are objective
//! facts about a resource.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use map_format::{Record, Tensor};

pub mod error;
pub mod payload;

pub use error::{Error, Result};
pub use payload::{
    decode_tensor_payload, encode_tensor_payload, DescriptorPayload, SegmentsPayload, TensorPayload,
};

/// A resource the discoverer found, before any content is read.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Resource {
    /// Canonical key: root-relative, forward slashes, NFC (spec §6.1).
    pub key: String,
    pub size: u64,
}

/// Decoded, normalized content of one resource.
///
/// Every offset produced downstream indexes into `text`, not into the bytes on
/// disk — a consumer resolving a span back to a file must re-normalize first,
/// or it lands in the wrong place on a CRLF checkout.
#[derive(Clone, Debug)]
pub struct Content {
    pub key: String,
    pub text: String,
}

/// A span of normalized content.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Segment {
    pub start: u32,
    pub end: u32,
}

impl Segment {
    pub fn slice<'a>(&self, content: &'a Content) -> &'a str {
        &content.text[self.start as usize..self.end as usize]
    }
}

/// Records produced for one segment, keyed by dimension name.
pub type DimensionRecords = BTreeMap<String, Record>;

/// Tensors produced for one segment, keyed by dimension name.
pub type DimensionTensors = BTreeMap<String, Tensor>;

/// Identity and configuration common to every stage implementation.
pub trait Stage {
    /// Implementation name, matching the `impl` field in config.
    fn implementation(&self) -> &str;

    /// Configuration fingerprint input; folded into object keys.
    ///
    /// Must cover every setting that changes the stage's **stored output**, and
    /// nothing else. A dimension's model-facing `description` is deliberately
    /// excluded: editing it must not invalidate an index.
    fn config(&self) -> String;

    /// Fingerprint input for the stage's contribution to *cluster* records.
    ///
    /// Separate from [`config`](Stage::config) because the two key different
    /// objects and must be able to move independently. A cluster prompt cannot
    /// change a segment descriptor's bytes, so folding it into `config` would
    /// re-bill a whole corpus classification every time somebody reworded a
    /// label; leaving it out of *both* is the failure this exists to prevent —
    /// an edited label instruction that reuses the labels it no longer produces.
    ///
    /// Defaults to `config`, which is right for every stage whose behaviour
    /// above level 0 is the same as at level 0.
    fn fabric_config(&self) -> String {
        self.config()
    }
}

pub trait Discoverer: Stage {
    /// Enumerate resources, sorted by canonical key.
    ///
    /// Sorting is required, not incidental: filesystem enumeration order is
    /// unstable across platforms and runs, and everything downstream inherits
    /// this order (spec §6.1).
    fn discover(&self, root: &Path) -> Result<Vec<Resource>>;
}

/// Decodes and normalizes raw resource bytes.
///
/// Takes bytes rather than a path so the stage carries no assumption that a
/// resource is a file; fetching belongs to the driver, which is the only layer
/// that knows a resource's kind.
pub trait Preprocessor: Stage {
    /// Decode and normalize, or `Ok(None)` if the bytes aren't text.
    fn preprocess(&self, resource: &Resource, bytes: &[u8]) -> Result<Option<Content>>;
}

/// Cuts content into segments.
///
/// Segmentation is shared across dimensions on purpose — segment identity is
/// the join key that makes cross-dimension correlation well-defined.
pub trait Segmenter: Stage {
    /// Produce segments in ascending offset order.
    fn segment(&self, content: &Content) -> Result<Vec<Segment>>;
}

/// Everything a classifier needs for one invocation.
///
/// Carries a batch of items **and** the dimension set this implementation
/// owns — the two-dimensional batching that keeps one call covering many of
/// each.
///
/// An **item** is the texts one record is built from, and the shape is uniform
/// up the whole ladder: at level 0 an item is a single segment's own content,
/// and above that it is the descriptors of the group being summarized.
/// That uniformity is what lets one stage produce every rung — a cluster gets
/// classified exactly as a segment does, with only the prompt differing.
pub struct ClassifyBatch<'a> {
    /// One entry per record to produce; each holds that record's source texts.
    pub items: &'a [Vec<&'a str>],
    /// Dimensions this implementation is responsible for.
    pub dimensions: &'a [String],
    /// The fabric level being produced. Level 0 is the segment itself; each
    /// level above summarizes a group from the level below.
    pub level: u16,
}

pub trait Classifier: Stage {
    /// One record map per item, in the same order as the input.
    ///
    /// Returned records must leave `RecordMeta::segment` as
    /// `None`: the stored descriptor object carries no resource identity, so
    /// two byte-identical files derive the same object key and share one stored
    /// object. The span and resource are reattached at load time.
    ///
    /// One implementation serves every rung of the fabric, so a record must
    /// carry the altitude it was asked for:
    ///
    /// - `meta.level` is [`ClassifyBatch::level`], not always zero.
    /// - `meta.kind` is [`map_format::RecordKind::Cluster`] above level 0.
    /// - `meta.children` stays empty. The fabricator owns child identity and
    ///   attaches it; a classifier never sees the members' keys.
    ///
    /// The first two are load-bearing rather than cosmetic:
    /// `Record::validate` rejects a segment record carrying children, so a
    /// classifier that ignores the level makes the fabricator fail loudly
    /// instead of storing a mislabeled cluster.
    ///
    /// Honouring [`ClassifyBatch::dimensions`] matters for the same reason —
    /// above level 0 the fabricator drives one dimension at a time, because
    /// each dimension's clusters partition the corpus differently and there is
    /// no shared item to hang several facets off.
    fn classify(&self, batch: &ClassifyBatch<'_>) -> Result<Vec<DimensionRecords>>;
}

/// Everything an embedder needs for one invocation.
pub struct EmbedBatch<'a> {
    pub content: &'a Content,
    pub segments: &'a [Segment],
    pub dimensions: &'a [String],
    /// Records the classifier already produced, parallel to `segments`. A
    /// semantic embedder usually encodes the descriptor rather than the raw
    /// segment text, so it needs both.
    pub classified: &'a [DimensionRecords],
}

/// Produces tensor payloads, and encodes queries into the same space.
///
/// Both halves live on one trait because they must not diverge: a query encoded
/// by a different model than the corpus is silently meaningless rather than an
/// error.
pub trait Embedder: Stage {
    /// One tensor map per segment, in the same order as the input.
    fn embed(&self, batch: &EmbedBatch<'_>) -> Result<Vec<DimensionTensors>>;

    /// Encode query text for one dimension.
    ///
    /// Never invokes a classifier: query text arrives already dimension-split,
    /// which is what keeps query time free of network, key, and model-download
    /// requirements.
    fn encode_query(&self, dimension: &str, text: &str) -> Result<Tensor>;
}

/// One record offered to a scorer at build time, with its location attached.
pub struct IndexedRecord<'a> {
    /// Assigned by the driver from the segmentation rather than counted
    /// per-call: one scorer receives records for several dimensions, and the
    /// same span must keep one id across all of them. A per-call counter would
    /// renumber per dimension, so a candidate list built from one dimension
    /// would address the wrong spans in another — silently.
    pub id: u32,
    pub resource: &'a str,
    pub segment: Segment,
    pub record: &'a Record,
}

/// What the retriever needs back about a scored record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Located<'a> {
    pub resource: &'a str,
    pub start: u32,
    pub end: u32,
    /// Height in the fabric; segments are 0. Retrieval zoom is a filter on this
    /// field, not a separate mechanism.
    pub level: u16,
}

/// One dimension's half of an N-dimensional query.
///
/// Both payloads are optional and independent, mirroring the record itself.
#[derive(Clone, Debug, Default)]
pub struct QueryField {
    pub text: Option<String>,
    pub tensor: Option<Tensor>,
}

impl QueryField {
    pub fn text(text: impl Into<String>) -> Self {
        QueryField {
            text: Some(text.into()),
            tensor: None,
        }
    }

    pub fn is_empty(&self) -> bool {
        let no_text = self.text.as_ref().is_none_or(|t| t.trim().is_empty());
        let no_tensor = self.tensor.as_ref().is_none_or(|t| t.data.is_empty());
        no_text && no_tensor
    }
}

/// A whole N-dimensional query: one optional field per dimension, so a partial
/// query is always valid.
pub type QueryBundle = BTreeMap<String, QueryField>;

/// Normalized per-dimension scores for one candidate.
///
/// **Every value must lie in `[0, 1]`.** A dimension absent from the map scored
/// zero and is omitted rather than stored.
pub type DimensionScores = BTreeMap<String, f32>;

/// Corpus statistics one scorer holds for one query, summable across indexes.
///
/// BM25 is normalized against statistics of the corpus it scored — corpus size,
/// per-term document frequency, average document length. Those differ per index,
/// so scores from two indexes are not comparable by construction: a term one
/// index has never seen is excluded from its ceiling, which *raises* every score
/// it returns. Summing these and scoring every index against the total is what
/// makes a federated ranking mean anything.
///
/// A corpus-independent scorer (cosine over unit vectors) contributes nothing
/// and needs none of this.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CorpusStats {
    pub records: u64,
    /// Summed record lengths, kept rather than the mean so merging stays exact.
    pub total_length: f64,
    /// Document frequency per query term. A term absent here has `df == 0`
    /// across the whole federation, which is the only case where dropping it
    /// from the ceiling is correct.
    pub document_frequency: BTreeMap<String, u32>,
}

impl CorpusStats {
    /// Fold another index's contribution in.
    pub fn merge(&mut self, other: &CorpusStats) {
        self.records += other.records;
        self.total_length += other.total_length;
        for (term, df) in &other.document_frequency {
            *self.document_frequency.entry(term.clone()).or_insert(0) += df;
        }
    }

    pub fn average_length(&self) -> f32 {
        if self.records == 0 {
            0.0
        } else {
            (self.total_length / self.records as f64) as f32
        }
    }
}

/// Accumulates records into a queryable scorer.
///
/// Split from [`Scorer`] because the production scorer is a memory-mapped file:
/// preparation happens once at index time and the queryable form is immutable
/// and shared. A single `prepare(&mut self)` cannot express that.
pub trait ScorerBuilder: Stage {
    /// Offer one record. Implementations take the payload they understand and
    /// ignore the rest, so a mixed corpus needs no filtering by the driver.
    /// Ignoring a record renumbers nothing, since ids arrive on
    /// [`IndexedRecord::id`].
    fn push(&mut self, record: &IndexedRecord<'_>);

    fn build(self: Box<Self>) -> Result<Box<dyn Scorer>>;
}

/// Which records a query scores.
///
/// `All` is the whole corpus and the fast path: a scorer iterates its own
/// records directly instead of building an eligibility set sized to the corpus
/// and probing it per match. That set measured roughly half of query latency at
/// 100k+ records — pure waste when the answer is always "yes".
///
/// `Ids` restricts to an explicit set. The ids are **positional within the
/// scorer being called**, so a cross-dimension scope must be expressed by
/// location, never by handing one dimension's ids to another.
#[derive(Clone, Copy, Debug)]
pub enum CandidateScope<'a> {
    All,
    Ids(&'a [u32]),
}

/// Scores candidates for the dimensions one implementation owns.
///
/// # The candidate set is fixed before any dimension is consulted
///
/// The retriever settles which records are in play and hands **the same list to
/// every dimension**. That is a correctness property, not a performance one: if
/// a dimension proposed candidates by relevance, the cheapest dimension's
/// recall would silently cap the whole result set, and a record that is a weak
/// lexical match but a strong semantic one could never be retrieved at any
/// weighting. Dimensions decide *how well* a candidate matches, never *whether
/// it is considered*.
///
/// Efficiency survives this because a scorer only reports the candidates it can
/// say something about; a postings-based scorer stays O(matching).
///
/// # Ids are positional within a scorer, not shared across dimensions
///
/// A scorer numbers its records in push order, and a dimension that skips a
/// segment does not advance the numbering of one that scored it — so id `n`
/// need not denote the same span across dimensions. Fusion therefore keys on
/// location (see [`Scorer::located`]), never on id.
pub trait Scorer: Stage {
    /// Number of record slots, including ones this scorer cannot score.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// What this scorer contributes to the federated statistics for `query`.
    ///
    /// `None` from a scorer whose scale does not depend on its corpus, which is
    /// every tensor scorer: a cosine between unit vectors means the same thing
    /// in any index.
    fn corpus_stats(&self, _query: &QueryBundle, _dimensions: &[String]) -> Option<CorpusStats> {
        None
    }

    /// Score the records in `scope` across `dimensions`.
    ///
    /// Returns `(record id, scores)` ascending by id, for records this scorer
    /// has something to say about; omitting one means zero.
    ///
    /// Every score must be normalized to `[0, 1]`. Normalization belongs here
    /// because only the implementation knows its own scale: an unbounded BM25
    /// sum and a bounded cosine cannot be reconciled by a caller that sees
    /// nothing but floats.
    ///
    /// `corpus` carries the summed statistics of every index answering this
    /// query. Given it, a corpus-dependent scorer must normalize against those
    /// rather than its own, so that all of them land on one scale. `None` means
    /// a single index, where its own statistics *are* the totals.
    fn score(
        &self,
        scope: CandidateScope<'_>,
        query: &QueryBundle,
        dimensions: &[String],
        corpus: Option<&CorpusStats>,
    ) -> Result<Vec<(u32, DimensionScores)>>;

    fn located(&self, id: u32) -> Option<Located<'_>>;
}

/// Dimensions sharing one stage implementation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImplementationGroup {
    pub implementation: String,
    pub dimensions: Vec<String>,
}

/// Partition dimensions by the implementation they selected for a stage.
///
/// The orchestration rule the cost model rests on: three dimensions naming the
/// same classifier become **one** group, and therefore one call per segment
/// batch rather than three.
///
/// `select` returns the implementation a dimension chose, or `None` if it does
/// not participate. Groups and the dimensions within them come back sorted, so
/// traversal order is deterministic.
pub fn group_by_implementation<'a, D, F>(dimensions: D, select: F) -> Vec<ImplementationGroup>
where
    D: IntoIterator<Item = &'a str>,
    F: Fn(&str) -> Option<String>,
{
    let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for name in dimensions {
        if let Some(implementation) = select(name) {
            groups
                .entry(implementation)
                .or_default()
                .push(name.to_owned());
        }
    }
    groups
        .into_iter()
        .map(|(implementation, mut dimensions)| {
            dimensions.sort();
            ImplementationGroup {
                implementation,
                dimensions,
            }
        })
        .collect()
}

/// Locate the directory containing a `.map`, walking up from `start`.
pub fn find_map_root(start: &Path) -> Option<PathBuf> {
    let mut here = start.canonicalize().ok()?;
    loop {
        // Require `config.toml`, not merely a `.map` directory: the user-global
        // cache at `~/.map` holds the model cache and `llm.toml` but no config,
        // and would otherwise be mistaken for an index root.
        if here.join(".map").join("config.toml").is_file() {
            return Some(here);
        }
        if !here.pop() {
            return None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use map_format::{RecordKind, RecordMeta};

    #[test]
    fn a_bare_dot_map_directory_is_not_an_index_root() {
        let base = std::env::temp_dir().join(format!("map-root-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join(".map").join("models")).unwrap();
        let work = base.join("project");
        std::fs::create_dir_all(&work).unwrap();

        assert_eq!(find_map_root(&work), None, "cache .map must not be a root");

        std::fs::write(base.join(".map").join("config.toml"), b"version = 1\n").unwrap();
        assert_eq!(
            find_map_root(&work),
            Some(base.canonicalize().unwrap()),
            "a .map with config.toml is a root"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn shared_implementation_becomes_one_group() {
        let dims = ["descriptive", "data", "prose", "lexical"];
        let groups = group_by_implementation(dims, |d| {
            Some(match d {
                "lexical" => "structural".to_owned(),
                _ => "llm".to_owned(),
            })
        });

        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].implementation, "llm");
        assert_eq!(groups[0].dimensions, ["data", "descriptive", "prose"]);
        assert_eq!(groups[1].implementation, "structural");
        assert_eq!(groups[1].dimensions, ["lexical"]);
    }

    #[test]
    fn non_participating_dimensions_are_dropped() {
        let groups = group_by_implementation(["lexical", "descriptive"], |d| {
            (d != "lexical").then(|| "candle".to_owned())
        });
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].dimensions, ["descriptive"]);
    }

    #[test]
    fn grouping_is_order_independent() {
        let a = group_by_implementation(["b", "a", "c"], |_| Some("x".to_owned()));
        let b = group_by_implementation(["c", "b", "a"], |_| Some("x".to_owned()));
        assert_eq!(a, b);
    }

    #[test]
    fn segment_slices_normalized_content() {
        let content = Content {
            key: "src/main.rs".into(),
            text: "fn main() {}\nfn other() {}".into(),
        };
        let seg = Segment { start: 13, end: 25 };
        assert_eq!(seg.slice(&content), "fn other() {");
    }

    #[test]
    fn a_record_with_only_a_tensor_is_still_a_record() {
        let r = Record {
            descriptor: None,
            tensor: Some(Tensor {
                dtype: map_format::DType::Binary,
                shape: vec![64],
                data: vec![0xAB; 8],
            }),
            meta: RecordMeta {
                kind: RecordKind::Segment,
                dimension: "descriptive".into(),
                level: 0,
                children: vec![],
            },
        };
        let mut records = DimensionRecords::new();
        records.insert("descriptive".to_owned(), r);
        assert!(records["descriptive"].validate().is_ok());
    }

    #[test]
    fn an_empty_query_field_asks_for_nothing() {
        assert!(QueryField::default().is_empty());
        assert!(QueryField::text("   ").is_empty());
        assert!(!QueryField::text("binary detection").is_empty());
    }
}
