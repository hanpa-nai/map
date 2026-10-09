//! Errors produced by the format layer.

use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, Error>;

/// Anything that can go wrong reading or writing a `.map` directory.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("malformed digest {0:?}: expected 64 hex characters")]
    MalformedDigest(String),

    #[error(
        "dimension {0:?} produces neither a descriptor nor a tensor; \
         at least one is required (spec §3)"
    )]
    EmptyDimension(String),

    #[error("record has neither descriptor nor tensor; at least one is required (spec §3)")]
    EmptyRecord,

    #[error("inconsistent record: {0}")]
    InconsistentRecord(&'static str),

    /// A dimension name becomes a filename (`.map/cache/<name>.pack`) and
    /// `config.toml` is committed — so a hostile name in a cloned repository
    /// would otherwise escape the index directory.
    #[error("invalid dimension name {0:?}: expected 1-64 chars of [a-z0-9_-]")]
    InvalidDimensionName(String),

    /// A setting the format defines but the writer does not honour.
    ///
    /// Refused at load rather than ignored: these read as guarantees, and the
    /// gap between what one promises and what gets written is invisible
    /// afterwards.
    #[error(
        "{setting} = {value:?} is not implemented yet; only {implemented:?} is. \
         It would be silently ignored, so it is refused instead."
    )]
    UnimplementedSetting {
        /// Owned because a per-dimension setting has to name its dimension —
        /// `dimensions.semantic.fabricator.impl` — and a `&'static str` could
        /// only say which key, never which dimension wrote it.
        setting: String,
        /// Owned so the message can quote what was actually written. A
        /// `&'static str` forced settings with an open value space to say
        /// something vague, which defeats the point of refusing loudly.
        value: String,
        implemented: &'static str,
    },

    /// A fabricator over a dimension that stores no descriptors. A cluster is
    /// summarized from its members' stored descriptors; with none, the tree is
    /// built from nothing and comes out empty without any error.
    #[error(
        "dimension {0:?} has a fabricator but stores no descriptor, so its clusters would have \
         nothing to summarize. Give it a classifier that persists, or drop the fabricator."
    )]
    UnfabricatableDimension(String),

    /// A fabricator over a dimension with no embedder. Clustering is
    /// agglomerative on a cosine threshold (spec §3.1), so with no vectors
    /// there is nothing to measure similarity between and the tree comes out
    /// empty without any error.
    #[error(
        "dimension {0:?} has a fabricator but no embedder, so there are no vectors to cluster on. \
         Give it an embedder, or drop the fabricator."
    )]
    UnclusterableDimension(String),

    /// A fabricator tunable outside the range the fabricator can act on.
    ///
    /// Refused rather than clamped or defaulted: the indexer reads these with
    /// `as_integer`/`as_float` and falls back on anything that does not parse,
    /// so a typo silently substitutes a tree nobody asked for — and on an LLM
    /// dimension that tree is billed.
    #[error("dimension {dimension:?}: fabricator {setting} = {value} is unusable: {because}")]
    InvalidFabricatorSetting {
        dimension: String,
        setting: &'static str,
        value: String,
        because: &'static str,
    },

    /// Segmenter settings that produce no usable segmentation.
    ///
    /// Both cases fail quietly rather than loudly if unchecked: `lines = 0`
    /// emits no segments at all, so the index comes out empty and successful;
    /// `overlap >= lines` collapses the stride to one line, so a file becomes
    /// almost as many segments as it has lines — which on an LLM dimension is a
    /// bill, not a warning.
    #[error("segmenter {setting} = {value} is unusable: {because}")]
    InvalidSegmenter {
        setting: &'static str,
        value: usize,
        because: &'static str,
    },

    /// A setting that moved to another stage. Refused rather than ignored: it
    /// would otherwise be dropped silently and the default would quietly stand
    /// in for what the operator actually wrote.
    #[error(
        "dimension {dimension:?}: {from} has moved to {to}. \
         It would be silently ignored, so it is refused instead."
    )]
    MovedSetting {
        dimension: String,
        from: &'static str,
        to: &'static str,
    },

    /// Dimensions sharing a classifier share the one descriptor object it
    /// produces, so whether to store that object cannot be settled per
    /// dimension. Refused rather than resolved by picking one, which would
    /// either write a corpus somebody asked not to store or discard output
    /// somebody paid for.
    #[error(
        "dimensions {dimensions:?} share the {implementation:?} classifier but disagree on \
         persist_output; they produce one descriptor object, so it cannot be both stored and not"
    )]
    MixedPersistence {
        implementation: String,
        dimensions: Vec<String>,
    },

    #[error("tensor shape {shape:?} of {dtype} needs {expected} bytes, got {actual}")]
    TensorShapeMismatch {
        shape: Vec<u32>,
        dtype: &'static str,
        expected: usize,
        actual: usize,
    },

    /// Object keys are input-addressed, so bytes cannot be self-verified; this
    /// is the only integrity check available. See spec §4.
    #[error(
        "integrity failure for object {key}: manifest records {expected}, bytes hash to {actual}"
    )]
    IntegrityFailure {
        key: String,
        expected: String,
        actual: String,
    },

    /// Legal under input-addressing and NOT resolvable by union. See spec §4.
    #[error("object key {key} collides: same key, different content ({a} vs {b})")]
    KeyCollision { key: String, a: String, b: String },

    /// Reachable from untrusted committed bytes via [`crate::codec::decode_tensor`],
    /// so this must be an error rather than a wrapping multiply: a wrapped
    /// product can compute a required byte count of zero for an enormous
    /// declared shape, letting an empty buffer pass validation.
    #[error("tensor shape {shape:?} overflows when multiplied out")]
    TensorShapeOverflow { shape: Vec<u32> },

    #[error("malformed tensor object: {0}")]
    MalformedTensor(&'static str),

    #[error("index uses {kind} version {found}, this build supports {expected}")]
    UnsupportedVersion {
        /// `manifest` or `config`.
        kind: &'static str,
        found: u32,
        expected: u32,
    },

    #[error("index uses hash algorithm {found:?}, this build implements {expected:?}")]
    UnsupportedHashAlgo {
        found: String,
        expected: &'static str,
    },

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("config parse error: {0}")]
    TomlDe(#[from] toml::de::Error),

    #[error("config serialize error: {0}")]
    TomlSer(#[from] toml::ser::Error),
}

impl Error {
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Error::Io {
            path: path.into(),
            source,
        }
    }
}
