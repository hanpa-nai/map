//! `.map/config.toml` — the committed, human-editable settings.
//!
//! The dimension table is the interesting part. Each dimension is a
//! self-contained facet carrying its own parameter description (which becomes
//! the description the model reads when choosing what to put in that query
//! field), its stage selections, and its storage dials.
//!
//! A dimension must produce a descriptor, a tensor, or both (spec §3). Since
//! the classifier produces the descriptor and the embedder produces the
//! tensor, that invariant is enforced here as "at least one of `classifier`
//! or `embedder` must be configured".

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::codec::canonical_json;
use crate::error::{Error, Result};
use crate::hash::Fingerprint;

/// Current config schema version.
pub const CONFIG_VERSION: u32 = 1;

/// How tall the fabricator builds when its `max_levels` is not set. Mirrors the
/// indexer's own default; the two must agree or a dimension would search levels
/// it never builds.
const DEFAULT_MAX_LEVELS: u16 = 3;

/// Reference to a stage implementation plus its opaque settings.
///
/// `settings` is not interpreted here — it belongs to the named
/// implementation. It *is* folded into the dimension fingerprint, so editing
/// a prompt invalidates exactly the objects that prompt produced.
///
/// `persist_output` is the exception: the driver reads it, not the
/// implementation, which is why it is a field rather than one more opaque
/// setting.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StageRef {
    /// Implementation name, e.g. `structural`, `llm`, `distilled`, `content`.
    #[serde(rename = "impl")]
    pub implementation: String,
    /// Whether the driver stores what this stage produces.
    ///
    /// A stage exists to transform content on the way to the next one; storing
    /// the result is a separate question, and only worth it when recomputing
    /// costs more than the bytes do. An LLM classifier must persist — its
    /// output is billed. A `content` classifier must not: it would write the
    /// whole corpus into `desc/` a second time to save work that is nearly
    /// free to redo.
    ///
    /// A dimension whose classifier persists nothing emits no descriptor, so
    /// it must carry an embedder to be searchable at all — see
    /// [`DimensionConfig::emits_descriptor`].
    ///
    /// Skipped when serializing at its default, so adding this field moved no
    /// existing dimension's [`artifact_fingerprint`](DimensionConfig::artifact_fingerprint)
    /// and re-keyed nothing. Turning it off is a real change to what a run
    /// produces, and does move the fingerprint.
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub persist_output: bool,
    #[serde(default, flatten)]
    pub settings: BTreeMap<String, toml::Value>,
}

fn is_true(b: &bool) -> bool {
    *b
}

impl StageRef {
    pub fn new(implementation: impl Into<String>) -> Self {
        StageRef {
            implementation: implementation.into(),
            persist_output: true,
            settings: BTreeMap::new(),
        }
    }
}

/// How content is cut into segments, before any dimension sees it.
///
/// One top-level section rather than a per-dimension stage, because segment
/// identity is the join key that makes cross-dimension correlation well-defined
/// — two dimensions cutting differently could not be correlated at all.
///
/// This was hardcoded in the indexer until the key existed, and the defaults are
/// exactly the values it hardcoded, so declaring the section changes no
/// fingerprint and re-keys nothing. They are a starting point measured on code;
/// what suits prose, transcripts or records is an open question, and the point
/// of the key is that answering it is a config edit rather than a Rust edit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SegmenterConfig {
    /// Implementation name, matching a segmenter stage's `impl`.
    #[serde(rename = "impl")]
    pub implementation: String,
    /// Lines per segment.
    pub lines: usize,
    /// Lines of overlap between consecutive segments.
    ///
    /// Overlap matters more than it looks: something defined at a window
    /// boundary would otherwise land in neither segment's descriptor.
    pub overlap: usize,
}

impl Default for SegmenterConfig {
    fn default() -> Self {
        SegmenterConfig {
            implementation: "window".to_owned(),
            lines: 40,
            overlap: 8,
        }
    }
}

/// Vector quantization for a dimension's tensors.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Quant {
    F16,
    I8,
    /// One bit per component. The committed default: ~96 B per 768-d vector.
    #[default]
    Binary,
}

/// How much derived data enters the repository.
///
/// **Only [`Full`](CommitLevel::Full) is implemented**, and a config selecting
/// any other level is rejected at load rather than accepted and ignored. The
/// quantized tiers need `quant` (unimplemented), and the narrower levels need
/// the writer to consult this field, which nothing does.
///
/// Rejecting is the point: these read as guarantees. Somebody choosing
/// [`None`](CommitLevel::None) so that nothing derived enters their repository
/// would otherwise get a fully committed index with no sign anything was
/// disregarded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CommitLevel {
    /// Everything, including full-precision tensors. What the indexer does.
    #[default]
    Full,
    /// Descriptors plus binary-quantized tensors. **Not implemented.**
    Binary,
    /// Descriptors only; tensors rebuilt locally. **Not implemented.**
    DescriptorsOnly,
    /// Nothing derived is committed; the index is purely local. **Not
    /// implemented.**
    None,
}

impl CommitLevel {
    /// The levels the writer can actually honour.
    fn implemented(self) -> bool {
        matches!(self, CommitLevel::Full)
    }

    fn as_str(self) -> &'static str {
        match self {
            CommitLevel::Full => "full",
            CommitLevel::Binary => "binary",
            CommitLevel::DescriptorsOnly => "descriptors-only",
            CommitLevel::None => "none",
        }
    }
}

/// Where the committed index physically lives.
///
/// **Only [`WorkingTree`](Location::WorkingTree) is implemented**; see
/// [`CommitLevel`] for why the alternative is refused rather than ignored.
/// Whether the orphan ref should even be the eventual default is still open —
/// packing cut the working-tree file count from 5,000 to 16, which removed most
/// of the argument for it (spec §9.1).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Location {
    /// `.map/index/` in the working tree. Works with no git at all.
    #[default]
    WorkingTree,
    /// An orphan ref (`refs/map/index`), out of the working tree and history.
    /// **Not implemented.**
    Ref,
}

impl Location {
    fn implemented(self) -> bool {
        matches!(self, Location::WorkingTree)
    }

    fn as_str(self) -> &'static str {
        match self {
            Location::WorkingTree => "working-tree",
            Location::Ref => "ref",
        }
    }
}

/// Repository-wide storage policy.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StoragePolicy {
    pub commit: CommitLevel,
    pub location: Location,
}

/// One facet of the index.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DimensionConfig {
    /// Shown to the model as the description of this query parameter.
    ///
    /// This is the whole non-speculative-routing mechanism: the model reads
    /// these descriptions once and fills in the fields it can, rather than
    /// probing to discover what exists.
    pub description: String,

    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Produces the descriptor. Absent for pure-embedding dimensions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub classifier: Option<StageRef>,

    /// Produces the tensor. Absent for lexical dimensions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedder: Option<StageRef>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fabricator: Option<StageRef>,

    /// Fabric levels this dimension searches.
    ///
    /// A query's `--level` scope intersects with this per dimension, and the
    /// intersection is the candidate set that dimension scores. A dimension
    /// whose intersection is empty contributes nothing — and, because it could
    /// not have scored those candidates, is left out of the fused denominator
    /// rather than counting against them.
    ///
    /// Omitted means [`DimensionConfig::search_levels`] derives it from the
    /// stages, which is accurate by construction — **prefer that**.
    ///
    /// Declaring it is only useful to search *more* levels than exist, which is
    /// harmless. Declaring *fewer* than were built is **refused at load**: those
    /// records are on disk and referenced by the manifest, but no query would
    /// ever return them, which for an LLM dimension silently discards what
    /// somebody paid for. This reverses an earlier reading of the field as "the
    /// operator's explicit choice to search less than was built" — that choice
    /// is indistinguishable from a mistake, and the failure is invisible.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub levels: Option<Vec<u16>>,

    /// Matryoshka truncation, if the embedder supports it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dims: Option<u32>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quant: Option<Quant>,
}

fn default_true() -> bool {
    true
}

impl DimensionConfig {
    /// Whether this dimension stores a descriptor a query can match against.
    ///
    /// A classifier that does not persist its output transforms content on the
    /// way to the embedder and leaves nothing behind, so it makes the dimension
    /// no more searchable than having no classifier at all.
    pub fn emits_descriptor(&self) -> bool {
        self.classifier
            .as_ref()
            .is_some_and(|classifier| classifier.persist_output)
    }

    pub fn emits_tensor(&self) -> bool {
        self.embedder.is_some()
    }

    /// Whether the embedder encodes this dimension's classifier output rather
    /// than falling through to the raw segment text.
    ///
    /// Deliberately **not** gated on `persist_output`: a classifier that stores
    /// nothing still shapes what gets embedded in the same run. Conflating this
    /// with [`emits_descriptor`](Self::emits_descriptor) would make a
    /// pass-through classifier's transformation silently vanish, which is the
    /// one thing it exists to do.
    pub fn embeds_descriptor(&self) -> bool {
        self.classifier.is_some() && self.embedder.is_some()
    }

    /// The levels this dimension searches.
    ///
    /// Declared, or derived from the stages when omitted: a fabricator can
    /// build up to its `max_levels`, and without one a dimension holds only the
    /// segments themselves.
    ///
    /// Over-declaring is harmless — a level no dimension holds records at
    /// yields no candidates, so nothing is scored and no denominator is
    /// consulted. Under-declaring is **refused** when the index opens, against
    /// the levels the manifest records as built; this function does not know
    /// them, so it reports what was asked for and the check lives at load.
    pub fn search_levels(&self) -> BTreeSet<u16> {
        if let Some(declared) = &self.levels {
            return declared.iter().copied().collect();
        }
        // The fallback means "`max_levels` was not written", and nothing else:
        // `validate_fabricator` refuses a `max_levels` that is not an integer,
        // is below 1, or does not fit a `u16`, so the two `and_then`s below
        // cannot quietly substitute a tree height nobody configured.
        let top = self
            .fabricator
            .as_ref()
            .map(|f| {
                f.settings
                    .get("max_levels")
                    .and_then(|v| v.as_integer())
                    .and_then(|i| u16::try_from(i).ok())
                    .unwrap_or(DEFAULT_MAX_LEVELS)
            })
            .unwrap_or(0);
        (0..=top).collect()
    }

    /// Fingerprint over only the fields that produce stored artifacts.
    ///
    /// This is what object keys and federation identity derive from, and it
    /// deliberately **excludes** `description` and `enabled`:
    ///
    /// - `description` is model-facing prose. Users should iterate on it
    ///   freely to improve routing; fixing a typo must not invalidate every
    ///   object of the dimension or make previously-built indices
    ///   incomparable.
    /// - `enabled` is a switch, not an input.
    ///
    /// Changing a classifier prompt or swapping an embedder *does* change
    /// stored bytes, so those are in.
    pub fn artifact_fingerprint(&self) -> Result<Fingerprint> {
        #[derive(Serialize)]
        struct Artifacts<'a> {
            classifier: &'a Option<StageRef>,
            embedder: &'a Option<StageRef>,
            fabricator: &'a Option<StageRef>,
            dims: &'a Option<u32>,
            quant: &'a Option<Quant>,
        }

        let view = Artifacts {
            classifier: &self.classifier,
            embedder: &self.embedder,
            fabricator: &self.fabricator,
            dims: &self.dims,
            quant: &self.quant,
        };
        Ok(Fingerprint::of(&canonical_json(&view)?))
    }

    /// Fingerprint over the dimension's complete configuration.
    ///
    /// Informational — recorded so a reader can tell whether two indices were
    /// built from identical config. Never use it for object keys or
    /// federation identity; see [`DimensionConfig::artifact_fingerprint`].
    pub fn config_fingerprint(&self) -> Result<Fingerprint> {
        Ok(Fingerprint::of(&canonical_json(self)?))
    }
}

/// Refuse a fabricator the indexer cannot actually run as written.
///
/// Every case here used to be absorbed silently. The indexer builds an
/// agglomerative fabricator and nothing else, so any other `impl` was skipped
/// with no clusters and no message — exactly the silent partial index spec §9.3
/// forbids. The tunables are read with `as_integer`/`as_float` and fall back to
/// a default on anything that does not parse, so `max_levels = "3"` quietly
/// became 3 and `threshold = 70` quietly became a threshold no pair of unit
/// vectors can reach.
///
/// This refuses at load and changes no fingerprint input: `settings` is still
/// the same opaque map that
/// [`artifact_fingerprint`](DimensionConfig::artifact_fingerprint) hashes, so
/// nothing is re-keyed.
fn validate_fabricator(name: &str, dim: &DimensionConfig, fabricator: &StageRef) -> Result<()> {
    if fabricator.implementation != "agglomerative" {
        return Err(Error::UnimplementedSetting {
            setting: format!("dimensions.{name}.fabricator.impl"),
            value: fabricator.implementation.clone(),
            implemented: "agglomerative",
        });
    }

    // Clustering is agglomerative on a cosine threshold, so the vectors are the
    // input; descriptors are only what the labels are written from.
    if !dim.emits_tensor() {
        return Err(Error::UnclusterableDimension(name.to_owned()));
    }

    let invalid = |setting: &'static str, value: &toml::Value, because: &'static str| {
        Error::InvalidFabricatorSetting {
            dimension: name.to_owned(),
            setting,
            value: value.to_string(),
            because,
        }
    };

    if let Some(threshold) = fabricator.settings.get("threshold") {
        let as_number = threshold
            .as_float()
            .or_else(|| threshold.as_integer().map(|i| i as f64));
        match as_number {
            Some(t) if t > 0.0 && t <= 1.0 => {}
            _ => {
                return Err(invalid(
                    "threshold",
                    threshold,
                    "cosine similarity is a number in (0, 1]; outside it either every pair merges \
                     or none does",
                ))
            }
        }
    }

    let integer = |setting: &'static str| -> Result<Option<(i64, &toml::Value)>> {
        match fabricator.settings.get(setting) {
            None => Ok(None),
            Some(value) => match value.as_integer() {
                Some(i) => Ok(Some((i, value))),
                None => Err(invalid(setting, value, "expected an integer")),
            },
        }
    };

    let min_cluster = integer("min_cluster")?;
    if let Some((n, value)) = min_cluster {
        if n < 2 {
            return Err(invalid(
                "min_cluster",
                value,
                "a cluster of fewer than two records summarizes nothing",
            ));
        }
    }
    if let Some((n, value)) = integer("max_cluster")? {
        let floor = min_cluster.map_or(2, |(m, _)| m);
        if n != 0 && n < floor {
            return Err(invalid(
                "max_cluster",
                value,
                "must be 0 for unbounded, or at least `min_cluster`",
            ));
        }
    }
    if let Some((n, value)) = integer("max_levels")? {
        // The upper bound is `search_levels`' own: it reads this through
        // `u16::try_from`, and anything wider would silently fall back to the
        // default height instead.
        if n < 1 || u16::try_from(n).is_err() {
            return Err(invalid(
                "max_levels",
                value,
                "a tree of no levels above the segments is a fabricator that does nothing; \
                 the ceiling is 65535",
            ));
        }
    }
    if let Some((n, value)) = integer("min_remaining")? {
        if n < 1 {
            return Err(invalid(
                "min_remaining",
                value,
                "fabrication stops when fewer than this many records remain, so zero never stops",
            ));
        }
    }
    Ok(())
}

/// Whether a dimension name is safe to use as a path component.
///
/// A dimension name becomes a filename — `.map/cache/<name>.pack` — and
/// `config.toml` is a committed file, so in the clone-someone's-repository
/// case the name is attacker-controlled. Restricting the charset is simpler
/// and more durable than sanitizing.
pub fn valid_dimension_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-')
}

/// The parsed `.map/config.toml`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub version: u32,
    #[serde(default)]
    pub storage: StoragePolicy,
    #[serde(default)]
    pub segmenter: SegmenterConfig,
    /// Dimension table, keyed by name.
    ///
    /// `BTreeMap` rather than `HashMap`: iteration order feeds serialized
    /// output and fingerprints, and randomized order would break Tier A
    /// determinism (spec §6).
    #[serde(default)]
    pub dimensions: BTreeMap<String, DimensionConfig>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            version: CONFIG_VERSION,
            storage: StoragePolicy::default(),
            segmenter: SegmenterConfig::default(),
            dimensions: BTreeMap::new(),
        }
    }
}

impl Config {
    /// The zero-configuration default written by `map init`.
    ///
    /// Only the lexical dimension is enabled: structural classification plus
    /// BM25, with no tensor. That path needs no API key, no network, no model
    /// download, and is fully Tier A deterministic — so `map init && map index
    /// && map find` succeeds on a fresh machine with nothing configured.
    ///
    /// Semantic dimensions are one config edit away and are emitted commented
    /// out by the `map init` template.
    pub fn zero_config() -> Self {
        let mut dimensions = BTreeMap::new();
        dimensions.insert(
            "declaration".to_owned(),
            DimensionConfig {
                description: "The exact name of the thing whose declaration you want -- a function, type, class, module, or constant. A name here ranks the place that declares it above the places that use it."
                    .to_owned(),
                enabled: true,
                classifier: Some(StageRef::new("declaration")),
                embedder: None,
                fabricator: None,
                levels: None,
                dims: None,
                quant: None,
            },
        );
        dimensions.insert(
            "lexical".to_owned(),
            DimensionConfig {
                description: "Exact words, names, and literals as they appear in the text. \
                              Use this for anything you would otherwise search for verbatim."
                    .to_owned(),
                enabled: true,
                classifier: Some(StageRef::new("structural")),
                embedder: None,
                fabricator: None,
                levels: None,
                dims: None,
                quant: None,
            },
        );
        Config {
            version: CONFIG_VERSION,
            storage: StoragePolicy::default(),
            segmenter: SegmenterConfig::default(),
            dimensions,
        }
    }

    /// Parse from TOML text.
    pub fn parse(text: &str) -> Result<Self> {
        let cfg: Config = toml::from_str(text)?;
        if cfg.version != CONFIG_VERSION {
            return Err(Error::UnsupportedVersion {
                kind: "config",
                found: cfg.version,
                expected: CONFIG_VERSION,
            });
        }
        cfg.validate()?;
        Ok(cfg)
    }

    /// Render to TOML text.
    pub fn to_toml(&self) -> Result<String> {
        Ok(toml::to_string_pretty(self)?)
    }

    /// Dimensions that participate in indexing, in canonical order.
    pub fn active(&self) -> impl Iterator<Item = (&String, &DimensionConfig)> {
        self.dimensions.iter().filter(|(_, d)| d.enabled)
    }

    /// Enforce the spec §3 invariant and name safety across every dimension.
    ///
    /// Names are checked for *all* dimensions, not just enabled ones — a
    /// disabled dimension with a traversing name is still a hostile config,
    /// and enabling it later must not be the moment the check happens.
    pub fn validate(&self) -> Result<()> {
        // Refused rather than ignored, on the same reasoning as `CommitLevel`:
        // an unrecognized name would otherwise fall back to `window` and index
        // the corpus under a segmentation nobody asked for.
        if self.segmenter.implementation != "window" {
            return Err(Error::UnimplementedSetting {
                setting: "segmenter.impl".to_owned(),
                value: self.segmenter.implementation.clone(),
                implemented: "window",
            });
        }
        if self.segmenter.lines == 0 {
            return Err(Error::InvalidSegmenter {
                setting: "lines",
                value: 0,
                because: "a segment of no lines is never emitted, so the index would come out \
                          empty and report success",
            });
        }
        if self.segmenter.overlap >= self.segmenter.lines {
            return Err(Error::InvalidSegmenter {
                setting: "overlap",
                value: self.segmenter.overlap,
                because: "overlap must be smaller than `lines`, or the window advances one line \
                          at a time and a resource becomes nearly as many segments as it has \
                          lines",
            });
        }
        for name in self.dimensions.keys() {
            if !valid_dimension_name(name) {
                return Err(Error::InvalidDimensionName(name.clone()));
            }
        }
        for (name, dim) in self.active() {
            if !dim.emits_descriptor() && !dim.emits_tensor() {
                return Err(Error::EmptyDimension(name.clone()));
            }
        }

        // Prompts became one table keyed by level, and the classifier took over
        // labeling clusters from the fabricator. Both old spellings still parse
        // — `settings` is an open map — so silence here would replace an
        // operator's prompt with the default and bill an LLM to produce it.
        for (name, dim) in self.active() {
            if let Some(classifier) = dim.classifier.as_ref() {
                if classifier.settings.contains_key("prompt") {
                    return Err(Error::MovedSetting {
                        dimension: name.clone(),
                        from: "classifier.prompt",
                        to: r#"classifier.prompts = { "0" = … }"#,
                    });
                }
            }
            if let Some(fabricator) = dim.fabricator.as_ref() {
                if fabricator.settings.contains_key("label_prompt") {
                    return Err(Error::MovedSetting {
                        dimension: name.clone(),
                        from: "fabricator.label_prompt",
                        to: r#"classifier.prompts = { "1" = … }"#,
                    });
                }
            }
        }

        // A cluster is summarized from its members' *stored* descriptors, so a
        // dimension that persists none has nothing to fabricate from. Left
        // unchecked this builds an empty tree in silence: groups form, the
        // classifier is handed empty items, every label comes back blank, and
        // the run reports success with no clusters.
        for (name, dim) in self.active() {
            if dim.fabricator.is_some() && !dim.emits_descriptor() {
                return Err(Error::UnfabricatableDimension(name.clone()));
            }
        }

        for (name, dim) in self.active() {
            if let Some(fabricator) = dim.fabricator.as_ref() {
                validate_fabricator(name, dim, fabricator)?;
            }
        }

        // One classifier implementation produces one descriptor object per
        // resource, covering every dimension that named it. `persist_output` is
        // therefore a property of the group, and a group that disagrees with
        // itself describes no possible run.
        let mut persistence: BTreeMap<&str, (bool, Vec<String>)> = BTreeMap::new();
        for (name, dim) in self.active() {
            let Some(classifier) = dim.classifier.as_ref() else {
                continue;
            };
            let entry = persistence
                .entry(classifier.implementation.as_str())
                .or_insert((classifier.persist_output, Vec::new()));
            entry.1.push(name.clone());
            if entry.0 != classifier.persist_output {
                return Err(Error::MixedPersistence {
                    implementation: classifier.implementation.clone(),
                    dimensions: entry.1.clone(),
                });
            }
        }

        // A storage setting the writer ignores is worse than one that does not
        // exist: `commit = "none"` reads as a guarantee that nothing derived
        // enters the repository, and silently committing a full index anyway is
        // the kind of failure nobody discovers by looking.
        if !self.storage.commit.implemented() {
            return Err(Error::UnimplementedSetting {
                setting: "storage.commit".to_owned(),
                value: self.storage.commit.as_str().to_owned(),
                implemented: "full",
            });
        }
        if !self.storage.location.implemented() {
            return Err(Error::UnimplementedSetting {
                setting: "storage.location".to_owned(),
                value: self.storage.location.as_str().to_owned(),
                implemented: "working-tree",
            });
        }

        // The same rule, applied to the two per-dimension storage dials that
        // were exempt from it. `dims` and `quant` are defined by the format and
        // read by nothing: the embedder emits full-precision vectors at the
        // model's own width regardless. Someone setting `quant = "binary"` to
        // shrink a committed index gets a full-precision one and no sign that
        // the request was dropped — which is exactly what refusing
        // `commit = "none"` exists to prevent.
        for (name, dim) in self.active() {
            if dim.dims.is_some() {
                return Err(Error::UnimplementedSetting {
                    setting: "dimension `dims`".to_owned(),
                    value: "set".to_owned(),
                    implemented: "the embedder's own width",
                });
            }
            if dim.quant.is_some() {
                return Err(Error::UnimplementedSetting {
                    setting: "dimension `quant`".to_owned(),
                    value: "set".to_owned(),
                    implemented: "full precision",
                });
            }
            let _ = name;
        }
        Ok(())
    }

    /// Fingerprint over the whole configuration.
    pub fn fingerprint(&self) -> Result<Fingerprint> {
        Ok(Fingerprint::of(&canonical_json(self)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_storage_setting_the_writer_ignores_is_refused() {
        // `commit = "none"` reads as "nothing derived enters my repository".
        // Accepting it and committing a full index anyway is a promise broken
        // where nobody can see it.
        let text = r#"
version = 1
[storage]
commit = "none"
[dimensions.lexical]
description = "x"
classifier = { impl = "structural" }
"#;
        let err = Config::parse(text).unwrap_err();
        assert!(
            matches!(&err, Error::UnimplementedSetting { setting, .. } if setting == "storage.commit"),
            "got {err:?}"
        );
    }

    #[test]
    fn an_unimplemented_location_is_refused() {
        let text = r#"
version = 1
[storage]
location = "ref"
[dimensions.lexical]
description = "x"
classifier = { impl = "structural" }
"#;
        let err = Config::parse(text).unwrap_err();
        assert!(
            matches!(&err, Error::UnimplementedSetting { setting, .. } if setting == "storage.location"),
            "got {err:?}"
        );
    }

    #[test]
    fn the_implemented_storage_settings_are_accepted() {
        let text = r#"
version = 1
[storage]
commit = "full"
location = "working-tree"
[dimensions.lexical]
description = "x"
classifier = { impl = "structural" }
"#;
        Config::parse(text).unwrap();
    }

    #[test]
    fn the_default_storage_policy_describes_what_is_written() {
        // The default has to be a true statement about the bytes on disk, since
        // it is what `map init` stamps into every new index.
        let storage = StoragePolicy::default();
        assert_eq!(storage.commit, CommitLevel::Full);
        assert_eq!(storage.location, Location::WorkingTree);
        Config::zero_config().validate().unwrap();
    }

    #[test]
    fn zero_config_is_valid_and_descriptor_only() {
        let cfg = Config::zero_config();
        cfg.validate().unwrap();
        assert_eq!(cfg.dimensions.len(), 2);

        for name in ["lexical", "declaration"] {
            let dim = &cfg.dimensions[name];
            assert!(
                dim.emits_descriptor(),
                "{name}: BM25 lives in the descriptor text"
            );
            assert!(
                !dim.emits_tensor(),
                "{name}: the offline dimensions have no tensor"
            );
        }
    }

    #[test]
    fn zero_config_roundtrips_through_toml() {
        let cfg = Config::zero_config();
        let text = cfg.to_toml().unwrap();
        assert_eq!(Config::parse(&text).unwrap(), cfg);
    }

    #[test]
    fn dimension_with_neither_payload_is_rejected() {
        let text = r#"
version = 1
[dimensions.broken]
description = "produces nothing searchable"
"#;
        let err = Config::parse(text).unwrap_err();
        assert!(matches!(err, Error::EmptyDimension(n) if n == "broken"));
    }

    #[test]
    fn disabled_dimension_escapes_validation() {
        // A half-written dimension shouldn't block indexing while it's off.
        let text = r#"
version = 1
[dimensions.wip]
description = "not finished yet"
enabled = false
"#;
        assert!(Config::parse(text).is_ok());
    }

    #[test]
    fn tensor_only_dimension_is_valid() {
        let text = r#"
version = 1
[dimensions.embedding]
description = "raw content embedding"
embedder = { impl = "candle", model = "bge-small" }
"#;
        let cfg = Config::parse(text).unwrap();
        let dim = &cfg.dimensions["embedding"];
        assert!(!dim.emits_descriptor());
        assert!(dim.emits_tensor());
    }

    #[test]
    fn a_nested_prompts_table_round_trips_through_to_toml() {
        // `settings` is a #[serde(flatten)] map, and toml serializes values
        // before tables — a nested table among flattened scalars is exactly
        // where "value after table" bites. Proving it round-trips is what
        // licenses the per-level prompt spelling.
        let text = r#"
version = 1
[dimensions.descriptive]
description = "a description of your target"
embedder = { impl = "distilled" }
classifier = { impl = "llm", temperature = 0.0, prompts = { "0" = "describe it", "2" = "name the domain" } }
"#;
        let parsed = Config::parse(text).unwrap();
        let round_tripped = Config::parse(&parsed.to_toml().unwrap()).unwrap();
        assert_eq!(parsed, round_tripped);

        let prompts = &round_tripped.dimensions["descriptive"]
            .classifier
            .as_ref()
            .unwrap()
            .settings["prompts"];
        assert_eq!(
            prompts.get("0").and_then(|v| v.as_str()),
            Some("describe it")
        );
        assert_eq!(
            prompts.get("2").and_then(|v| v.as_str()),
            Some("name the domain")
        );
    }

    #[test]
    fn a_stale_prompt_spelling_is_refused_with_its_replacement() {
        // `settings` is an open map, so both old spellings still parse. Silence
        // would swap the operator's prompt for the default and bill an LLM to
        // produce output they never asked for.
        let scalar = r#"
version = 1
[dimensions.descriptive]
description = "x"
classifier = { impl = "llm", prompt = "describe it" }
"#;
        let err = Config::parse(scalar).unwrap_err();
        assert!(
            matches!(&err, Error::MovedSetting { from, .. } if *from == "classifier.prompt"),
            "{err:?}"
        );
        assert!(
            err.to_string().contains("classifier.prompts"),
            "the error must name the replacement: {err}"
        );

        let on_fabricator = r#"
version = 1
[dimensions.descriptive]
description = "x"
classifier = { impl = "llm", prompts = { "0" = "describe it" } }
embedder = { impl = "distilled" }
fabricator = { impl = "agglomerative", label_prompt = "name the theme" }
"#;
        let err = Config::parse(on_fabricator).unwrap_err();
        assert!(
            matches!(&err, Error::MovedSetting { from, .. } if *from == "fabricator.label_prompt"),
            "{err:?}"
        );
    }

    #[test]
    fn a_storage_dial_the_embedder_ignores_is_refused_too() {
        // `commit` and `location` were refused for reading as guarantees while
        // doing nothing; `dims` and `quant` had the same property and were
        // exempt. Someone setting `quant = "binary"` to shrink a committed
        // index got a full-precision one and no sign the request was dropped.
        for (key, line) in [
            ("dimension `dims`", "dims = 256"),
            ("dimension `quant`", "quant = \"binary\""),
        ] {
            let text = format!(
                "version = 1\n[dimensions.semantic]\ndescription = \"x\"\n\
                 embedder = {{ impl = \"distilled\" }}\n{line}\n"
            );
            let err = Config::parse(&text).unwrap_err();
            assert!(
                matches!(&err, Error::UnimplementedSetting { setting, .. } if *setting == key),
                "{line} should be refused, got {err:?}"
            );
        }

        // And the same config without them still parses, so this refuses the
        // dial rather than the dimension.
        assert!(Config::parse(
            "version = 1\n[dimensions.semantic]\ndescription = \"x\"\n\
             embedder = { impl = \"distilled\" }\n"
        )
        .is_ok());
    }

    #[test]
    fn a_fabricator_over_a_dimension_that_stores_nothing_is_refused() {
        // Found by building it: the groups form, the classifier is handed empty
        // items because there are no stored descriptors to summarize, every
        // label comes back blank, and the run reports success with no clusters.
        // Silent, and it looks exactly like a corpus with no structure.
        let text = r#"
version = 1
[dimensions.semantic]
description = "content resembling your target"
classifier = { impl = "content", persist_output = false }
embedder = { impl = "distilled" }
fabricator = { impl = "agglomerative" }
"#;
        let err = Config::parse(text).unwrap_err();
        assert!(
            matches!(&err, Error::UnfabricatableDimension(n) if n == "semantic"),
            "{err:?}"
        );

        // The same dimension without a fabricator is the supported shape.
        assert!(
            Config::parse(&text.replace("fabricator = { impl = \"agglomerative\" }\n", "")).is_ok()
        );
    }

    /// A dimension with every stage a fabricator needs, plus whatever
    /// `fabricator` line the test wants to try.
    fn with_fabricator(line: &str) -> Result<Config> {
        Config::parse(&format!(
            "version = 1\n[dimensions.semantic]\ndescription = \"x\"\n\
             classifier = {{ impl = \"llm\", prompts = {{ \"0\" = \"p\" }} }}\n\
             embedder = {{ impl = \"distilled\" }}\n{line}\n"
        ))
    }

    #[test]
    fn the_ripgrep_fabricator_still_parses() {
        // The shipped shape. Every refusal below must leave it alone.
        let cfg = with_fabricator(
            "fabricator = { impl = \"agglomerative\", threshold = 0.70, min_cluster = 5, \
             max_cluster = 15, max_levels = 3, min_remaining = 8 }",
        )
        .unwrap();
        assert_eq!(
            cfg.dimensions["semantic"]
                .search_levels()
                .into_iter()
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
    }

    #[test]
    fn a_fabricator_implementation_the_indexer_cannot_run_is_refused() {
        // The indexer builds an agglomerative fabricator and nothing else, so
        // any other name produced no clusters and said nothing — the silent
        // partial index spec §9.3 forbids.
        let err = with_fabricator("fabricator = { impl = \"hdbscan\" }").unwrap_err();
        assert!(
            matches!(&err, Error::UnimplementedSetting { setting, .. }
                if setting == "dimensions.semantic.fabricator.impl"),
            "{err:?}"
        );
    }

    #[test]
    fn a_fabricator_with_no_vectors_to_cluster_is_refused() {
        // Clustering is agglomerative on a cosine threshold; with no embedder
        // there is nothing to measure similarity between.
        let text = r#"
version = 1
[dimensions.semantic]
description = "x"
classifier = { impl = "llm", prompts = { "0" = "p" } }
fabricator = { impl = "agglomerative" }
"#;
        let err = Config::parse(text).unwrap_err();
        assert!(
            matches!(&err, Error::UnclusterableDimension(n) if n == "semantic"),
            "{err:?}"
        );
    }

    #[test]
    fn a_fabricator_tunable_outside_its_range_is_refused() {
        // Each of these was absorbed: the indexer reads them with
        // `as_integer`/`as_float` and falls back on anything that does not
        // parse, so the run builds a tree nobody configured.
        for (setting, line) in [
            ("threshold", "threshold = 0"),
            ("threshold", "threshold = 1.5"),
            ("threshold", "threshold = \"0.7\""),
            ("min_cluster", "min_cluster = 1"),
            ("min_cluster", "min_cluster = 0.5"),
            ("max_cluster", "min_cluster = 5, max_cluster = 3"),
            ("max_cluster", "max_cluster = 1"),
            ("max_levels", "max_levels = 0"),
            ("max_levels", "max_levels = \"3\""),
            ("max_levels", "max_levels = 70000"),
            ("min_remaining", "min_remaining = 0"),
        ] {
            let err = with_fabricator(&format!(
                "fabricator = {{ impl = \"agglomerative\", {line} }}"
            ))
            .unwrap_err();
            assert!(
                matches!(&err, Error::InvalidFabricatorSetting { setting: s, .. } if *s == setting),
                "{line} should be refused, got {err:?}"
            );
        }

        // `max_cluster = 0` is unbounded, not a mistake; `threshold = 1` is the
        // exact-duplicates-only end of the range.
        assert!(with_fabricator(
            "fabricator = { impl = \"agglomerative\", min_cluster = 5, max_cluster = 0, \
             threshold = 1 }"
        )
        .is_ok());
    }

    #[test]
    fn a_classifier_that_persists_nothing_emits_no_descriptor() {
        // The searchability question: a pass-through leaves nothing on disk to
        // match against, so the dimension rests entirely on its tensor.
        let text = r#"
version = 1
[dimensions.semantic]
description = "content resembling your target"
classifier = { impl = "content", persist_output = false }
embedder = { impl = "distilled" }
"#;
        let dim = &Config::parse(text).unwrap().dimensions["semantic"];
        assert!(!dim.emits_descriptor());
        assert!(dim.emits_tensor());
        // But the embedder still encodes what that classifier produced, which
        // is the whole reason the stage is named rather than implicit.
        assert!(dim.embeds_descriptor());
    }

    #[test]
    fn a_pass_through_classifier_alone_leaves_the_dimension_empty() {
        // Nothing stored and nothing embedded is nothing searchable, and must
        // be refused the same way a stageless dimension is.
        let text = r#"
version = 1
[dimensions.semantic]
description = "content resembling your target"
classifier = { impl = "content", persist_output = false }
"#;
        let err = Config::parse(text).unwrap_err();
        assert!(matches!(err, Error::EmptyDimension(n) if n == "semantic"));
    }

    #[test]
    fn persist_output_defaults_on_and_leaves_the_fingerprint_untouched() {
        // Adding the field must not have re-keyed every object in every
        // existing index — that would be a full LLM re-bill for a no-op.
        let text = r#"
version = 1
[dimensions.descriptive]
description = "a description of your target"
classifier = { impl = "llm", prompts = { "0" = "p" } }
"#;
        let dim = &Config::parse(text).unwrap().dimensions["descriptive"];
        assert!(dim.classifier.as_ref().unwrap().persist_output);
        assert!(
            !crate::codec::canonical_json(&dim.classifier)
                .unwrap()
                .windows(14)
                .any(|w| w == b"persist_output"),
            "a defaulted persist_output must not reach the fingerprint bytes"
        );
    }

    #[test]
    fn turning_persist_output_off_does_move_the_fingerprint() {
        // It changes what a run produces, so it has to invalidate.
        let base = r#"
version = 1
[dimensions.semantic]
description = "content resembling your target"
classifier = { impl = "content" }
embedder = { impl = "distilled" }
"#;
        let off = base.replace(
            r#"impl = "content" }"#,
            r#"impl = "content", persist_output = false }"#,
        );
        let a = Config::parse(base).unwrap().dimensions["semantic"]
            .artifact_fingerprint()
            .unwrap();
        let b = Config::parse(&off).unwrap().dimensions["semantic"]
            .artifact_fingerprint()
            .unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn one_classifier_cannot_both_store_and_not_store_its_output() {
        // The two dimensions share a group, and a group produces one descriptor
        // object. Picking a winner would either write a corpus somebody asked
        // not to store or discard output somebody paid for.
        let text = r#"
version = 1
[dimensions.a]
description = "x"
classifier = { impl = "content", persist_output = false }
embedder = { impl = "distilled" }
[dimensions.b]
description = "y"
classifier = { impl = "content" }
embedder = { impl = "distilled" }
"#;
        let err = Config::parse(text).unwrap_err();
        assert!(
            matches!(&err, Error::MixedPersistence { implementation, .. } if implementation == "content"),
            "{err:?}"
        );
    }

    #[test]
    fn editing_a_prompt_changes_the_fingerprint() {
        // This is what scopes cache invalidation to exactly the affected group.
        let base = r#"
version = 1
[dimensions.descriptive]
description = "what the code does"
classifier = { impl = "model-api", prompts = { "0" = "describe behavior" } }
"#;
        let edited = base.replace("describe behavior", "describe behaviour precisely");

        let a = Config::parse(base).unwrap();
        let b = Config::parse(&edited).unwrap();
        assert_ne!(
            a.dimensions["descriptive"].artifact_fingerprint().unwrap(),
            b.dimensions["descriptive"].artifact_fingerprint().unwrap()
        );
    }

    #[test]
    fn editing_a_description_does_not_invalidate_artifacts() {
        // The description is model-facing prose users should tune freely.
        // If it fed the artifact fingerprint, fixing a typo would change
        // federation identity and force a full re-classify of the dimension.
        let base = r#"
version = 1
[dimensions.descriptive]
description = "what the code does"
classifier = { impl = "llm", prompts = { "0" = "describe behavior" } }
"#;
        let reworded = base.replace(
            r#"description = "what the code does""#,
            r#"description = "What this code does at runtime -- effects and control flow.""#,
        );

        let a = Config::parse(base).unwrap();
        let b = Config::parse(&reworded).unwrap();

        assert_eq!(
            a.dimensions["descriptive"].artifact_fingerprint().unwrap(),
            b.dimensions["descriptive"].artifact_fingerprint().unwrap(),
            "rewording a description must not invalidate stored artifacts"
        );
        assert_ne!(
            a.dimensions["descriptive"].config_fingerprint().unwrap(),
            b.dimensions["descriptive"].config_fingerprint().unwrap(),
            "the full-config fingerprint should still register the change"
        );
    }

    #[test]
    fn a_scorer_key_is_refused_rather_than_ignored() {
        // The scorer was config nothing read: which scorer runs is decided by
        // whether the dimension carries a tensor, so naming one was a setting
        // that could disagree with reality and never be contradicted. Removed
        // rather than validated, and `deny_unknown_fields` is what makes the
        // removal audible instead of silent.
        let text = r#"
version = 1
[dimensions.lexical]
description = "x"
classifier = { impl = "structural" }
scorer = { impl = "bm25" }
"#;
        assert!(Config::parse(text).is_err());
    }

    #[test]
    fn traversing_dimension_name_is_rejected() {
        // A dimension name becomes .map/cache/<name>.pack, and config.toml
        // arrives with a cloned repository.
        let text = r#"
version = 1
[dimensions."../../../etc"]
description = "escape"
classifier = { impl = "structural" }
"#;
        let err = Config::parse(text).unwrap_err();
        assert!(matches!(err, Error::InvalidDimensionName(_)));
    }

    #[test]
    fn disabled_dimension_name_is_still_validated() {
        let text = r#"
version = 1
[dimensions."../evil"]
description = "escape"
enabled = false
classifier = { impl = "structural" }
"#;
        assert!(matches!(
            Config::parse(text).unwrap_err(),
            Error::InvalidDimensionName(_)
        ));
    }

    #[test]
    fn accepts_ordinary_names_only() {
        assert!(valid_dimension_name("lexical"));
        assert!(valid_dimension_name("api_surface-2"));
        assert!(!valid_dimension_name(""));
        assert!(!valid_dimension_name("Descriptive")); // uppercase: case-folding filesystems
        assert!(!valid_dimension_name("a/b"));
        assert!(!valid_dimension_name(".."));
        assert!(!valid_dimension_name(&"x".repeat(65)));
    }

    #[test]
    fn future_config_version_is_rejected() {
        let text = r#"
version = 99
[dimensions.lexical]
description = "x"
classifier = { impl = "structural" }
"#;
        assert!(matches!(
            Config::parse(text).unwrap_err(),
            Error::UnsupportedVersion { kind: "config", .. }
        ));
    }

    #[test]
    fn fingerprint_is_stable_across_declaration_order() {
        let one = r#"
version = 1
[dimensions.alpha]
description = "a"
classifier = { impl = "structural" }
[dimensions.beta]
description = "b"
classifier = { impl = "structural" }
"#;
        let two = r#"
version = 1
[dimensions.beta]
description = "b"
classifier = { impl = "structural" }
[dimensions.alpha]
description = "a"
classifier = { impl = "structural" }
"#;
        let a = Config::parse(one).unwrap();
        let b = Config::parse(two).unwrap();
        assert_eq!(a.fingerprint().unwrap(), b.fingerprint().unwrap());
    }

    #[test]
    fn unknown_keys_are_rejected() {
        // Typos in a committed config should fail loudly, not silently
        // produce a different index than the author intended.
        let text = r#"
version = 1
[dimensions.lexical]
description = "x"
classifier = { impl = "structural" }
quantt = "binary"
"#;
        assert!(Config::parse(text).is_err());
    }

    fn with_segmenter(body: &str) -> Result<Config> {
        Config::parse(&format!(
            "version = 1\n[segmenter]\n{body}\n[dimensions.lexical]\n\
             description = \"x\"\nclassifier = {{ impl = \"structural\" }}\n"
        ))
    }

    #[test]
    fn the_segmenter_defaults_to_what_the_indexer_used_to_hardcode() {
        // The whole reason this section could be added without re-keying any
        // existing index: the segmenter's `config()` string, which feeds every
        // segment's object key, comes out identical to the hardcoded one.
        let s = Config::default().segmenter;
        assert_eq!(
            (s.implementation.as_str(), s.lines, s.overlap),
            ("window", 40, 8)
        );
    }

    #[test]
    fn declaring_the_defaults_explicitly_is_the_same_config() {
        // Somebody writing the current values down to start tuning from them
        // must not thereby invalidate their index.
        assert_eq!(
            with_segmenter("impl = \"window\"\nlines = 40\noverlap = 8").unwrap(),
            with_segmenter("").unwrap(),
        );
    }

    #[test]
    fn a_segmenter_that_would_emit_nothing_is_refused() {
        // `lines = 0` pushes no segment at all, so the corpus indexes to an
        // empty index and reports success — the quiet failure worth refusing.
        assert!(matches!(
            with_segmenter("lines = 0"),
            Err(Error::InvalidSegmenter {
                setting: "lines",
                ..
            })
        ));
    }

    #[test]
    fn overlap_at_or_above_lines_is_refused() {
        // The stride collapses to one line, so a 4,000-line resource becomes
        // ~4,000 segments. On an LLM dimension that is a bill, not a warning.
        for body in ["lines = 40\noverlap = 40", "lines = 40\noverlap = 64"] {
            assert!(
                matches!(
                    with_segmenter(body),
                    Err(Error::InvalidSegmenter {
                        setting: "overlap",
                        ..
                    })
                ),
                "{body} should be refused"
            );
        }
        assert!(with_segmenter("lines = 40\noverlap = 39").is_ok());
    }

    #[test]
    fn an_unimplemented_segmenter_is_refused_rather_than_ignored() {
        // Silently falling back to `window` would index the corpus under a
        // segmentation the author did not choose and could not see.
        assert!(matches!(
            with_segmenter("impl = \"paragraph\""),
            Err(Error::UnimplementedSetting { setting, .. }) if setting == "segmenter.impl"
        ));
    }
}
