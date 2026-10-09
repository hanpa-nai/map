//! The retriever: load a committed index and answer N-dimensional queries.
//!
//! The query is a set of per-dimension fields, not a single string. One
//! retriever sees every selected dimension at once and fuses their scores.
//!
//! # Loading
//!
//! The fast path memory-maps a derived pack per dimension (see
//! [`map_stages::pack`]): no per-object opens, no JSON parse, no hash verify.
//!
//! If a pack is missing — a fresh clone, a cleared cache — the index is rebuilt
//! from committed objects in memory. That path reads every object through
//! `get_verified`, because descriptors are attacker-authored text bound for
//! model context and input-addressed keys cannot self-verify.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use map_core::{CandidateScope, CorpusStats, Embedder, QueryBundle, QueryField, Scorer};
use map_format::Config;
use map_stages::cache;
use map_stages::pack::PackFile;
use map_stages::{DensePack, Pack};

/// Errors from loading or querying an index.
#[derive(Debug, thiserror::Error)]
pub enum QueryError {
    #[error("no .map directory found at or above {0} — run `map init` first")]
    NotInitialized(PathBuf),
    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("unknown dimension {name:?}; this index has: {available}")]
    UnknownDimension { name: String, available: String },
    #[error(
        "dimension {dimension:?} is dense and needs the distilled embedder plugin \
         (build with --features distilled and install the model) to encode a query"
    )]
    EmbedderUnavailable { dimension: String },
    #[error("index contains an unsafe resource key: {0:?}")]
    UnsafeResourceKey(String),
    #[error("two indexes share the origin {0:?}; an origin identifies a hit and must be unique")]
    DuplicateOrigin(Origin),
    #[error("no index in this federation has origin {0:?}")]
    UnknownOrigin(Origin),
    #[error(
        "dimension {name:?} is built differently in {a:?} and {b:?}; \
         same name, different fingerprint means the scores are not comparable"
    )]
    IncompatibleDimension { name: String, a: Origin, b: Origin },
    /// Refused rather than silently returning less than the index holds. Those
    /// records cost something to build — for an LLM dimension, money — and a
    /// query that never returns them looks identical to a corpus that has none.
    #[error(
        "dimension {name:?} declares levels {declared:?} but the index holds records at \
         {built:?}; level(s) {unsearched:?} would never be returned. Widen `levels` or \
         remove it to search everything that was built.",
        declared = .built.iter().filter(|l| !.unsearched.contains(l)).collect::<Vec<_>>()
    )]
    UnderDeclaredLevels {
        name: String,
        unsearched: Vec<u16>,
        built: Vec<u16>,
    },
    #[error(transparent)]
    Pack(#[from] map_stages::pack::PackError),
    #[error(transparent)]
    DensePack(#[from] map_stages::densepack::DensePackError),
    #[error(transparent)]
    Cache(#[from] cache::CacheError),
    #[error(transparent)]
    Format(#[from] map_format::Error),
    #[error(transparent)]
    Stage(#[from] map_core::Error),
    #[error("could not decode stored payload: {0}")]
    Decode(#[from] serde_json::Error),
}

/// One dimension's query: the text to match and its weight in the fused score.
///
/// Text and weight live in one value on purpose. Separate maps keyed by
/// dimension would be two sources of truth that must agree on their keys, with
/// nothing enforcing it.
#[derive(Clone, Debug)]
pub struct QueryTerm {
    pub text: String,
    /// `1.0` is neutral; `0.0` or negative excludes the dimension entirely — it
    /// is not even scored.
    pub weight: f32,
}

impl QueryTerm {
    pub fn new(text: impl Into<String>) -> Self {
        QueryTerm {
            text: text.into(),
            weight: 1.0,
        }
    }

    pub fn weighted(text: impl Into<String>, weight: f32) -> Self {
        QueryTerm {
            text: text.into(),
            weight,
        }
    }
}

/// An N-dimensional query. Every dimension is optional, so a partial query is
/// always valid.
pub type Query = BTreeMap<String, QueryTerm>;

/// Fuse per-dimension scores into one relevance: the weighted mean over the
/// dimensions that **apply** to this candidate.
///
/// `asked` must be the dimensions that could have scored this candidate, not
/// every dimension in the query. The distinction is the whole policy:
///
/// - A dimension that **has a record here and did not match** belongs in the
///   denominator. That is a real zero, and it is what makes a one-of-two match
///   worth about half a two-of-two match — agreement falls out of the
///   arithmetic rather than needing its own rule.
/// - A dimension that **has no record here at all** does not. Absence is not
///   disagreement, and counting it would mean a cluster only its own dimension
///   can represent is divided by every dimension in the query, making a strong
///   single-dimension match unable to outrank a mediocre multi-dimension one.
///
/// Callers derive applicability from level coverage — see
/// `Scorer::levels`.
///
/// This is the single definition of the policy: the CLI retriever and the eval
/// harness both call it, so the two cannot silently diverge.
pub fn fuse_weighted_mean(
    per_dimension: &BTreeMap<String, f32>,
    asked: &[String],
    weight_of: impl Fn(&str) -> f32,
) -> f32 {
    let denom: f32 = asked.iter().map(|d| weight_of(d)).sum();
    // `denom > 0.0` rejects zero, negatives, and NaN (every NaN comparison is
    // false); `is_finite` rejects a sum overflowed to +inf under absurd weights.
    // Together they keep the fused score finite and in [0, 1] — never a NaN,
    // which `total_cmp` would sort above every real hit.
    let usable_denom = denom.is_finite() && denom > 0.0;
    if !usable_denom {
        return 0.0;
    }
    let numer: f32 = per_dimension.iter().map(|(d, s)| weight_of(d) * s).sum();
    numer / denom
}

/// Which index a result came from.
///
/// Resource keys are root-relative, so a key alone does not identify anything
/// once more than one index is in play — every repository has a `src/lib.rs`.
pub type Origin = String;

/// A hit, with enough context to act on without a second lookup.
#[derive(Clone, Debug)]
pub struct Hit {
    /// The index this came from. Empty for a standalone [`Index`].
    pub origin: Origin,
    /// Canonical resource key for a segment; the cluster's object key for a
    /// cluster, which spans no single resource.
    pub resource: String,
    /// Byte offset into normalized content. Zero for a cluster.
    pub start: u32,
    pub end: u32,
    /// 0 for a segment, higher for a cluster. What a caller checks to decide
    /// whether the span is meaningful or whether to ask for
    /// [`Index::cluster_label`] instead.
    pub level: u16,
    pub score: f32,
    /// Per-dimension contribution, for explainability.
    pub per_dimension: BTreeMap<String, f32>,
}

/// Which fabric levels a query considers — retrieval "zoom".
///
/// Level is a scope filter, not a relevance signal: it selects *which* records
/// are in play, never how well they match. Segments are level 0; the
/// fabricator's clusters are 1 and above, each level a coarser view of the one
/// below.
///
/// [`Segments`](LevelFilter::Segments) remains the `Default` because it is what
/// [`Index::find`] means and what the benchmark baselines were measured
/// against. The CLI deliberately chooses [`All`](LevelFilter::All) instead, so
/// a bare query returns cluster overviews alongside precise spans.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum LevelFilter {
    /// Level 0 only. What [`Index::find`] uses and what the baselines froze.
    #[default]
    Segments,
    /// Every level above 0 — the fabricator's output, at any height.
    Clusters,
    /// Segments and clusters ranked together.
    All,
    /// Exactly these levels, e.g. `{1, 3, 4}`.
    ///
    /// Ordered so the same request always describes the same scope; the set is
    /// also what makes `--level 1,3` expressible, which neither
    /// [`Clusters`](LevelFilter::Clusters) nor a single height can say.
    Only(BTreeSet<u16>),
}

impl LevelFilter {
    fn allows(&self, level: u16) -> bool {
        match self {
            LevelFilter::Segments => level == 0,
            LevelFilter::Clusters => level > 0,
            LevelFilter::All => true,
            LevelFilter::Only(levels) => levels.contains(&level),
        }
    }
}

/// How an index was loaded — useful for diagnosing latency.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadPath {
    Mapped,
    /// Rebuilt from committed objects because a pack was missing.
    Rebuilt,
}

/// Stage implementations, resolved once and shared by every index that needs
/// them.
///
/// A stage implementation has no per-index state — an embedder is a weight
/// table and a tokenizer — so resolving one inside [`Index::open`] would load
/// the same ~130 MB model once per index in a fan-out. Implementations live
/// here and indexes borrow them.
///
/// Resolution is lazy and keyed by implementation name, because which
/// implementations are needed is a property of each index's config, not of the
/// registry. A failed resolution is cached as a failure: a missing model should
/// be looked for once, not once per index.
#[derive(Default)]
pub struct Stages {
    embedders: RefCell<BTreeMap<String, Option<Arc<dyn Embedder>>>>,
}

impl Stages {
    pub fn new() -> Self {
        Self::default()
    }

    /// Supply an implementation instead of resolving one.
    ///
    /// For a caller that already holds a loaded embedder — an embedding host, a
    /// long-lived frontend — and for tests, which cannot resolve a real model.
    pub fn insert_embedder(&self, implementation: &str, embedder: Arc<dyn Embedder>) {
        self.embedders
            .borrow_mut()
            .insert(implementation.to_owned(), Some(embedder));
    }

    /// The embedder for `implementation`, loading it on first request.
    pub fn embedder(&self, implementation: &str) -> Option<Arc<dyn Embedder>> {
        if let Some(cached) = self.embedders.borrow().get(implementation) {
            return cached.clone();
        }
        let resolved = load_embedder(implementation);
        self.embedders
            .borrow_mut()
            .insert(implementation.to_owned(), resolved.clone());
        resolved
    }
}

/// Candidates accumulated across every scorer and every index of one query.
///
/// The key carries the origin because resource keys are root-relative: without
/// it, repository A's `src/lib.rs` at bytes 0..1200 and repository B's are the
/// same entry, and the second `extend` silently overwrites the first — one
/// surviving hit whose score blends two unrelated files.
type Fused = BTreeMap<(Origin, String, u32, u32), (u16, BTreeMap<String, f32>)>;

/// Which asked-for dimensions hold records at each level.
///
/// The denominator of the fused mean, resolved per candidate from its level.
/// Unioned across a federation, because a dimension may reach a level in one
/// index and not another, and a candidate is only ever scored by the index that
/// holds it.
type Applicable = BTreeMap<u16, Vec<String>>;

/// Fuse, sort, and truncate accumulated candidates.
///
/// `applicable` maps a level to the dimensions that could score a candidate
/// there. It must cover every dimension the *query* put in play that reaches
/// that level, not the subset one index happened to carry — the weighted mean
/// divides by it, so an index holding one of two dimensions would otherwise
/// score its hits twice as high as an index that answered both.
fn rank(fused: Fused, applicable: &Applicable, query: &Query, limit: usize) -> Vec<Hit> {
    let empty: Vec<String> = Vec::new();
    let mut hits: Vec<Hit> = fused
        .into_iter()
        .map(
            |((origin, resource, start, end), (level, per_dimension))| Hit {
                origin,
                resource,
                start,
                end,
                level,
                score: fuse_weighted_mean(
                    &per_dimension,
                    applicable.get(&level).unwrap_or(&empty),
                    |d| query.get(d).map_or(1.0, |t| t.weight),
                ),
                per_dimension,
            },
        )
        .collect();

    // Ties break by origin, resource, then offset so repeated queries agree.
    hits.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then(a.origin.cmp(&b.origin))
            .then(a.resource.cmp(&b.resource))
            .then(a.start.cmp(&b.start))
    });
    hits.truncate(limit);
    hits
}

/// Whether `bytes` are the pack bytes this machine recorded a marker for.
///
/// Called on the raw file image before anything parses it, so bytes that are
/// not even a pack are rebuilt rather than reported as malformed.
///
/// `rebuildable` says a manifest exists: without one there is nothing to
/// rebuild from, so the pack is all there is and is used as-is. With one, an
/// unmatched or uncomputable marker means "cannot confirm this pack" and the
/// dimension is rebuilt from the verified committed objects instead.
fn vouched(
    map_dir: &Path,
    dimension: &str,
    bytes: &[u8],
    markers: &cache::PackMarkers,
    rebuildable: bool,
) -> bool {
    if !rebuildable {
        return true;
    }
    cache::pack_marker(map_dir, dimension, bytes)
        .is_some_and(|marker| markers.trusts(dimension, &marker))
}

/// An index loaded and ready to answer queries.
///
/// Scorers are held behind [`map_core::Scorer`] rather than as concrete packs,
/// so a tensor-backed dimension slots in beside a BM25 one without the
/// retriever knowing the difference.
pub struct Index {
    root: PathBuf,
    origin: Origin,
    scorers: BTreeMap<String, Box<dyn Scorer>>,
    dimensions: Vec<String>,
    /// Artifact fingerprint per dimension, from the manifest.
    ///
    /// A dimension's identity is its name *and* this, never the name alone: two
    /// `descriptive` dimensions built from different prompts describe their
    /// material in different terms, and averaging their scores produces a number
    /// with no meaning. Empty when no manifest has been written yet.
    identities: BTreeMap<String, map_format::Fingerprint>,
    /// The committed manifest, parsed once at open.
    ///
    /// Held because it carries the content hash of every object, and reading a
    /// stored record without one means reading it unverified. `None` for an
    /// index that has been configured but never built.
    manifest: Option<map_format::Manifest>,
    /// Fabric levels each dimension searches, from its config.
    ///
    /// A query's level scope intersects with this per dimension: the
    /// intersection is what that dimension scores, and a dimension whose
    /// intersection is empty contributes nothing *and* stays out of the fused
    /// denominator, since it could not have scored those candidates.
    search_levels: BTreeMap<String, BTreeSet<u16>>,
    /// Dimensions whose query text must be embedded rather than tokenized.
    dense: BTreeSet<String>,
    embedder: Option<Arc<dyn Embedder>>,
    load_path: LoadPath,
}

impl Index {
    /// Load the index rooted at or above `start`, with its own stage registry.
    ///
    /// Fine for a single index. Opening several this way resolves each
    /// implementation once per index — use [`Index::open_with`] and one shared
    /// [`Stages`] for a fan-out.
    pub fn open(start: &Path) -> Result<Self, QueryError> {
        Index::open_with(start, &Stages::new())
    }

    /// Load the index rooted at or above `start`, borrowing implementations
    /// from `stages`.
    pub fn open_with(start: &Path, stages: &Stages) -> Result<Self, QueryError> {
        let root = map_core::find_map_root(start)
            .ok_or_else(|| QueryError::NotInitialized(start.to_path_buf()))?;
        let map_dir = root.join(".map");

        let config_path = map_dir.join("config.toml");
        let config_text = std::fs::read_to_string(&config_path).map_err(|e| QueryError::Io {
            path: config_path,
            source: e,
        })?;
        let config = Config::parse(&config_text)?;
        let dimensions: Vec<String> = config.active().map(|(n, _)| n.clone()).collect();

        // A dimension is dense iff it has an embedder: its pack is a DensePack
        // and its query text must be embedded, not tokenized.
        let dense: BTreeSet<String> = config
            .active()
            .filter(|(_, d)| d.emits_tensor())
            .map(|(n, _)| n.clone())
            .collect();

        let search_levels: BTreeMap<String, BTreeSet<u16>> = config
            .active()
            .map(|(n, d)| (n.clone(), d.search_levels()))
            .collect();

        let mut scorers: BTreeMap<String, Box<dyn Scorer>> = BTreeMap::new();
        let mut load_path = LoadPath::Mapped;

        // Trust a mmap'd pack only when this machine recorded a marker for
        // *those exact bytes* under the manifest now on disk. A marker over the
        // manifest alone left the packs themselves unhashed, and `cache/` is
        // gitignored — git overwrites ignored files on pull without a word — so
        // a `cache/lexical.pack` force-committed by a hostile clone was mapped
        // as trusted under an untouched manifest. The markers are per pack
        // because they bind pack bytes, so they can only be checked one pack at
        // a time. Hashing all three of ripgrep's packs — 12.8 MB — measured
        // 8.2 ms release: that is the price of the check, paid once per open.
        // Fail closed — only a *missing* manifest
        // trusts a pack as-is, since there is then nothing to rebuild from, and
        // a pack alone grants no more than the authored objects the rebuild
        // path already sanitizes on read.
        //
        // The marker is checked over the *unparsed* file, before the bytes are
        // interpreted. Parsing first made an unparseable substitute abort the
        // whole query with "malformed pack" — bytes nothing vouches for are
        // untrusted whatever they contain, and only a pack this machine built
        // has earned the right to fail loudly on corruption.
        let cache_dir = map_dir.join("cache");
        let rebuildable = cache::has_manifest(&map_dir);
        let mut markers = cache::PackMarkers::read(&cache_dir);

        for dimension in &dimensions {
            let path = cache_dir.join(format!("{dimension}.pack"));
            if !path.is_file() {
                load_path = LoadPath::Rebuilt;
                continue;
            }
            let file = PackFile::open(&path).map_err(|e| QueryError::Io {
                path: path.clone(),
                source: e,
            })?;
            if !vouched(&map_dir, dimension, file.bytes(), &markers, rebuildable) {
                load_path = LoadPath::Rebuilt;
                continue;
            }
            // A trusted pack this build cannot read is treated as a pack that is
            // not there: it is gitignored derived cache, rebuildable from the
            // objects, and costing a rebuild is the right price for a format
            // bump. Only a *version* mismatch qualifies — corruption still
            // fails loudly, because silently rebuilding over bytes this machine
            // vouched for would hide a real problem.
            let opened: Option<Box<dyn Scorer>> = if dense.contains(dimension) {
                match DensePack::from_file(file, dimension) {
                    Ok(pack) => Some(Box::new(pack) as Box<dyn Scorer>),
                    Err(e) if e.is_rebuildable() => None,
                    Err(e) => return Err(e.into()),
                }
            } else {
                match Pack::from_file(file, dimension) {
                    Ok(pack) => Some(Box::new(pack) as Box<dyn Scorer>),
                    Err(e) if e.is_rebuildable() => None,
                    Err(e) => return Err(e.into()),
                }
            };
            match opened {
                Some(scorer) => {
                    scorers.insert(dimension.clone(), scorer);
                }
                None => load_path = LoadPath::Rebuilt,
            }
        }

        // Rebuild every missing dimension, then **write it through** to the
        // cache so the next query mmaps instead of rebuilding. Without this a
        // fresh clone pays load+verify+build on every CLI call until an explicit
        // `map index` runs — the cache would never fill itself.
        if load_path == LoadPath::Rebuilt {
            let missing: Vec<String> = dimensions
                .iter()
                .filter(|d| !scorers.contains_key(*d))
                .cloned()
                .collect();

            // A read-only cache (CI, a read-only mount) must not fail the query.
            let cache_writable = std::fs::create_dir_all(&cache_dir).is_ok();

            for (dimension, bytes) in cache::rebuild_packs(&map_dir, &missing, &dense, true)? {
                if cache_writable {
                    let path = cache_dir.join(format!("{dimension}.pack"));
                    if cache::write_pack(&path, &bytes).is_ok() {
                        match cache::pack_marker(&map_dir, &dimension, &bytes) {
                            Some(marker) => markers.insert(&dimension, marker),
                            // Nothing to vouch with. Drop any stale marker
                            // rather than leave one describing the bytes this
                            // write just replaced.
                            None => markers.remove(&dimension),
                        }
                    }
                }
                // Open from the bytes just built, whether or not the write landed.
                let scorer: Box<dyn Scorer> = if dense.contains(&dimension) {
                    Box::new(DensePack::from_bytes(bytes, &dimension)?)
                } else {
                    Box::new(Pack::from_bytes(bytes, &dimension)?)
                };
                scorers.insert(dimension, scorer);
            }

            if cache_writable {
                let _ = markers.write(&cache_dir);
            }
        }

        // Only ask the registry for implementations this index actually
        // configures, so a lexical-only index never triggers a model load.
        let embedder = config
            .active()
            .filter(|(n, _)| dense.contains(*n))
            .find_map(|(_, d)| d.embedder.as_ref())
            .and_then(|e| stages.embedder(&e.implementation));

        // Best-effort: an index that has been configured but never built has no
        // manifest, and that is not a query-time error.
        let manifest = std::fs::read(map_dir.join("manifest.json"))
            .ok()
            .and_then(|bytes| map_format::Manifest::from_bytes(&bytes).ok());

        // Refuse a config that searches less than was built. Those records are
        // on disk and referenced by the manifest, but no query would ever return
        // them — for an LLM dimension that is silently discarding what somebody
        // paid for. Only an *explicit* `levels` can be wrong: when it is
        // omitted, `search_levels()` derives the range from the stages and is
        // accurate by construction, and a fabricator legitimately stops short of
        // `max_levels` on a diffuse corpus.
        if let Some(manifest) = manifest.as_ref() {
            for (name, dimension) in config.active() {
                if dimension.levels.is_none() {
                    continue;
                }
                let Some(identity) = manifest.dimensions.get(name) else {
                    continue;
                };
                // Empty means a manifest written before built levels were
                // recorded — no information, so nothing to check against.
                let built: BTreeSet<u16> = identity.levels.iter().copied().collect();
                if built.is_empty() {
                    continue;
                }
                let declared = dimension.search_levels();
                let unsearched: Vec<u16> = built.difference(&declared).copied().collect();
                if !unsearched.is_empty() {
                    return Err(QueryError::UnderDeclaredLevels {
                        name: name.clone(),
                        unsearched,
                        built: built.into_iter().collect(),
                    });
                }
            }
        }

        let identities = manifest
            .as_ref()
            .map(|m| {
                m.dimensions
                    .iter()
                    .map(|(name, identity)| (name.clone(), identity.fingerprint))
                    .collect()
            })
            .unwrap_or_default();

        Ok(Index {
            root,
            origin: Origin::new(),
            scorers,
            dimensions,
            identities,
            manifest,
            search_levels,
            dense,
            embedder,
            load_path,
        })
    }

    /// Label this index's hits. Set by [`Federation`]; empty standalone.
    pub fn with_origin(mut self, origin: impl Into<Origin>) -> Self {
        self.origin = origin.into();
        self
    }

    pub fn origin(&self) -> &str {
        &self.origin
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn dimensions(&self) -> &[String] {
        &self.dimensions
    }

    /// How many cluster records the manifest reaches, across every dimension.
    ///
    /// Zero means no fabricator has built a tree here, so every level above 0
    /// is empty whatever `--level` asks for.
    pub fn cluster_count(&self) -> usize {
        self.manifest.as_ref().map_or(0, |m| m.clusters.len())
    }

    pub fn load_path(&self) -> LoadPath {
        self.load_path
    }

    /// Number of searchable records, across the largest dimension.
    pub fn len(&self) -> usize {
        self.scorers.values().map(|s| s.len()).max().unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Run an N-dimensional query over segments.
    ///
    /// Scores are fused as a weighted mean (see [`fuse_weighted_mean`]) — the
    /// reference policy. Anything smarter replaces this method without changing
    /// the call shape.
    pub fn find(&self, query: &Query, limit: usize) -> Result<Vec<Hit>, QueryError> {
        // Segments only by default: precise, and identical to the pre-cluster
        // behaviour the baselines were frozen against. Zoom is opt-in.
        self.find_at(query, limit, LevelFilter::Segments)
    }

    /// Run an N-dimensional query, restricted to a fabric level range.
    pub fn find_at(
        &self,
        query: &Query,
        limit: usize,
        levels: LevelFilter,
    ) -> Result<Vec<Hit>, QueryError> {
        for name in query.keys() {
            if !self.scorers.contains_key(name) {
                return Err(QueryError::UnknownDimension {
                    name: name.clone(),
                    available: self.dimensions.join(", "),
                });
            }
        }

        let bundle = self.encode(query)?;
        if bundle.is_empty() {
            return Ok(Vec::new());
        }

        // Stage 1 — fix the candidate scope, before any dimension is consulted.
        // Membership is a question of scope, never of relevance: letting a
        // dimension propose candidates would cap the result set at that
        // dimension's recall. `All` also lets each scorer skip building an
        // eligibility set sized to the corpus, measured at ~half of query
        // latency past 100k records.
        let asked: Vec<String> = bundle.keys().cloned().collect();
        if self.is_empty() {
            return Ok(Vec::new());
        }

        // No federated statistics for a lone index: its own *are* the totals,
        // so passing them would be the same arithmetic by a longer route.
        let mut fused = Fused::new();
        self.collect_into(&bundle, &levels, &BTreeMap::new(), &mut fused)?;

        let mut applicable = Applicable::new();
        self.note_applicable(&asked, &levels, &mut applicable);
        Ok(rank(fused, &applicable, query, limit))
    }

    /// Encode `query` into the bundle this index can score.
    ///
    /// Dimensions this index does not carry are skipped rather than refused —
    /// [`find_at`](Self::find_at) rejects them before calling this, but a
    /// federation asks every index for what it has.
    fn encode(&self, query: &Query) -> Result<QueryBundle, QueryError> {
        let mut bundle = QueryBundle::new();
        for (dimension, term) in query {
            if let Some(field) = self.encode_one(dimension, term)? {
                bundle.insert(dimension.clone(), field);
            }
        }
        Ok(bundle)
    }

    /// Encode one dimension, or `None` if this index cannot contribute it.
    fn encode_one(
        &self,
        dimension: &str,
        term: &QueryTerm,
    ) -> Result<Option<QueryField>, QueryError> {
        // A zero, negative, or NaN weight excludes a dimension outright: skip it
        // here so it is never embedded and never lands in the fused mean.
        // `weight > 0.0` is false for NaN too, so a NaN weight is excluded
        // rather than let through to poison every score.
        let usable = !term.text.trim().is_empty() && term.weight > 0.0;
        if !usable || !self.scorers.contains_key(dimension) {
            return Ok(None);
        }
        if !self.dense.contains(dimension) {
            return Ok(Some(QueryField::text(term.text.clone())));
        }
        let embedder = self
            .embedder
            .as_ref()
            .ok_or_else(|| QueryError::EmbedderUnavailable {
                dimension: dimension.to_owned(),
            })?;
        Ok(Some(QueryField {
            text: None,
            tensor: Some(embedder.encode_query(dimension, &term.text)?),
        }))
    }

    /// The levels `dimension` scores for this query: its configured levels
    /// intersected with the query's scope.
    ///
    /// An empty intersection means the dimension contributes nothing — no
    /// candidates, and no place in any denominator.
    fn scope_for(&self, dimension: &str, levels: &LevelFilter) -> BTreeSet<u16> {
        self.search_levels
            .get(dimension)
            .map(|declared| {
                declared
                    .iter()
                    .copied()
                    .filter(|level| levels.allows(*level))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Record, per level, which of `asked` this index can score there.
    ///
    /// Unions into `out` rather than replacing it, so a federation ends up with
    /// the true reach of each dimension across every member.
    fn note_applicable(&self, asked: &[String], levels: &LevelFilter, out: &mut Applicable) {
        for dimension in asked {
            for level in self.scope_for(dimension, levels) {
                let at = out.entry(level).or_default();
                if !at.contains(dimension) {
                    at.push(dimension.clone());
                }
            }
        }
    }

    /// Score `bundle` and accumulate into a shared candidate map.
    ///
    /// Keyed by origin as well as location, so two indexes holding the same
    /// path at the same offsets stay separate candidates.
    fn collect_into(
        &self,
        bundle: &QueryBundle,
        levels: &LevelFilter,
        corpus: &BTreeMap<String, CorpusStats>,
        fused: &mut Fused,
    ) -> Result<(), QueryError> {
        for (dimension, scorer) in &self.scorers {
            if !bundle.contains_key(dimension) {
                continue;
            }
            let group = [dimension.clone()];
            let stats = corpus.get(dimension);
            // The candidate set for a dimension is its configured levels
            // intersected with the query's scope. Empty means it searches
            // nothing here — not that it searches everything.
            let scope = self.scope_for(dimension, levels);
            if scope.is_empty() {
                continue;
            }
            for (id, per_dimension) in scorer.score(CandidateScope::All, bundle, &group, stats)? {
                let Some(at) = scorer.located(id) else {
                    continue;
                };
                // Filters before fusion, so a scoped query never even sees the
                // scores of records outside it.
                if !scope.contains(&at.level) {
                    continue;
                }
                let entry = fused
                    .entry((
                        self.origin.clone(),
                        at.resource.to_owned(),
                        at.start,
                        at.end,
                    ))
                    .or_insert_with(|| (at.level, BTreeMap::new()));
                entry.1.extend(per_dimension);
            }
        }
        Ok(())
    }

    /// Read the text a hit refers to, with its starting line number.
    ///
    /// Re-normalizes line endings first: spans are offsets into *normalized*
    /// content, so on a CRLF checkout raw bytes would land in the wrong place
    /// (spec §6.1).
    pub fn snippet(&self, hit: &Hit) -> Result<(String, usize), QueryError> {
        map_stages::discover::check_resource_key(&hit.resource)
            .map_err(|_| QueryError::UnsafeResourceKey(hit.resource.clone()))?;

        let path = self.root.join(&hit.resource);
        let raw = std::fs::read_to_string(&path).map_err(|e| QueryError::Io { path, source: e })?;
        let text = map_stages::preprocess::normalize_line_endings(&raw);

        let (start, end) = clamp_span(&text, hit.start as usize, hit.end as usize);
        Ok((text[start..end].to_owned(), line_of(&text, hit.start)))
    }

    /// The label of a cluster hit, read from its stored record.
    ///
    /// A cluster spans no file, so its `resource` is its own object key.
    /// Returns `None` for a segment hit, or if the object is missing or fails
    /// verification.
    pub fn cluster_label(&self, hit: &Hit) -> Option<String> {
        if hit.level == 0 {
            return None;
        }
        let key = map_format::ObjectKey(map_format::Digest::from_hex(&hit.resource).ok()?);
        let store = map_format::ObjectStore::open(self.root.join(".map/index/cluster/objects"));
        let record: map_format::Record =
            serde_json::from_slice(&self.read_verified(&store, key)?).ok()?;
        record.descriptor
    }

    /// Read a stored object, checked against the manifest's content hash.
    ///
    /// `ObjectStore::get` is the unverified path, and what it returns must not
    /// reach a model's context: a committed index is untrusted data that steers
    /// attention (spec §8), and an input-addressed key cannot self-verify
    /// (spec §4) — the manifest's per-object hash is the only thing that can.
    /// A cluster label is exactly that text, so it comes through here.
    ///
    /// `None` when there is no manifest, when it does not describe the object,
    /// or when the bytes on disk do not match what it records.
    fn read_verified(
        &self,
        store: &map_format::ObjectStore,
        key: map_format::ObjectKey,
    ) -> Option<Vec<u8>> {
        let entry = self.manifest.as_ref()?.object(key)?;
        store.get_verified(key, entry).ok()
    }

    /// Every level-0 segment beneath a cluster hit, as located spans.
    ///
    /// This is what a cluster *retrieves*: one hit standing for the whole
    /// subtree. Naming a component in a single result rather than ten
    /// speculative probes is the entire claim of the fabric, and without this
    /// there is no way to check it — a cluster's `resource` is its own object
    /// key, which resolves to no file.
    ///
    /// A leaf's id is `hash(resource + span, dimension fingerprint)` and is
    /// **one-way**, so the mapping has to be rebuilt by deriving the same id for
    /// every segment in the index. That makes this O(corpus) and deliberately a
    /// cold path — for explaining or scoring a cluster, never inside ranking.
    ///
    /// Returns `None` for a segment hit or when the cluster object is missing.
    pub fn cluster_members(&self, hit: &Hit) -> Option<Vec<Located>> {
        if hit.level == 0 {
            return None;
        }
        let clusters = map_format::ObjectStore::open(self.root.join(".map/index/cluster/objects"));
        let read = |key: map_format::ObjectKey| -> Option<map_format::Record> {
            serde_json::from_slice(&self.read_verified(&clusters, key)?).ok()
        };

        let root_key = map_format::ObjectKey(map_format::Digest::from_hex(&hit.resource).ok()?);
        let dimension = read(root_key)?.meta.dimension;
        let leaf_ids = leaf_ids_of(root_key, &read, &mut BTreeSet::new())?;
        let by_id = self.level0_ids(&dimension)?;

        let mut out: Vec<Located> = leaf_ids
            .iter()
            .filter_map(|id| by_id.get(id).cloned())
            .collect();
        out.sort_by(|a, b| a.resource.cmp(&b.resource).then(a.start.cmp(&b.start)));
        out.dedup();
        Some(out)
    }

    /// Rebuild `level-0 id -> located span` for one dimension.
    fn level0_ids(&self, dimension: &str) -> Option<BTreeMap<map_format::ObjectKey, Located>> {
        let dim_fp = *self.identities.get(dimension)?;
        let manifest = self.manifest.as_ref()?;
        let shared = map_format::ObjectStore::open(self.root.join(".map/index/shared/objects"));

        let mut out = BTreeMap::new();
        for (resource, root) in &manifest.roots {
            let Some(bytes) = self.read_verified(&shared, root.segments) else {
                continue;
            };
            let Ok(payload) = serde_json::from_slice::<map_core::SegmentsPayload>(&bytes) else {
                continue;
            };
            for segment in &payload.segments {
                // The indexer's `fabricate` derives a leaf id with this same
                // call. Both sides must go through this one function: a leaf id
                // is one-way, so the only way a query resolves a cluster's
                // children back to spans is by deriving byte-identical ids.
                let id = map_format::ObjectKey::segment(
                    resource,
                    segment.start,
                    segment.end,
                    root.segments,
                    dim_fp,
                );
                out.insert(
                    id,
                    Located {
                        resource: resource.clone(),
                        start: segment.start,
                        end: segment.end,
                    },
                );
            }
        }
        Some(out)
    }
}

/// A resolved span: where a cluster's member actually lives.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Located {
    pub resource: String,
    pub start: u32,
    pub end: u32,
}

/// Mirrors the indexer: `None` when the plugin is not compiled or the model is
/// absent. A dense dimension then cannot be *queried* even though its pack
/// loaded; the lexical dimensions still work.
#[cfg(feature = "distilled")]
fn load_embedder(implementation: &str) -> Option<Arc<dyn Embedder>> {
    if implementation != "distilled" {
        return None;
    }
    let home = std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
        .unwrap_or_default();
    let dir = home.join(".map/models/potion-retrieval-32M");
    map_embed::DistilledEmbedder::from_dir(&dir, "potion-retrieval-32M")
        .ok()
        .map(|e| Arc::new(e) as Arc<dyn Embedder>)
}

#[cfg(not(feature = "distilled"))]
fn load_embedder(_implementation: &str) -> Option<Arc<dyn Embedder>> {
    None
}

/// Roots listed in `~/.map/config.toml`.
///
/// User-global rather than committed: which repositories you search is a
/// property of your checkout, and a path from one machine means nothing on
/// another. Writing a root here *is* the request to search it, so these apply
/// to every query — the origin on each hit says where it came from.
///
/// Lives here rather than in `map-format` because it is not part of the
/// on-disk format, and here rather than in the CLI because every frontend
/// needs the same set.
///
/// Unreadable or malformed config yields no roots rather than failing a query:
/// it is a convenience list, not an index.
pub fn configured_roots() -> Vec<PathBuf> {
    #[derive(serde::Deserialize, Default)]
    struct UserConfig {
        #[serde(default)]
        roots: Vec<Root>,
    }
    #[derive(serde::Deserialize)]
    struct Root {
        path: PathBuf,
    }

    let Some(home) = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) else {
        return Vec::new();
    };
    let path = PathBuf::from(home).join(".map").join("config.toml");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    toml::from_str::<UserConfig>(&text)
        .map(|c| c.roots.into_iter().map(|r| r.path).collect())
        .unwrap_or_default()
}

/// Several indexes answering one query.
///
/// A `.map` covers only its own root (spec §1); spanning roots is composition,
/// never a wider index. So this holds independent [`Index`] values and merges
/// their results — it never merges their data.
///
/// Every index shares one [`Stages`], so an implementation is resolved once no
/// matter how many roots are in play, and the query is encoded once and handed
/// to all of them.
pub struct Federation {
    indexes: Vec<Index>,
    stages: Stages,
}

impl Federation {
    /// Open every `(origin, root)` pair into one federation.
    ///
    /// An origin must be unique — it is half the identity of every hit.
    pub fn open<I, O, P>(roots: I) -> Result<Self, QueryError>
    where
        I: IntoIterator<Item = (O, P)>,
        O: Into<Origin>,
        P: AsRef<Path>,
    {
        let stages = Stages::new();
        let mut indexes = Vec::new();
        let mut seen: BTreeSet<Origin> = BTreeSet::new();
        for (origin, root) in roots {
            let origin = origin.into();
            if !seen.insert(origin.clone()) {
                return Err(QueryError::DuplicateOrigin(origin));
            }
            indexes.push(Index::open_with(root.as_ref(), &stages)?.with_origin(origin));
        }
        let federation = Federation { indexes, stages };
        federation.check_compatibility()?;
        Ok(federation)
    }

    /// Refuse a federation where one dimension name means two different things.
    ///
    /// Fusing them would average scores from dimensions built by different
    /// prompts or embedders — arithmetically fine and semantically empty. The
    /// query names a dimension by its bare name, so there is no way to ask for
    /// one and not the other; refusing is the only honest answer.
    fn check_compatibility(&self) -> Result<(), QueryError> {
        let mut seen: BTreeMap<&str, (&Origin, &map_format::Fingerprint)> = BTreeMap::new();
        for index in &self.indexes {
            for (name, fingerprint) in &index.identities {
                match seen.get(name.as_str()) {
                    Some((origin, known)) if *known != fingerprint => {
                        return Err(QueryError::IncompatibleDimension {
                            name: name.clone(),
                            a: (*origin).clone(),
                            b: index.origin.clone(),
                        });
                    }
                    Some(_) => {}
                    None => {
                        seen.insert(name, (&index.origin, fingerprint));
                    }
                }
            }
        }
        Ok(())
    }

    pub fn stages(&self) -> &Stages {
        &self.stages
    }

    pub fn indexes(&self) -> &[Index] {
        &self.indexes
    }

    /// Every dimension any member carries, in canonical order.
    ///
    /// The query surface is the union: an index scores the parameters it has
    /// and ignores the rest (spec §7). A dimension missing from one index is
    /// not an error, because federating a lexical-only repo with a semantic one
    /// is the normal case.
    pub fn dimensions(&self) -> Vec<String> {
        let mut all: BTreeSet<&str> = BTreeSet::new();
        for index in &self.indexes {
            all.extend(index.dimensions.iter().map(String::as_str));
        }
        all.into_iter().map(str::to_owned).collect()
    }

    /// Cluster records across every member; see [`Index::cluster_count`].
    pub fn cluster_count(&self) -> usize {
        self.indexes.iter().map(Index::cluster_count).sum()
    }

    /// Run an N-dimensional query over segments across every index.
    pub fn find(&self, query: &Query, limit: usize) -> Result<Vec<Hit>, QueryError> {
        self.find_at(query, limit, LevelFilter::Segments)
    }

    /// Run an N-dimensional query across every index, at a fabric level range.
    pub fn find_at(
        &self,
        query: &Query,
        limit: usize,
        levels: LevelFilter,
    ) -> Result<Vec<Hit>, QueryError> {
        // Reject a dimension no member has. Ignoring it silently would turn a
        // typo into a quietly narrower query.
        let available = self.dimensions();
        for name in query.keys() {
            if !available.iter().any(|d| d == name) {
                return Err(QueryError::UnknownDimension {
                    name: name.clone(),
                    available: available.join(", "),
                });
            }
        }

        // Encode once, not once per index: the text is identical, and encoding
        // is the one model call on the query path.
        let bundle = self.encode(query)?;
        if bundle.is_empty() {
            return Ok(Vec::new());
        }
        let asked: Vec<String> = bundle.keys().cloned().collect();

        // Sum every member's corpus statistics before anyone scores. BM25
        // normalizes against the corpus it saw, so without this an index that
        // has never seen a query term drops it from its ceiling and returns
        // *higher* scores than an index that matched the whole query.
        let corpus = self.corpus_stats(&bundle);

        // Accumulate every index into one candidate map, then rank once —
        // truncating per index would discard candidates the merge would have
        // promoted.
        let mut fused = Fused::new();
        let mut applicable = Applicable::new();
        for index in &self.indexes {
            if index.is_empty() {
                continue;
            }
            index.collect_into(&bundle, &levels, &corpus, &mut fused)?;
            index.note_applicable(&asked, &levels, &mut applicable);
        }
        Ok(rank(fused, &applicable, query, limit))
    }

    /// Federated corpus statistics per dimension.
    ///
    /// Costs one term lookup per query term per index over packs already
    /// memory-mapped, so it is a pre-pass in name only.
    fn corpus_stats(&self, bundle: &QueryBundle) -> BTreeMap<String, CorpusStats> {
        let mut totals: BTreeMap<String, CorpusStats> = BTreeMap::new();
        for index in &self.indexes {
            for (dimension, scorer) in &index.scorers {
                if !bundle.contains_key(dimension) {
                    continue;
                }
                let group = [dimension.clone()];
                if let Some(stats) = scorer.corpus_stats(bundle, &group) {
                    totals.entry(dimension.clone()).or_default().merge(&stats);
                }
            }
        }
        totals
    }

    /// Encode each queried dimension once, using whichever member carries it.
    fn encode(&self, query: &Query) -> Result<QueryBundle, QueryError> {
        let mut bundle = QueryBundle::new();
        for (dimension, term) in query {
            for index in &self.indexes {
                if let Some(field) = index.encode_one(dimension, term)? {
                    bundle.insert(dimension.clone(), field);
                    break;
                }
            }
        }
        Ok(bundle)
    }

    /// The index a hit came from, for [`Index::snippet`] and
    /// [`Index::cluster_label`] — both resolve against their own root.
    pub fn index_of(&self, hit: &Hit) -> Option<&Index> {
        self.indexes.iter().find(|i| i.origin == hit.origin)
    }

    /// Read the text a hit refers to, resolving against the right root.
    pub fn snippet(&self, hit: &Hit) -> Result<(String, usize), QueryError> {
        self.index_of(hit)
            .ok_or_else(|| QueryError::UnknownOrigin(hit.origin.clone()))?
            .snippet(hit)
    }

    /// The label of a cluster hit, resolved against the right root.
    pub fn cluster_label(&self, hit: &Hit) -> Option<String> {
        self.index_of(hit)?.cluster_label(hit)
    }

    /// Every level-0 segment beneath a cluster hit — see
    /// [`Index::cluster_members`]. Resolved against the index the hit came
    /// from, since a cluster's children are ids only that index can map back.
    pub fn cluster_members(&self, hit: &Hit) -> Option<Vec<Located>> {
        self.index_of(hit)?.cluster_members(hit)
    }
}

/// One-based line number of a byte offset in normalized content.
pub fn line_of(text: &str, offset: u32) -> usize {
    text.bytes()
        .take(offset as usize)
        .filter(|b| *b == b'\n')
        .count()
        + 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use map_core::{IndexedRecord, ScorerBuilder, Segment};
    use map_format::{Record, RecordKind, RecordMeta, Tensor};
    use map_stages::lexical::{encode_frequencies, tokenize};
    use map_stages::PackBuilder;

    struct TempMap(PathBuf);
    impl TempMap {
        fn new() -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};
            static N: AtomicU64 = AtomicU64::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!("map-zoom-{}-{n}", std::process::id()));
            std::fs::create_dir_all(path.join(".map/cache")).unwrap();
            TempMap(path)
        }
    }
    impl Drop for TempMap {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
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

    /// A lexical-only index whose pack holds one segment and one cluster that
    /// share the term "alpha". No embedder or model required.
    fn index_with_a_segment_and_a_cluster() -> (TempMap, Index) {
        let tmp = TempMap::new();
        // `levels` is declared because this fixture hand-builds a cluster
        // without a fabricator; a real config derives the same range from one.
        std::fs::write(
            tmp.0.join(".map/config.toml"),
            "version = 1\n[dimensions.lexical]\ndescription = \"x\"\nlevels = [0, 1]\n\
             classifier = { impl = \"structural\" }\n",
        )
        .unwrap();

        let mut builder = PackBuilder::new();
        builder.push(&IndexedRecord {
            id: 0,
            resource: "src/a.rs",
            segment: Segment { start: 0, end: 12 },
            record: &leveled("alpha beta", 0),
        });
        builder.push(&IndexedRecord {
            id: 1,
            resource: "clusterkeyhex",
            segment: Segment { start: 0, end: 0 },
            record: &leveled("alpha gamma", 1),
        });
        std::fs::write(tmp.0.join(".map/cache/lexical.pack"), builder.finish()).unwrap();

        let index = Index::open(&tmp.0).unwrap();
        (tmp, index)
    }

    /// An index whose config declares `levels` and whose manifest records
    /// `built` as what the fabricator actually produced.
    fn index_declaring(
        levels: Option<&str>,
        built: &[u16],
    ) -> (TempMap, Result<Index, QueryError>) {
        let tmp = TempMap::new();
        let declared = levels
            .map(|l| format!("levels = {l}\n"))
            .unwrap_or_default();
        let config_text = format!(
            "version = 1\n[dimensions.lexical]\ndescription = \"x\"\n{declared}\
             classifier = {{ impl = \"structural\" }}\n"
        );
        std::fs::write(tmp.0.join(".map/config.toml"), &config_text).unwrap();

        let config = Config::parse(&config_text).unwrap();
        let mut manifest = map_format::Manifest::new(&config, "test", 0).unwrap();
        manifest.dimensions.get_mut("lexical").unwrap().levels = built.to_vec();
        std::fs::write(
            tmp.0.join(".map/manifest.json"),
            manifest.to_bytes().unwrap(),
        )
        .unwrap();

        let opened = Index::open(&tmp.0);
        (tmp, opened)
    }

    #[test]
    fn a_corrupt_pack_still_fails_loudly() {
        // Rebuilding over corruption would hide a real problem, so only a
        // version mismatch is forgiven. "Corrupt" now means corrupt *and
        // trusted*: the marker is rewritten over the damaged bytes, because
        // bytes no marker vouches for are rebuilt rather than reported.
        let tmp = lexical_index_with_a_vouched_pack("alpha beta");
        let path = tmp.0.join(".map/cache/lexical.pack");
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[0..8].copy_from_slice(b"NOTAPACK");
        std::fs::write(&path, &bytes).unwrap();
        vouch_for(&tmp, &bytes);

        let Err(err) = Pack::open(&path, "lexical") else {
            panic!("bad magic must be rejected");
        };
        assert!(!err.is_rebuildable(), "bad magic is not a version problem");

        assert!(
            matches!(Index::open(&tmp.0), Err(QueryError::Pack(_))),
            "a pack this machine vouched for must not be silently rebuilt over"
        );
    }

    /// A lexical index whose *committed* objects hold one cluster record, with a
    /// pack and a marker vouching for it. Unlike the marker-only fixture this
    /// one has something to rebuild from, so a query survives losing the pack.
    fn lexical_index_over_a_committed_cluster(text: &str) -> TempMap {
        let tmp = TempMap::new();
        let config_text = "version = 1
[dimensions.lexical]
description = \"x\"
                           levels = [0, 1]
classifier = { impl = \"structural\" }
";
        std::fs::write(tmp.0.join(".map/config.toml"), config_text).unwrap();

        let record = leveled(text, 1);
        let object = serde_json::to_vec(&record).unwrap();
        let key = map_format::ObjectKey::cluster(&[], map_format::Fingerprint::of(b"lexical"));
        let store = map_format::ObjectStore::open(tmp.0.join(".map/index/cluster/objects"));
        let entry = store.put(key, &object, map_format::Tier::A).unwrap();

        let config = Config::parse(config_text).unwrap();
        let mut manifest = map_format::Manifest::new(&config, "test", 0).unwrap();
        manifest.insert_object(key, entry).unwrap();
        manifest.clusters.insert("lexical".to_owned(), vec![key]);
        std::fs::write(
            tmp.0.join(".map/manifest.json"),
            manifest.to_bytes().unwrap(),
        )
        .unwrap();

        let mut builder = PackBuilder::new();
        builder.push(&IndexedRecord {
            id: 0,
            resource: &key.to_string(),
            segment: Segment { start: 0, end: 0 },
            record: &record,
        });
        let bytes = builder.finish();
        std::fs::write(tmp.0.join(".map/cache/lexical.pack"), &bytes).unwrap();
        vouch_for(&tmp, &bytes);
        tmp
    }

    #[test]
    fn random_bytes_under_an_unchanged_manifest_are_rebuilt_not_served() {
        // Parsing before checking the marker meant bytes that are not a pack at
        // all aborted the whole query with "malformed pack". Nothing vouches for
        // them, so they are a missing pack whatever they contain — and the
        // gitignored cache is rebuilt from the committed objects.
        let tmp = lexical_index_over_a_committed_cluster("alpha beta");
        assert_eq!(
            Index::open(&tmp.0).unwrap().load_path(),
            LoadPath::Mapped,
            "the vouched pack is mapped before the bytes are replaced"
        );

        std::fs::write(
            tmp.0.join(".map/cache/lexical.pack"),
            vec![0x5au8; 2048], // not the pack magic
        )
        .unwrap();

        let index = match Index::open(&tmp.0) {
            Ok(index) => index,
            Err(e) => panic!("unparseable untrusted pack must be rebuilt, not reported: {e}"),
        };
        assert_eq!(index.load_path(), LoadPath::Rebuilt);
        assert_eq!(
            index
                .find_at(&q("alpha"), 10, LevelFilter::Clusters)
                .unwrap()
                .len(),
            1,
            "the rebuilt dimension must still answer the query"
        );
    }

    #[test]
    fn a_config_searching_less_than_was_built_is_refused() {
        // The silent failure this exists to stop: the clusters are on disk and
        // in the manifest, and every query quietly returns none of them.
        let (_tmp, opened) = index_declaring(Some("[0]"), &[0, 1, 2]);
        let Err(err) = opened else {
            panic!("declaring [0] against a 3-level index must refuse");
        };
        assert!(
            matches!(&err, QueryError::UnderDeclaredLevels { name, unsearched, .. }
                if name == "lexical" && unsearched == &[1, 2]),
            "{err:?}"
        );
        // The message has to name the levels that would vanish, or it cannot be
        // acted on. Rendering it also proves the format string compiles to
        // something sensible rather than merely compiling.
        let text = err.to_string();
        assert!(text.contains("[1, 2]"), "must name the lost levels: {text}");
        assert!(
            text.contains("declares levels [0]"),
            "must name what was declared, not render it empty: {text}"
        );
        assert!(
            text.contains("holds records at [0, 1, 2]"),
            "must name what exists: {text}"
        );
    }

    #[test]
    fn over_declaring_levels_is_still_harmless() {
        // A level nothing holds records at yields no candidates; refusing it
        // would break the fixture style that declares a range up front.
        let (_tmp, opened) = index_declaring(Some("[0, 1, 2, 3]"), &[0, 1]);
        if let Err(e) = opened {
            panic!("must open: {e}");
        }
    }

    #[test]
    fn derived_levels_are_never_refused() {
        // Omitted means derived from the stages, and a fabricator legitimately
        // stops short of max_levels on a diffuse corpus. Refusing that would
        // fail an index that is behaving exactly as designed.
        let (_tmp, opened) = index_declaring(None, &[0, 1, 2]);
        if let Err(e) = opened {
            panic!("must open: {e}");
        }
    }

    #[test]
    fn a_manifest_predating_built_levels_opens() {
        // Empty is "unknown", not "level 0 only" — an index built before the
        // field existed must not fail closed on a fact it never recorded.
        let (_tmp, opened) = index_declaring(Some("[0]"), &[]);
        if let Err(e) = opened {
            panic!("must open: {e}");
        }
    }

    /// Pack bytes holding one level-0 record of `text` at `src/a.rs`.
    fn pack_holding(text: &str) -> Vec<u8> {
        let mut builder = PackBuilder::new();
        builder.push(&IndexedRecord {
            id: 0,
            resource: "src/a.rs",
            segment: Segment {
                start: 0,
                end: text.len() as u32,
            },
            record: &leveled(text, 0),
        });
        builder.finish()
    }

    /// A lexical index with a manifest, a pack, and a marker vouching for
    /// exactly those pack bytes — the state a real `map index` leaves behind.
    fn lexical_index_with_a_vouched_pack(text: &str) -> TempMap {
        let tmp = TempMap::new();
        let config_text = "version = 1\n[dimensions.lexical]\ndescription = \"x\"\n\
                           classifier = { impl = \"structural\" }\n";
        std::fs::write(tmp.0.join(".map/config.toml"), config_text).unwrap();

        let config = Config::parse(config_text).unwrap();
        let manifest = map_format::Manifest::new(&config, "test", 0).unwrap();
        std::fs::write(
            tmp.0.join(".map/manifest.json"),
            manifest.to_bytes().unwrap(),
        )
        .unwrap();

        let bytes = pack_holding(text);
        let cache_dir = tmp.0.join(".map/cache");
        std::fs::write(cache_dir.join("lexical.pack"), &bytes).unwrap();

        vouch_for(&tmp, &bytes);
        tmp
    }

    /// Record this machine's marker for `bytes` as the pack of `tmp`.
    fn vouch_for(tmp: &TempMap, bytes: &[u8]) {
        let mut markers = cache::PackMarkers::default();
        markers.insert(
            "lexical",
            cache::pack_marker(&tmp.0.join(".map"), "lexical", bytes)
                .expect("a manifest and a machine key are both available in tests"),
        );
        markers.write(&tmp.0.join(".map/cache")).unwrap();
    }

    #[test]
    fn a_pack_substituted_under_an_unchanged_manifest_is_rebuilt_not_served() {
        // `cache/` is gitignored and git overwrites ignored files on pull, so a
        // hostile clone can force-commit its own perfectly valid pack beside an
        // untouched manifest and marker file. The marker binds the pack bytes,
        // so the substituted ones fail the check and the dimension is rebuilt
        // from the verified committed objects rather than mapped.
        let tmp = lexical_index_with_a_vouched_pack("alpha beta");

        let index = Index::open(&tmp.0).unwrap();
        assert_eq!(
            index.load_path(),
            LoadPath::Mapped,
            "a pack this machine vouched for is mapped"
        );
        assert_eq!(index.find(&q("alpha"), 10).unwrap().len(), 1);
        drop(index);

        std::fs::write(
            tmp.0.join(".map/cache/lexical.pack"),
            pack_holding("alpha smuggled"),
        )
        .unwrap();

        let index = Index::open(&tmp.0).unwrap();
        assert_eq!(
            index.load_path(),
            LoadPath::Rebuilt,
            "substituted pack bytes must not be trusted on a marker for other bytes"
        );
        assert!(
            index.find(&q("smuggled"), 10).unwrap().is_empty(),
            "nothing from the substituted pack may reach a result"
        );
    }

    /// A lexical index whose manifest describes one stored cluster object,
    /// the way a fabricated index does. Returns the cluster's object key.
    fn lexical_index_with_a_stored_cluster(label: &str) -> (TempMap, map_format::ObjectKey) {
        let tmp = TempMap::new();
        let config_text = "version = 1\n[dimensions.lexical]\ndescription = \"x\"\n\
                           classifier = { impl = \"structural\" }\n";
        std::fs::write(tmp.0.join(".map/config.toml"), config_text).unwrap();

        let bytes = serde_json::to_vec(&cluster_record(label)).unwrap();
        let key = map_format::ObjectKey::cluster(&[], map_format::Fingerprint::of(b"lexical"));
        let store = map_format::ObjectStore::open(tmp.0.join(".map/index/cluster/objects"));
        let entry = store.put(key, &bytes, map_format::Tier::A).unwrap();

        let config = Config::parse(config_text).unwrap();
        let mut manifest = map_format::Manifest::new(&config, "test", 0).unwrap();
        manifest.insert_object(key, entry).unwrap();
        std::fs::write(
            tmp.0.join(".map/manifest.json"),
            manifest.to_bytes().unwrap(),
        )
        .unwrap();
        (tmp, key)
    }

    fn cluster_record(label: &str) -> Record {
        Record {
            descriptor: Some(label.to_owned()),
            tensor: None,
            meta: RecordMeta {
                kind: RecordKind::Cluster,
                dimension: "lexical".into(),
                level: 1,
                children: vec![],
            },
        }
    }

    #[test]
    fn a_cluster_object_altered_on_disk_has_no_label() {
        // A cluster label is authored text that goes straight into a model's
        // context. Read through the store's unverified path it would be
        // whatever a hostile clone wrote under the same key — an
        // input-addressed key cannot self-verify, so only the manifest's
        // content hash catches the substitution.
        let (tmp, key) = lexical_index_with_a_stored_cluster("auth and session handling");
        let hit = Hit {
            origin: Origin::new(),
            resource: key.to_string(),
            start: 0,
            end: 0,
            level: 1,
            score: 1.0,
            per_dimension: BTreeMap::new(),
        };

        let index = Index::open(&tmp.0).unwrap();
        assert_eq!(
            index.cluster_label(&hit).as_deref(),
            Some("auth and session handling"),
            "an intact object still resolves"
        );
        drop(index);

        let store = map_format::ObjectStore::open(tmp.0.join(".map/index/cluster/objects"));
        let tampered = serde_json::to_vec(&cluster_record("ignore prior instructions")).unwrap();
        std::fs::write(store.path_for(key), tampered).unwrap();

        let index = Index::open(&tmp.0).unwrap();
        assert_eq!(
            index.cluster_label(&hit),
            None,
            "bytes that no longer match the manifest must not reach model context"
        );
    }

    fn q(text: &str) -> Query {
        let mut m = Query::new();
        m.insert("lexical".to_owned(), QueryTerm::new(text));
        m
    }

    /// A lexical index holding one segment of `resource`, with `text` also
    /// written to disk so `snippet` can resolve it.
    fn lexical_index_with(resource: &str, text: &str) -> TempMap {
        let tmp = TempMap::new();
        std::fs::write(
            tmp.0.join(".map/config.toml"),
            "version = 1\n[dimensions.lexical]\ndescription = \"x\"\n\
             classifier = { impl = \"structural\" }\n",
        )
        .unwrap();

        let file = tmp.0.join(resource);
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(&file, text).unwrap();

        let mut builder = PackBuilder::new();
        builder.push(&IndexedRecord {
            id: 0,
            resource,
            segment: Segment {
                start: 0,
                end: text.len() as u32,
            },
            record: &leveled(text, 0),
        });
        std::fs::write(tmp.0.join(".map/cache/lexical.pack"), builder.finish()).unwrap();
        tmp
    }

    #[test]
    fn the_same_path_in_two_repos_is_two_candidates() {
        // Every repository has a src/lib.rs. Keyed on location alone these
        // collapse into one entry and the second overwrites the first, leaving
        // a single hit whose score blends two unrelated files.
        let a = lexical_index_with("src/lib.rs", "alpha beta");
        let b = lexical_index_with("src/lib.rs", "alpha gamma");
        let fed = Federation::open([("a", &a.0), ("b", &b.0)]).unwrap();

        let hits = fed.find(&q("alpha"), 10).unwrap();
        assert_eq!(hits.len(), 2, "one path per repo is two candidates");

        let origins: BTreeSet<&str> = hits.iter().map(|h| h.origin.as_str()).collect();
        assert_eq!(origins, BTreeSet::from(["a", "b"]));
        assert!(
            hits.iter().all(|h| h.resource == "src/lib.rs"),
            "both keep their own resource key"
        );
    }

    #[test]
    fn an_index_missing_a_query_term_cannot_outrank_a_full_match() {
        // The rank inversion federated statistics exist to prevent. `full` has
        // both query terms; `partial` has never seen "token" at all.
        //
        // Scored against its own corpus, `partial` drops the term it lacks from
        // its ceiling — a smaller divisor — so its HALF match normalizes to
        // roughly what a WHOLE match scores elsewhere.
        let full = lexical_index_with("src/lib.rs", "refresh token");
        let partial = lexical_index_with("src/other.rs", "refresh alpha");

        let alone = Index::open(&partial.0).unwrap();
        let solo = alone.find(&q("refresh token"), 10).unwrap();
        let solo_score = solo[0].score;

        let fed = Federation::open([("full", &full.0), ("partial", &partial.0)]).unwrap();
        let hits = fed.find(&q("refresh token"), 10).unwrap();
        assert_eq!(hits.len(), 2);

        let score_of = |origin: &str| {
            hits.iter()
                .find(|h| h.origin == origin)
                .expect("both indexes contribute")
                .score
        };
        assert!(
            score_of("full") > score_of("partial"),
            "a full match must outrank a half match: full={} partial={}",
            score_of("full"),
            score_of("partial")
        );
        assert!(
            score_of("partial") < solo_score,
            "federating must deflate the half match, not leave it inflated: \
             alone={solo_score} federated={}",
            score_of("partial")
        );
    }

    #[test]
    fn federating_one_index_matches_querying_it_alone() {
        // Global statistics over a single index *are* its local statistics, so
        // the two code paths must not disagree. This is what makes the frozen
        // single-index baselines a regression gate for the federated path.
        let only = lexical_index_with("src/lib.rs", "refresh token session");

        let standalone = Index::open(&only.0).unwrap();
        let fed = Federation::open([("only", &only.0)]).unwrap();

        for text in ["refresh", "refresh token", "session token absent"] {
            let a = standalone.find(&q(text), 10).unwrap();
            let b = fed.find(&q(text), 10).unwrap();
            assert_eq!(a.len(), b.len(), "{text:?}: different hit counts");
            for (x, y) in a.iter().zip(&b) {
                assert_eq!(x.resource, y.resource, "{text:?}");
                assert_eq!(
                    x.score, y.score,
                    "{text:?}: federated score drifted from standalone"
                );
            }
        }
    }

    #[test]
    fn a_hit_resolves_against_its_own_root() {
        let a = lexical_index_with("src/lib.rs", "alpha beta");
        let b = lexical_index_with("src/lib.rs", "alpha gamma");
        let fed = Federation::open([("a", &a.0), ("b", &b.0)]).unwrap();

        for hit in fed.find(&q("alpha"), 10).unwrap() {
            let (text, _line) = fed.snippet(&hit).unwrap();
            let expected = match hit.origin.as_str() {
                "a" => "alpha beta",
                _ => "alpha gamma",
            };
            assert_eq!(text, expected, "snippet read the wrong repository");
        }
    }

    #[test]
    fn an_index_without_a_dimension_is_skipped_not_an_error() {
        // Federating a lexical-only repo with a lexical+dense one is ordinary;
        // the lexical-only member scores what it has and ignores the rest.
        let lexical = lexical_index_with("src/lib.rs", "alpha beta");
        let dense = dense_index();
        let stages = Stages::new();
        let counter = Arc::new(CountingEmbedder::new());
        stages.insert_embedder("fake", counter.clone() as Arc<dyn Embedder>);

        let fed = Federation {
            indexes: vec![
                Index::open_with(&lexical.0, &stages)
                    .unwrap()
                    .with_origin("l"),
                Index::open_with(&dense.0, &stages)
                    .unwrap()
                    .with_origin("d"),
            ],
            stages,
        };
        assert_eq!(
            fed.dimensions(),
            vec!["lexical".to_owned(), "semantic".to_owned()]
        );

        let mut query = Query::new();
        query.insert("lexical".to_owned(), QueryTerm::new("alpha"));
        query.insert("semantic".to_owned(), QueryTerm::new("alpha"));
        let hits = fed.find(&query, 10).unwrap();

        assert!(!hits.is_empty(), "the union query returns hits");
        assert_eq!(
            counter.encodes(),
            1,
            "one encode for the whole federation, not one per index"
        );
    }

    #[test]
    fn a_dimension_no_member_has_is_still_an_error() {
        let a = lexical_index_with("src/lib.rs", "alpha beta");
        let fed = Federation::open([("a", &a.0)]).unwrap();
        let mut query = Query::new();
        query.insert("nonesuch".to_owned(), QueryTerm::new("alpha"));
        assert!(matches!(
            fed.find(&query, 10),
            Err(QueryError::UnknownDimension { .. })
        ));
    }

    /// Like [`lexical_index_with`], plus a manifest recording the dimension's
    /// artifact fingerprint. `prompt` is folded into that fingerprint.
    fn lexical_index_with_manifest(text: &str, prompt: &str) -> TempMap {
        let tmp = lexical_index_with("src/lib.rs", text);
        let config_text = format!(
            "version = 1\n[dimensions.lexical]\ndescription = \"x\"\n\
             classifier = {{ impl = \"structural\", prompts = {{ \"0\" = \"{prompt}\" }} }}\n"
        );
        std::fs::write(tmp.0.join(".map/config.toml"), &config_text).unwrap();
        let config = Config::parse(&config_text).unwrap();
        let manifest = map_format::Manifest::new(&config, "test", 0).unwrap();
        std::fs::write(
            tmp.0.join(".map/manifest.json"),
            manifest.to_bytes().unwrap(),
        )
        .unwrap();
        tmp
    }

    #[test]
    fn one_dimension_name_meaning_two_things_is_refused() {
        // Same name, different prompt: the descriptors describe their material
        // in different terms, so averaging their scores is arithmetic without
        // meaning. A query names a dimension by its bare name and cannot
        // distinguish them, so the federation refuses rather than fusing.
        let a = lexical_index_with_manifest("refresh token", "describe behaviour");
        let b = lexical_index_with_manifest("refresh token", "describe data shapes");

        assert!(matches!(
            Federation::open([("a", &a.0), ("b", &b.0)]),
            Err(QueryError::IncompatibleDimension { .. })
        ));
    }

    #[test]
    fn the_same_dimension_built_the_same_way_federates() {
        let a = lexical_index_with_manifest("refresh token", "describe behaviour");
        let b = lexical_index_with_manifest("session expiry", "describe behaviour");
        let fed = Federation::open([("a", &a.0), ("b", &b.0)]).expect("identical builds federate");
        assert_eq!(fed.dimensions(), vec!["lexical".to_owned()]);
    }

    #[test]
    fn origins_must_be_unique() {
        let a = lexical_index_with("src/lib.rs", "alpha");
        let b = lexical_index_with("src/lib.rs", "beta");
        assert!(matches!(
            Federation::open([("same", &a.0), ("same", &b.0)]),
            Err(QueryError::DuplicateOrigin(_))
        ));
    }

    /// Counts how often it is asked to encode, so a fan-out can assert it was
    /// neither reconstructed nor re-invoked per index.
    struct CountingEmbedder {
        encodes: std::sync::atomic::AtomicUsize,
    }

    impl CountingEmbedder {
        fn new() -> Self {
            CountingEmbedder {
                encodes: std::sync::atomic::AtomicUsize::new(0),
            }
        }
        fn encodes(&self) -> usize {
            self.encodes.load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    impl map_core::Stage for CountingEmbedder {
        fn implementation(&self) -> &str {
            "fake"
        }
        fn config(&self) -> String {
            "fake:v1".to_owned()
        }
    }

    impl Embedder for CountingEmbedder {
        fn embed(
            &self,
            _batch: &map_core::EmbedBatch<'_>,
        ) -> map_core::Result<Vec<map_core::DimensionTensors>> {
            Ok(Vec::new())
        }
        fn encode_query(&self, _dimension: &str, _text: &str) -> map_core::Result<Tensor> {
            self.encodes
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(unit_tensor())
        }
    }

    fn unit_tensor() -> Tensor {
        let mut data = Vec::new();
        for c in [1.0f32, 0.0f32] {
            data.extend_from_slice(&c.to_le_bytes());
        }
        Tensor {
            dtype: map_format::DType::F32,
            shape: vec![2],
            data,
        }
    }

    /// A dense-only index whose pack holds one record. Its embedder is named
    /// `fake`, so only an injected implementation can serve it.
    fn dense_index() -> TempMap {
        let tmp = TempMap::new();
        std::fs::write(
            tmp.0.join(".map/config.toml"),
            "version = 1\n[dimensions.semantic]\ndescription = \"x\"\n\
             embedder = { impl = \"fake\" }\n",
        )
        .unwrap();

        let record = Record {
            descriptor: None,
            tensor: Some(unit_tensor()),
            meta: RecordMeta {
                kind: RecordKind::Segment,
                dimension: "semantic".into(),
                level: 0,
                children: vec![],
            },
        };
        let mut builder = map_stages::DensePackBuilder::new();
        builder.push(&IndexedRecord {
            id: 0,
            resource: "src/a.rs",
            segment: Segment { start: 0, end: 12 },
            record: &record,
        });
        std::fs::write(tmp.0.join(".map/cache/semantic.pack"), builder.finish()).unwrap();
        tmp
    }

    #[test]
    fn three_indexes_share_one_embedder_and_encode_once_each_query() {
        let stages = Stages::new();
        let counter = Arc::new(CountingEmbedder::new());
        stages.insert_embedder("fake", counter.clone() as Arc<dyn Embedder>);

        let (a, b, c) = (dense_index(), dense_index(), dense_index());
        let indexes: Vec<Index> = [&a, &b, &c]
            .iter()
            .map(|t| Index::open_with(&t.0, &stages).unwrap())
            .collect();

        // One instance, held by: this test, the registry, and each index. A
        // per-index resolve would have built three separate embedders.
        assert_eq!(
            Arc::strong_count(&counter),
            5,
            "every index must borrow the same embedder"
        );
        assert_eq!(
            counter.encodes(),
            0,
            "opening an index must not encode anything"
        );

        // Each index still encodes for itself today; the federation is what
        // will hoist this to once per query.
        let mut query = Query::new();
        query.insert("semantic".to_owned(), QueryTerm::new("alpha"));
        for index in &indexes {
            index.find(&query, 10).unwrap();
        }
        assert_eq!(counter.encodes(), indexes.len());
    }

    #[test]
    fn zoom_selects_by_level_and_defaults_to_segments() {
        let (_tmp, index) = index_with_a_segment_and_a_cluster();

        let segments = index
            .find_at(&q("alpha"), 10, LevelFilter::Segments)
            .unwrap();
        assert_eq!(segments.len(), 1, "one segment matches alpha");
        assert_eq!(segments[0].level, 0);
        assert_eq!(segments[0].resource, "src/a.rs");

        let clusters = index
            .find_at(&q("alpha"), 10, LevelFilter::Clusters)
            .unwrap();
        assert_eq!(clusters.len(), 1, "one cluster matches alpha");
        assert_eq!(clusters[0].level, 1);

        let all = index.find_at(&q("alpha"), 10, LevelFilter::All).unwrap();
        assert_eq!(all.len(), 2, "both are returned together");

        assert_eq!(index.find(&q("alpha"), 10).unwrap().len(), 1);
    }

    /// Two dimensions: `fabric` searches levels 0 and 1 and holds a record at
    /// each, `flat` searches only level 0. That asymmetry is the whole point —
    /// a query's level scope has to mean something different to each.
    fn index_with_one_stratified_and_one_flat_dimension() -> (TempMap, Index) {
        let tmp = TempMap::new();
        std::fs::write(
            tmp.0.join(".map/config.toml"),
            "version = 1\n\
             [dimensions.fabric]\ndescription = \"x\"\nlevels = [0, 1]\n\
             classifier = { impl = \"structural\" }\n\
             [dimensions.flat]\ndescription = \"y\"\nlevels = [0]\n\
             classifier = { impl = \"structural\" }\n",
        )
        .unwrap();

        let mut fabric = PackBuilder::new();
        fabric.push(&IndexedRecord {
            id: 0,
            resource: "src/a.rs",
            segment: Segment { start: 0, end: 12 },
            record: &leveled("alpha beta", 0),
        });
        fabric.push(&IndexedRecord {
            id: 1,
            resource: "clusterkeyhex",
            segment: Segment { start: 0, end: 0 },
            record: &leveled("alpha gamma", 1),
        });
        std::fs::write(tmp.0.join(".map/cache/fabric.pack"), fabric.finish()).unwrap();

        let mut flat = PackBuilder::new();
        flat.push(&IndexedRecord {
            id: 0,
            resource: "src/a.rs",
            segment: Segment { start: 0, end: 12 },
            record: &leveled("alpha delta", 0),
        });
        std::fs::write(tmp.0.join(".map/cache/flat.pack"), flat.finish()).unwrap();

        let index = Index::open(&tmp.0).unwrap();
        (tmp, index)
    }

    #[test]
    fn a_dimension_scores_only_the_levels_it_declares() {
        // The candidate set for a dimension is its configured levels
        // intersected with the query's scope. `flat` declares level 0 only, so
        // at level 1 its intersection is empty: it scores nothing, and — the
        // part that matters — it is not in the denominator either, so the
        // cluster keeps its full score instead of being divided by a dimension
        // that was never in the running.
        let (_tmp, index) = index_with_one_stratified_and_one_flat_dimension();

        let mut query = Query::new();
        query.insert("fabric".to_owned(), QueryTerm::new("alpha"));
        query.insert("flat".to_owned(), QueryTerm::new("alpha"));

        let scoped = index
            .find_at(&query, 10, LevelFilter::Only([1].into()))
            .unwrap();
        assert_eq!(scoped.len(), 1, "only the level-1 cluster is in scope");
        assert_eq!(scoped[0].level, 1);
        assert_eq!(
            scoped[0].per_dimension.keys().collect::<Vec<_>>(),
            vec!["fabric"],
            "flat cannot reach level 1"
        );

        // The same cluster scored on its own: identical, because a dimension
        // that could not have scored it must not divide it.
        let mut alone = Query::new();
        alone.insert("fabric".to_owned(), QueryTerm::new("alpha"));
        let solo = index
            .find_at(&alone, 10, LevelFilter::Only([1].into()))
            .unwrap();
        assert_eq!(scoped[0].score, solo[0].score);
    }

    #[test]
    fn declared_levels_bound_what_a_dimension_searches() {
        // `fabric` holds a level-1 record, but a query scoped to level 0 must
        // not see it, and `flat` must still answer there.
        let (_tmp, index) = index_with_one_stratified_and_one_flat_dimension();

        let mut query = Query::new();
        query.insert("fabric".to_owned(), QueryTerm::new("alpha"));
        query.insert("flat".to_owned(), QueryTerm::new("alpha"));

        let hits = index
            .find_at(&query, 10, LevelFilter::Only([0].into()))
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].level, 0);
        assert_eq!(hits[0].resource, "src/a.rs");
        // Both dimensions reach level 0, so both are in play here.
        let scored: Vec<&String> = hits[0].per_dimension.keys().collect();
        assert_eq!(scored, vec!["fabric", "flat"]);
    }

    #[test]
    fn an_explicit_level_set_selects_exactly_those_heights() {
        let (_tmp, index) = index_with_a_segment_and_a_cluster();

        let zero = index
            .find_at(&q("alpha"), 10, LevelFilter::Only([0].into()))
            .unwrap();
        assert_eq!(zero.len(), 1);
        assert_eq!(zero[0].level, 0);

        let one = index
            .find_at(&q("alpha"), 10, LevelFilter::Only([1].into()))
            .unwrap();
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].level, 1);

        // Naming both heights is the same scope as `All` here, and naming a
        // height the fabric never reached returns nothing rather than erroring:
        // an empty level is a real answer, not a malformed request.
        let both = index
            .find_at(&q("alpha"), 10, LevelFilter::Only([0, 1].into()))
            .unwrap();
        assert_eq!(both.len(), 2);
        let absent = index
            .find_at(&q("alpha"), 10, LevelFilter::Only([7].into()))
            .unwrap();
        assert!(absent.is_empty());
    }

    #[test]
    fn equal_weights_match_the_unweighted_mean() {
        let per = BTreeMap::from([
            ("lexical".to_owned(), 0.8_f32),
            ("descriptive".to_owned(), 0.4),
        ]);
        let asked = vec!["lexical".to_owned(), "descriptive".to_owned()];

        let weighted = fuse_weighted_mean(&per, &asked, |_| 1.0);
        let plain = per.values().sum::<f32>() / asked.len() as f32;
        assert!(
            (weighted - plain).abs() < 1e-6,
            "weight 1.0 everywhere must reproduce the old mean exactly"
        );
    }

    #[test]
    fn raising_a_dimension_weight_reorders_two_candidates() {
        // A is strong in lexical, weak in descriptive; B is the mirror image.
        let a = BTreeMap::from([
            ("lexical".to_owned(), 0.9_f32),
            ("descriptive".to_owned(), 0.1),
        ]);
        let b = BTreeMap::from([
            ("lexical".to_owned(), 0.1_f32),
            ("descriptive".to_owned(), 0.9),
        ]);
        let asked = vec!["lexical".to_owned(), "descriptive".to_owned()];

        let ea = fuse_weighted_mean(&a, &asked, |_| 1.0);
        let eb = fuse_weighted_mean(&b, &asked, |_| 1.0);
        assert!(
            (ea - eb).abs() < 1e-6,
            "symmetric candidates tie at equal weight"
        );

        let favor_descriptive = |d: &str| if d == "descriptive" { 3.0 } else { 1.0 };
        assert!(
            fuse_weighted_mean(&b, &asked, favor_descriptive)
                > fuse_weighted_mean(&a, &asked, favor_descriptive),
            "a heavier descriptive weight must promote the descriptive-strong candidate"
        );
    }

    #[test]
    fn zero_weight_excludes_a_dimension_from_the_search() {
        let (_tmp, index) = index_with_a_segment_and_a_cluster();

        let mut query = Query::new();
        query.insert("lexical".to_owned(), QueryTerm::weighted("alpha", 0.0));
        let hits = index.find_at(&query, 10, LevelFilter::Segments).unwrap();
        assert!(hits.is_empty(), "a zero-weight dimension is never scored");
    }

    #[test]
    fn a_non_finite_denominator_fuses_to_zero_not_nan() {
        let per = BTreeMap::from([("lexical".to_owned(), 0.9_f32)]);
        let asked = vec!["lexical".to_owned()];
        assert_eq!(fuse_weighted_mean(&per, &asked, |_| f32::NAN), 0.0);
        assert_eq!(fuse_weighted_mean(&per, &asked, |_| f32::INFINITY), 0.0);
    }

    #[test]
    fn a_nan_weight_is_excluded_before_scoring() {
        let (_tmp, index) = index_with_a_segment_and_a_cluster();

        let mut query = Query::new();
        query.insert("lexical".to_owned(), QueryTerm::weighted("alpha", f32::NAN));
        let hits = index.find_at(&query, 10, LevelFilter::Segments).unwrap();
        assert!(
            hits.is_empty(),
            "a NaN weight is excluded, not scored to NaN"
        );
    }

    #[test]
    fn a_weight_on_a_lone_dimension_cancels_out() {
        let (_tmp, index) = index_with_a_segment_and_a_cluster();

        // With one dimension the weighted mean is that dimension's own score,
        // whatever the weight: numerator and denominator both carry it.
        let mut query = Query::new();
        query.insert("lexical".to_owned(), QueryTerm::weighted("alpha", 5.0));
        let hits = index.find_at(&query, 10, LevelFilter::Segments).unwrap();
        assert_eq!(hits.len(), 1);
        assert!((hits[0].score - hits[0].per_dimension["lexical"]).abs() < 1e-6);
    }
}

/// Walk a cluster's children down to level-0 ids.
///
/// A child is a cluster if the cluster store holds it, and a leaf otherwise
/// — leaves have no object of their own, only a derived id.
///
/// `visited` is what makes a committed index safe to walk: cluster objects are
/// attacker-writable, so a child list may name an ancestor or itself. A key
/// seen before contributes nothing a second time, which bounds the walk by the
/// number of objects rather than by the shape of the graph.
fn leaf_ids_of(
    key: map_format::ObjectKey,
    read: &dyn Fn(map_format::ObjectKey) -> Option<map_format::Record>,
    visited: &mut BTreeSet<map_format::ObjectKey>,
) -> Option<BTreeSet<map_format::ObjectKey>> {
    if !visited.insert(key) {
        return Some(BTreeSet::new());
    }
    let record = read(key)?;
    let mut out = BTreeSet::new();
    for child in &record.meta.children {
        match leaf_ids_of(*child, read, visited) {
            Some(deeper) => out.extend(deeper),
            None => {
                out.insert(*child);
            }
        }
    }
    Some(out)
}

/// Clamp a stored byte span to something `text[start..end]` accepts.
///
/// Spans come from a committed segments object, so they are untrusted: an end
/// before its start, or an offset inside a multi-byte character, would panic
/// the slice. Each end is pulled back to the nearest char boundary at or
/// below it, and an inverted span collapses to empty rather than aborting the
/// query.
fn clamp_span(text: &str, start: usize, end: usize) -> (usize, usize) {
    let snap = |mut i: usize| {
        i = i.min(text.len());
        while !text.is_char_boundary(i) {
            i -= 1;
        }
        i
    };
    let start = snap(start);
    let end = snap(end).max(start);
    (start, end)
}

#[cfg(test)]
mod hostile_index_tests {
    use super::*;

    #[test]
    fn a_span_from_a_hostile_segments_object_cannot_panic_the_slice() {
        let text = "héllo wörld";
        // Inside the two-byte `é`, inverted, and past the end.
        for (start, end) in [(2, 1), (1, 2), (3, 2), (100, 200), (5, 3)] {
            let (s, e) = clamp_span(text, start, end);
            assert!(s <= e, "{start}..{end} became {s}..{e}");
            let _ = &text[s..e];
        }
        assert_eq!(clamp_span(text, 0, 5), (0, 5));
        assert_eq!(
            clamp_span(text, 7, 2),
            (7, 7),
            "an inverted span collapses to empty"
        );
    }

    #[test]
    fn a_cyclic_cluster_graph_terminates_and_yields_its_leaves() {
        use map_format::{Digest, ObjectKey, Record, RecordKind, RecordMeta};

        let key = |n: u8| ObjectKey(Digest::from_hex(&format!("{n:02x}").repeat(32)).unwrap());
        let cluster = |children: Vec<ObjectKey>| Record {
            descriptor: Some("label".into()),
            tensor: None,
            meta: RecordMeta {
                kind: RecordKind::Cluster,
                dimension: "d".into(),
                level: 1,
                children,
            },
        };
        // 1 -> {2, 3, leaf 9}; 2 -> {1, leaf 8}; 3 -> {3}: a back edge and a self loop.
        let read = move |k: ObjectKey| -> Option<Record> {
            if k == key(1) {
                Some(cluster(vec![key(2), key(3), key(9)]))
            } else if k == key(2) {
                Some(cluster(vec![key(1), key(8)]))
            } else if k == key(3) {
                Some(cluster(vec![key(3)]))
            } else {
                None
            }
        };
        let leaves = leaf_ids_of(key(1), &read, &mut BTreeSet::new()).unwrap();
        assert_eq!(leaves, BTreeSet::from([key(8), key(9)]));
    }
}
