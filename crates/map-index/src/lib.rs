//! The offline indexer: discover through fabricate.
//!
//! Writes the result into a `.map` directory. The orchestration rule that
//! matters is in [`map_core::group_by_implementation`]: dimensions are grouped
//! by the implementation they selected, so N dimensions sharing a classifier
//! cost one call per segment batch rather than N.
//!
//! Classify (`structural`, plus `llm` under its feature), embed
//! (`distilled` under its feature), and fabricate (needing both) are wired. The
//! lexical dimension needs neither embed nor an LLM, so a default build indexes
//! with nothing configured.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use map_core::{
    group_by_implementation, Classifier, ClassifyBatch, DescriptorPayload, DimensionRecords,
    Discoverer, EmbedBatch, Embedder, Preprocessor, Segmenter, SegmentsPayload, Stage,
    TensorPayload,
};
use map_format::manifest::ResourceRoot;
use map_format::{Config, Fingerprint, Manifest, ObjectKey, ObjectStore, Tier};
use map_stages::{
    cache, ContentClassifier, FsDiscoverer, StructuralClassifier, TextPreprocessor, WindowSegmenter,
};

pub mod degraded;
pub mod fabricate;
pub mod gc;
#[cfg(test)]
mod ledger;
pub mod staleness;

pub use degraded::{Degraded, UnresolvedStage};
pub use gc::{GcReport, GitScope};
pub use staleness::{Snapshot, StaleReport};

/// Re-exported so a caller can match on a classifier's raw answer without
/// taking its own `serde_json` dependency.
#[cfg(feature = "llm")]
pub use serde_json::Value as JsonValue;

/// Errors from an indexing run.
#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    #[error("no .map directory found at or above {0} — run `map init` first")]
    NotInitialized(PathBuf),
    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error(transparent)]
    Stage(#[from] map_core::Error),
    #[error(transparent)]
    Format(#[from] map_format::Error),
    #[error(transparent)]
    Cache(#[from] cache::CacheError),

    /// Declined to delete objects because what still reaches them is unknown.
    #[error("cannot safely collect: {0}")]
    GcUnsafe(String),

    /// The manifest is the only record of what each object's bytes should hash
    /// to, so a run that cannot read it has no integrity baseline and would
    /// re-record whatever is on disk as correct (spec §8). Refusing matches
    /// what collection already does with an unparseable manifest (spec §9.4).
    #[error(
        "the manifest at {path} could not be read ({detail}), so this run has nothing to check \
         the stored objects against. Restore it from version control, or delete it to rebuild \
         the index from scratch."
    )]
    UnreadableManifest { path: PathBuf, detail: String },

    /// A stored object's bytes no longer hash to what the manifest recorded.
    ///
    /// Reusing it would launder the tampered bytes into the next manifest under
    /// a freshly computed hash, which is precisely the prompt-injection vector
    /// spec §8 names: a descriptor is model context with a home in the repo.
    #[error(
        "object {key} was altered after it was recorded: the manifest expects {expected}, the \
         bytes on disk are {actual}. Restore it from version control, or delete it so the stage \
         re-runs — which for an `llm` dimension re-bills the request."
    )]
    Tampered {
        key: String,
        expected: String,
        actual: String,
    },

    /// Two `llm` dimensions disagree about what to ask.
    #[error(
        "dimensions {first:?} and {second:?} both use the `llm` classifier but configure \
         different `{setting}`. One request serves the whole group, so every `llm` dimension \
         must agree on `prompts`, `prompt` and `facets` — or be a single dimension."
    )]
    DivergentLlmGroup {
        first: String,
        second: String,
        setting: &'static str,
    },

    /// A dimension reached fabrication in a shape `Config::validate` refuses.
    #[error("dimension {dimension:?} cannot be fabricated: {detail} — config validation should have refused this at load")]
    Unfabricatable { dimension: String, detail: String },

    /// Refused rather than built a partial index. The dimensions listed would
    /// have been silently omitted, or — worse for a semantic dimension whose
    /// classifier is missing — built from raw text under a description
    /// promising otherwise.
    #[error(
        "this build cannot run {} configured stage(s):\n{}\n\
         Nothing was written. Pass --degraded=allow to build without them.",
        .0.len(),
        degraded::describe(.0)
    )]
    Degraded(Vec<UnresolvedStage>),
}

/// What an indexing run produced.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IndexStats {
    pub discovered: usize,
    pub indexed: usize,
    /// Skipped as binary or undecodable.
    pub skipped: usize,
    pub segments: usize,
    pub objects_written: usize,
    /// Already present with identical bytes.
    pub objects_reused: usize,
    pub packs_written: usize,
    /// The number to watch: with N dimensions sharing one implementation this
    /// stays at one call per resource per *group*, never per dimension.
    pub classifier_calls: usize,
    pub embed_calls: usize,
    pub clusters_written: usize,
    /// Classifier invocations the fabricator made — one per level that had any
    /// group needing a label, not one per cluster. Reused clusters cost none.
    pub label_calls: usize,
    /// Groups the classifier returned no label for.
    ///
    /// The fabricate-side analogue of `undescribed_segments`. Their members are
    /// left ungrouped so they carry forward as orphans rather than vanishing
    /// into a cluster that was never written.
    pub unlabeled_clusters: usize,
    /// Segments an LLM classifier could not get usable structured output for.
    ///
    /// Non-zero is a configuration problem, not a content one: an endpoint that
    /// accepts `response_format` and ignores it yields prose the schema rejects,
    /// and the index would otherwise be built with those segments silently
    /// blank.
    pub undescribed_segments: usize,
    /// Configured stages this build could not run, which the caller accepted.
    ///
    /// Non-empty means the index is missing dimensions its `config.toml`
    /// describes. Recorded so an allowed run still reports what it left out —
    /// consent to build without a stage is not a reason to stop mentioning it.
    pub degraded: Vec<UnresolvedStage>,
}

/// Bring the index up to date, returning `None` if nothing had drifted.
///
/// Refreshing costs an incremental index, O(changed) — milliseconds of local
/// work with only the structural classifier wired.
///
/// **An LLM-backed classifier changes the stakes:** re-classifying a changed
/// file spends money. So this is never automatic — calling it *is* the consent,
/// which on the CLI means an explicit `-u`. It repairs every dimension, cheap
/// and expensive alike, so it can re-bill the LLM for an edit unrelated to that
/// dimension.
pub fn refresh(root: &Path) -> Result<Option<IndexStats>, IndexError> {
    refresh_with_progress(root, Degraded::Abort, |_| {})
}

/// [`refresh`], reporting progress while it works.
pub fn refresh_with_progress(
    root: &Path,
    policy: Degraded<'_>,
    mut progress: impl FnMut(Progress<'_>),
) -> Result<Option<IndexStats>, IndexError> {
    // Short-circuit on stat before doing anything expensive. A full pass must
    // read and segment every resource just to learn nothing changed — ~220 ms on
    // ripgrep, and O(corpus). Walking and stat-ing is ~22 ms and answers the
    // same question for the overwhelmingly common case.
    if matches!(stale(root)?, Some(report) if !report.is_stale()) {
        return Ok(None);
    }

    Ok(Some(run_inner(root, policy, &mut progress)?))
}

/// How far the tree has drifted from the index, without reading any content.
///
/// `None` means there is no snapshot to compare against — a fresh clone, or a
/// cleared cache — so drift is unknown rather than zero.
///
/// Costs a directory walk and one `stat` per resource (~22 ms on ripgrep).
/// Callers that will not surface the answer should not ask: on the agent path
/// the notice is suppressed, so skipping this keeps a query at ~10 ms.
pub fn stale(root: &Path) -> Result<Option<StaleReport>, IndexError> {
    let map_root = map_core::find_map_root(root)
        .ok_or_else(|| IndexError::NotInitialized(root.to_path_buf()))?;

    let map_dir = map_root.join(".map");
    let Some(snapshot) = Snapshot::load(&map_dir.join("cache/stat.json")) else {
        return Ok(None);
    };
    let resources = FsDiscoverer::new().discover(&map_root)?;

    // Read rather than parsed into an error: this runs on the query path to
    // decide whether to print a notice, and a config that will not parse is a
    // problem for `map index` to report. Unreadable reads as "cannot confirm",
    // which the comparison already treats as changed.
    let config_fingerprint = std::fs::read_to_string(map_dir.join("config.toml"))
        .ok()
        .and_then(|text| Config::parse(&text).ok())
        .and_then(|config| config.fingerprint().ok())
        .map(|fp| fp.to_string());
    let manifest_hash = std::fs::read(map_dir.join("manifest.json"))
        .ok()
        .map(|bytes| map_format::ContentHash::of(&bytes).to_string());

    Ok(Some(snapshot.compare(
        &map_root,
        &resources,
        config_fingerprint.as_deref(),
        manifest_hash.as_deref(),
    )))
}

/// How far an indexing run has got.
///
/// Reported so a caller can show why a command is taking time. That matters
/// most for the expensive case: with a dense or LLM-backed dimension an update
/// is slow *and costs money*, and whoever asked for it should be able to watch
/// it happen and interrupt.
#[derive(Clone, Copy, Debug)]
pub struct Progress<'a> {
    pub done: usize,
    pub total: usize,
    pub classifier_calls: usize,
    pub resource: &'a str,
}

impl Progress<'_> {
    pub fn fraction(&self) -> f32 {
        if self.total == 0 {
            1.0
        } else {
            self.done as f32 / self.total as f32
        }
    }
}

/// Index the resources under `root` into its `.map` directory.
///
/// Refuses to build if any configured stage cannot be resolved; use
/// [`run_with_progress`] with [`Degraded::Allow`] to accept a partial index.
/// The strict default is deliberate — a silently-omitted dimension is
/// indistinguishable from a working one at query time.
pub fn run(root: &Path) -> Result<IndexStats, IndexError> {
    run_with_progress(root, Degraded::Abort, |_| {})
}

/// Index, reporting progress as each resource is processed.
pub fn run_with_progress(
    root: &Path,
    policy: Degraded<'_>,
    mut progress: impl FnMut(Progress<'_>),
) -> Result<IndexStats, IndexError> {
    run_inner(root, policy, &mut progress)
}

fn run_inner(
    root: &Path,
    policy: Degraded<'_>,
    progress: &mut dyn FnMut(Progress<'_>),
) -> Result<IndexStats, IndexError> {
    let root = map_core::find_map_root(root)
        .ok_or_else(|| IndexError::NotInitialized(root.to_path_buf()))?;
    let map_dir = root.join(".map");

    let config_path = map_dir.join("config.toml");
    let config_text = std::fs::read_to_string(&config_path).map_err(|e| IndexError::Io {
        path: config_path.clone(),
        source: e,
    })?;
    let config = Config::parse(&config_text)?;

    // Loaded before anything is written: it is the only record of what each
    // stored object's bytes should hash to, and every reuse below is checked
    // against it (spec §8).
    let prior = prior_manifest(&map_dir)?;

    let shared = ObjectStore::open(map_dir.join("index/shared/objects"));
    let descriptors = ObjectStore::open(map_dir.join("index/desc/objects"));
    let tensor_store = ObjectStore::open(map_dir.join("index/tensor/objects"));

    let discoverer = FsDiscoverer::new();
    let preprocessor = TextPreprocessor;
    let segmenter = segmenter_for(&config);
    // Structural descriptors are Tier A (bit-reproducible); an LLM's are Tier C
    // (authored, never expected to reproduce). A group whose implementation is
    // not resolved here is skipped, which is what lets a semantic dimension be
    // configured before its endpoint is live.
    let structural = StructuralClassifier;
    // Tier A alongside the structural one: a pass-through of the content is as
    // reproducible as tokenizing it.
    let passthrough = ContentClassifier;
    let declaration = map_stages::declaration::DeclarationClassifier;
    // `mut` is used only when a llm classifier is pushed below.
    #[allow(unused_mut)]
    let mut classifiers: Vec<(&dyn Classifier, Tier)> = vec![
        (&structural, Tier::A),
        (&passthrough, Tier::A),
        (&declaration, Tier::A),
    ];

    // Resolved once: both the segment classifier and the cluster labeler ride
    // the same adapter, so the endpoint and key live in one place and never
    // touch the committed config.
    #[cfg(feature = "llm")]
    let llm_adapter = resolve_llm_adapter(&config)?;
    #[cfg(feature = "llm")]
    let llm = match llm_adapter.as_ref() {
        Some(adapter) => build_llm_classifier(&config, adapter.clone())?,
        None => None,
    };
    #[cfg(feature = "llm")]
    if let Some(llm) = llm.as_ref() {
        classifiers.push((llm, Tier::C));
    }

    let embedder = resolve_embedder(&config);

    // Decide about missing stages here: every implementation has now resolved,
    // and not one resource has been read. Either this run is going to produce
    // what the config describes or the operator has said they are fine with
    // less — before any work, and before any object is written.
    // Vectors to cluster are the only hard requirement. Whether the dimension's
    // classifier resolves is a separate question the classifier check already
    // answers, and reporting it twice would send the operator after the
    // fabricator when the endpoint is what is missing.
    #[cfg(feature = "distilled")]
    let can_fabricate = embedder.is_some();
    #[cfg(not(feature = "distilled"))]
    let can_fabricate = false;

    let resolved: Vec<&str> = classifiers
        .iter()
        .map(|(c, _)| c.implementation())
        .collect();
    let unresolved = degraded::unresolved(
        &config,
        &resolved,
        embedder.as_ref().map(|e| e.implementation()),
        can_fabricate,
    );
    if !unresolved.is_empty() {
        let consented = match policy {
            Degraded::Allow => true,
            Degraded::Abort => false,
            Degraded::Ask(ask) => ask(&unresolved),
        };
        if !consented {
            return Err(IndexError::Degraded(unresolved));
        }
    }

    // The whole cost model: one call per group, not per dimension.
    let names: Vec<&str> = config.active().map(|(n, _)| n.as_str()).collect();
    let groups = group_by_implementation(names.iter().copied(), |name| {
        config
            .dimensions
            .get(name)
            .and_then(|d| d.classifier.as_ref())
            .map(|c| c.implementation.clone())
    });
    let embed_groups = group_by_implementation(names.iter().copied(), |name| {
        config
            .dimensions
            .get(name)
            .and_then(|d| d.embedder.as_ref())
            .map(|e| e.implementation.clone())
    });

    // Dimensions with both a classifier and an embedder: their tensor is the
    // embedding of the descriptor, not of the raw segment. Only these need the
    // classified records, so a lexical-only or pure-embedding dimension pays
    // nothing and the incremental re-index stays as cheap as before.
    let semantic_dims: BTreeSet<String> = config
        .active()
        .filter(|(_, d)| d.embeds_descriptor())
        .map(|(n, _)| n.clone())
        .collect();

    let segmenter_fp = Fingerprint::of(segmenter.config().as_bytes());

    let mut manifest = Manifest::new(&config, env!("CARGO_PKG_VERSION"), now_unix())?;
    let mut stats = IndexStats {
        degraded: unresolved,
        ..IndexStats::default()
    };

    let resources = discoverer.discover(&root)?;
    stats.discovered = resources.len();

    // Stat'd here, before a single resource is read, so the snapshot describes
    // the tree this run is about to index. Taking it at the end instead records
    // an edit that landed *during* the run as already-indexed: the content in
    // the index is what was read before the edit, and the next staleness check
    // compares against the post-edit stat and sees nothing to do. A long
    // LLM-backed run is exactly where that window is wide.
    let mut snapshot = Snapshot::take(&root, &resources);

    for (position, resource) in resources.iter().enumerate() {
        progress(Progress {
            done: position,
            total: resources.len(),
            classifier_calls: stats.classifier_calls,
            resource: &resource.key,
        });

        // Reading belongs to the driver, not the preprocessor: the driver is the
        // only layer that knows a resource's kind, so a non-file resource reuses
        // the same preprocessor unchanged.
        map_stages::discover::check_resource_key(&resource.key)?;
        let path = root.join(&resource.key);
        let bytes = std::fs::read(&path).map_err(|e| IndexError::Io { path, source: e })?;

        let Some(content) = preprocessor.preprocess(resource, &bytes)? else {
            stats.skipped += 1;
            continue;
        };

        let segments = segmenter.segment(&content)?;
        if segments.is_empty() {
            stats.skipped += 1;
            continue;
        }
        stats.segments += segments.len();

        // Segmentation is shared across dimensions, so it is keyed only by
        // content and segmenter config.
        let segments_key = ObjectKey::derive(&[content.text.as_bytes()], segmenter_fp);
        let segments_bytes = map_format::codec::canonical_json(&SegmentsPayload {
            segments: segments.clone(),
        })?;
        write_object(
            &shared,
            &mut manifest,
            &mut stats,
            segments_key,
            &segments_bytes,
            Tier::A,
        )?;

        let mut root_entry = ResourceRoot {
            segments: segments_key,
            descriptors: Default::default(),
            tensors: Default::default(),
        };

        // Empty and untouched unless a classifier+embedder dimension exists, so
        // a lexical-only incremental re-index is unchanged.
        let mut classified: Vec<DimensionRecords> = if semantic_dims.is_empty() {
            Vec::new()
        } else {
            vec![DimensionRecords::new(); segments.len()]
        };

        for group in &groups {
            let Some(&(classifier, tier)) = classifiers
                .iter()
                .find(|(c, _)| c.implementation() == group.implementation)
            else {
                continue;
            };

            // Derive the key from the inputs *before* running the stage: content
            // plus the classifier's config plus which dimensions this call
            // covered, so editing one group's settings invalidates exactly that
            // group.
            //
            // The segmenter is in because the payload is one record *per
            // segment*: re-segmenting the same content is a different question
            // asked of the classifier, and without this the answer to the old
            // one is reused under a key that still matches. That produced an
            // index whose spans said 8 segments and whose descriptors held 4.
            let fingerprint = Fingerprint::of(
                descriptor_fingerprint_input(&classifier.config(), &group.dimensions, segmenter_fp)
                    .as_bytes(),
            );
            let key = ObjectKey::derive(&[content.text.as_bytes()], fingerprint);

            // Skipping here rather than deduplicating on write is the difference
            // between an incremental re-index costing nothing and costing a full
            // pass — and for an LLM implementation, "we threw the answer away"
            // is a bill.
            let group_is_semantic = group.dimensions.iter().any(|d| semantic_dims.contains(d));

            // Uniform across the group — `Config::validate` refuses a group
            // that disagrees, because one object cannot be both stored and not.
            let persists = group
                .dimensions
                .first()
                .and_then(|d| config.dimensions.get(d))
                .and_then(|d| d.classifier.as_ref())
                .is_none_or(|c| c.persist_output);

            if persists {
                let recorded = prior.as_ref().and_then(|m| m.object(key));
                if let Some(entry) = descriptors.stat(key, tier, recorded).map_err(as_tampered)? {
                    manifest.insert_object(key, entry)?;
                    stats.objects_reused += 1;
                    root_entry
                        .descriptors
                        .insert(group.implementation.clone(), key);
                    // A reused descriptor still has to feed the embedder for a
                    // semantic dimension — but only then is it worth reading
                    // back.
                    if group_is_semantic {
                        let bytes = descriptors.get(key)?;
                        let payload: DescriptorPayload = serde_json::from_slice(&bytes)
                            .map_err(|e| IndexError::Format(map_format::Error::from(e)))?;
                        merge_classified(
                            &mut classified,
                            &payload.per_segment,
                            &group.dimensions,
                            &semantic_dims,
                        );
                    }
                    continue;
                }
            }

            // One item per segment, each holding that segment's own text. The
            // fabricator builds items from a group's descriptors instead, which
            // is the only difference between classifying a segment and
            // classifying a cluster.
            let items: Vec<Vec<&str>> = segments.iter().map(|s| vec![s.slice(&content)]).collect();
            let per_segment = classifier.classify(&ClassifyBatch {
                items: &items,
                dimensions: &group.dimensions,
                level: 0,
            })?;
            stats.classifier_calls += 1;

            if group_is_semantic {
                merge_classified(
                    &mut classified,
                    &per_segment,
                    &group.dimensions,
                    &semantic_dims,
                );
            }

            // A stage that persists nothing has already done its whole job: the
            // records fed the embedder above and are worth no bytes on disk.
            // Nothing references the key, so the manifest never learns of it.
            if !persists {
                continue;
            }

            let bytes = map_format::codec::canonical_json(&DescriptorPayload {
                dimensions: group.dimensions.clone(),
                per_segment,
            })?;

            write_object(&descriptors, &mut manifest, &mut stats, key, &bytes, tier)?;
            root_entry
                .descriptors
                .insert(group.implementation.clone(), key);
        }

        // The dense path mirrors the classifier path above. Runs only when an
        // embedder resolved; otherwise the dense dimension is skipped and the
        // rest of the index still builds.
        if let Some(embedder) = embedder.as_ref() {
            for group in &embed_groups {
                if group.implementation != embedder.implementation() {
                    continue;
                }
                // A tensor of a descriptor is a function of that descriptor, so
                // whatever produced it belongs in the key. Without this, editing
                // a prompt rewrites every descriptor — its own key folds in the
                // classifier config — while the tensor key stays put, so the
                // stored embedding of the *previous* prose is reused. The
                // dimension then ranks new labels by old vectors, and nothing
                // downstream can tell.
                let descriptor_identity = descriptor_identity(&config, &classifiers, group);
                let fingerprint = Fingerprint::of(
                    tensor_fingerprint_input(
                        &embedder.config(),
                        &group.dimensions,
                        &descriptor_identity,
                        segmenter_fp,
                    )
                    .as_bytes(),
                );
                let key = ObjectKey::derive(&[content.text.as_bytes()], fingerprint);

                // Embedding is the expensive stage for a dense dimension, so
                // reusing stored tensors is what makes re-indexing incremental.
                let recorded = prior.as_ref().and_then(|m| m.object(key));
                if let Some(entry) = tensor_store
                    .stat(key, Tier::B, recorded)
                    .map_err(as_tampered)?
                {
                    manifest.insert_object(key, entry)?;
                    stats.objects_reused += 1;
                    root_entry.tensors.insert(group.implementation.clone(), key);
                    continue;
                }

                let per_segment = embedder.embed(&EmbedBatch {
                    content: &content,
                    segments: &segments,
                    dimensions: &group.dimensions,
                    classified: &classified,
                })?;
                stats.embed_calls += 1;

                // Tier B: fp inference is producer-authoritative, not
                // cross-machine bit-identical like the structural descriptors.
                //
                // Raw frames rather than canonical JSON. JSON stored each float
                // byte as decimal digits, which git largely compressed away but
                // which cost 159 ms to parse on a fresh clone against 3.4 ms
                // here (spec §9.2).
                let bytes = map_core::encode_tensor_payload(&TensorPayload {
                    dimensions: group.dimensions.clone(),
                    per_segment,
                })?;
                write_object(
                    &tensor_store,
                    &mut manifest,
                    &mut stats,
                    key,
                    &bytes,
                    Tier::B,
                )?;
                root_entry.tensors.insert(group.implementation.clone(), key);
            }
        }

        manifest.roots.insert(resource.key.clone(), root_entry);
        stats.indexed += 1;
    }

    // Read before fabrication, which labels through a different stage: this
    // counts only segments the classifier gave up on.
    #[cfg(feature = "llm")]
    if let Some(llm) = llm.as_ref() {
        stats.undescribed_segments = llm.undescribed();
    }

    // Needs vectors to cluster, so it runs whenever an embedder resolved. The
    // classifier that labels is the dimension's own, which may be `structural`
    // or `content` and need no endpoint at all — so an LLM is no longer a
    // precondition for having a tree. A dimension whose classifier does not
    // resolve is skipped here and reported by the degraded check as a
    // classifier problem, which is the true cause.
    #[cfg(feature = "distilled")]
    if let Some(embedder) = embedder.as_ref() {
        run_fabrication(
            &config,
            &shared,
            &descriptors,
            &tensor_store,
            &map_dir,
            #[cfg(feature = "llm")]
            llm_adapter.clone(),
            embedder.as_ref(),
            prior.as_ref(),
            &mut manifest,
            &mut stats,
        )?;
    }

    let manifest_bytes = manifest.to_bytes()?;
    write_atomic(&map_dir.join("manifest.json"), &manifest_bytes)?;

    // Packs are derived: gitignored, rebuildable, and written after the manifest
    // so a crash leaves a stale cache rather than one that claims to describe an
    // index that was never written.
    let cache_dir = map_dir.join("cache");
    std::fs::create_dir_all(&cache_dir).map_err(|e| IndexError::Io {
        path: cache_dir.clone(),
        source: e,
    })?;
    // So the next refresh can decide there is nothing to do without reading a
    // single file. The identity is recorded alongside the stats: an edited
    // config or a manifest swapped in by a pull changes what the index should
    // hold without touching a single resource's mtime.
    let stat_path = cache_dir.join("stat.json");
    if stats.degraded.is_empty() {
        snapshot.config_fingerprint = Some(config.fingerprint()?.to_string());
        snapshot.manifest_hash = Some(map_format::ContentHash::of(&manifest_bytes).to_string());
        write_atomic(&stat_path, &snapshot.to_bytes())?;
    } else {
        // A degraded run built less than the config describes, so "nothing has
        // drifted" would be a lie: the missing dimensions are still missing
        // once the stage becomes available. Leaving no snapshot makes the next
        // `-u` do the work rather than short-circuit on an unchanged tree.
        match std::fs::remove_file(&stat_path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                return Err(IndexError::Io {
                    path: stat_path,
                    source: e,
                })
            }
        }
    }

    // The same function the query path uses on a cache miss, so an index-built
    // cache and a query-rebuilt one can never disagree. `verify = false`: these
    // objects were written microseconds ago and need no re-hashing.
    let wanted: Vec<String> = config.active().map(|(n, _)| n.clone()).collect();
    let dense: BTreeSet<String> = config
        .active()
        .filter(|(_, d)| d.emits_tensor())
        .map(|(n, _)| n.clone())
        .collect();
    // A later open rebuilds unless a pack's marker matches, catching both a
    // pulled manifest swapped under the gitignored cache and a pack
    // force-committed by a hostile clone, which cannot carry a valid marker
    // without the machine key. A marker binds the pack's own bytes, so there is
    // one per dimension and only the packs this run actually wrote get one.
    let mut markers = cache::PackMarkers::default();
    for (dimension, bytes) in cache::rebuild_packs(&map_dir, &wanted, &dense, false)? {
        let path = cache_dir.join(format!("{dimension}.pack"));
        cache::write_pack(&path, &bytes).map_err(|e| IndexError::Io { path, source: e })?;
        stats.packs_written += 1;
        if let Some(marker) = cache::pack_marker(&map_dir, &dimension, &bytes) {
            markers.insert(&dimension, marker);
        }
    }
    // Written once, after every pack has landed, so a crash leaves a marker
    // absent — read as "rebuild", never as a false all-clear.
    markers.write(&cache_dir).map_err(|e| IndexError::Io {
        path: cache_dir.join("packs.manifest"),
        source: e,
    })?;

    Ok(stats)
}

/// `None` when no dimension asks for a distilled embedder, when the `distilled`
/// feature is not compiled, or when the model is not present. Absence is
/// graceful: the run builds the lexical dimensions and omits the dense one.
#[cfg(feature = "distilled")]
fn resolve_embedder(config: &Config) -> Option<Box<dyn Embedder>> {
    let wants = config.active().any(|(_, d)| {
        d.embedder
            .as_ref()
            .is_some_and(|e| e.implementation == "distilled")
    });
    if !wants {
        return None;
    }
    let dir = model_cache_dir();
    match map_embed::DistilledEmbedder::from_dir(&dir, "potion-retrieval-32M") {
        Ok(e) => Some(Box::new(e)),
        Err(_) => None,
    }
}

#[cfg(not(feature = "distilled"))]
fn resolve_embedder(_config: &Config) -> Option<Box<dyn Embedder>> {
    None
}

/// Loads the cached connection or, on a terminal, prompts for it. Resolved here
/// rather than inside either stage because the same handle drives both the
/// segment classifier and the cluster labeler.
#[cfg(feature = "llm")]
fn resolve_llm_adapter(
    config: &Config,
) -> Result<Option<std::sync::Arc<map_llm::LlmAdapter>>, IndexError> {
    let wants = config.active().any(|(_, d)| {
        d.classifier
            .as_ref()
            .is_some_and(|c| c.implementation == "llm")
    });
    if !wants {
        return Ok(None);
    }
    let adapter =
        std::sync::Arc::new(
            map_llm::LlmAdapter::load_or_prompt().map_err(|e| IndexError::Io {
                path: std::path::PathBuf::from("<llm-adapter>"),
                source: std::io::Error::other(e.to_string()),
            })?,
        );
    Ok(Some(adapter))
}

/// Build the configured segmenter.
///
/// `Config::validate` has already refused any `impl` this build cannot run, so
/// there is exactly one arm and no silent fallback: an unrecognized name failed
/// at load rather than being indexed under a segmentation nobody chose.
fn segmenter_for(config: &Config) -> WindowSegmenter {
    WindowSegmenter {
        lines: config.segmenter.lines,
        overlap: config.segmenter.overlap,
    }
}

/// Read a classifier's `prompts` table into the per-level ladder.
///
/// Config spells it keyed by level as a string — `prompts = { "0" = … }` —
/// because TOML has no integer keys. A key that is not a level is ignored
/// rather than guessed at.
#[cfg(feature = "llm")]
fn prompt_set(classifier: &map_format::config::StageRef) -> map_stages::chat::PromptSet {
    let mut by_level = std::collections::BTreeMap::new();
    if let Some(table) = classifier
        .settings
        .get("prompts")
        .and_then(|v| v.as_table())
    {
        for (level, prompt) in table {
            if let (Ok(level), Some(prompt)) = (level.parse::<u16>(), prompt.as_str()) {
                by_level.insert(level, prompt.to_owned());
            }
        }
    }
    map_stages::chat::PromptSet::new(by_level)
}

/// One segment's classifier output, for inspection.
#[cfg(feature = "llm")]
pub struct Classified {
    /// 1-based inclusive line range in normalized content.
    pub lines: (usize, usize),
    /// Exactly what the model returned.
    pub raw: serde_json::Value,
    /// What would be stored, after flattening.
    pub composed: Option<String>,
}

/// Run the configured `llm` classifier over one resource and return what it
/// said, writing nothing.
///
/// Exists because the stored form is lossy: the structured answer is flattened
/// on the way to disk, so a prompt cannot be iterated on by reading an index.
/// This is the loop — change the prompt, read the actual parts, compare against
/// the same segments — that has to happen before any batch metric is believed.
#[cfg(feature = "llm")]
pub fn classify_resource(
    root: &Path,
    resource: &Path,
    dimension: &str,
    limit: usize,
) -> Result<Vec<Classified>, IndexError> {
    let map_root = map_core::find_map_root(root)
        .ok_or_else(|| IndexError::NotInitialized(root.to_path_buf()))?;
    let config_path = map_root.join(".map/config.toml");
    let config =
        Config::parse(
            &std::fs::read_to_string(&config_path).map_err(|e| IndexError::Io {
                path: config_path.clone(),
                source: e,
            })?,
        )?;

    let dim = config
        .dimensions
        .get(dimension)
        .ok_or_else(|| IndexError::Io {
            path: config_path.clone(),
            source: std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("no dimension named {dimension:?} in this config"),
            ),
        })?;
    let stage = dim
        .classifier
        .as_ref()
        .filter(|c| c.implementation == "llm")
        .ok_or_else(|| IndexError::Io {
            path: config_path.clone(),
            source: std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("dimension {dimension:?} has no `llm` classifier to inspect"),
            ),
        })?;

    let adapter = resolve_llm_adapter(&config)?.ok_or_else(|| IndexError::Io {
        path: config_path,
        source: std::io::Error::new(
            std::io::ErrorKind::NotConnected,
            "no LLM endpoint configured — run `map llm login`",
        ),
    })?;

    let bytes = std::fs::read(resource).map_err(|e| IndexError::Io {
        path: resource.to_path_buf(),
        source: e,
    })?;
    let key = resource
        .strip_prefix(&map_root)
        .unwrap_or(resource)
        .to_string_lossy()
        .replace('\\', "/");
    let resource_meta = map_core::Resource {
        key: key.clone(),
        size: bytes.len() as u64,
    };
    let content = TextPreprocessor
        .preprocess(&resource_meta, &bytes)?
        .ok_or_else(|| IndexError::Io {
            path: resource.to_path_buf(),
            source: std::io::Error::new(std::io::ErrorKind::InvalidData, "not text"),
        })?;
    let mut segments = segmenter_for(&config).segment(&content)?;
    segments.truncate(limit.max(1));

    let classifier = map_stages::chat::LlmClassifier::with_facets(
        adapter,
        prompt_set(stage),
        vec![dimension.to_owned()],
        facet_set(stage),
    );
    let items: Vec<Vec<&str>> = segments.iter().map(|s| vec![s.slice(&content)]).collect();
    let dimensions = [dimension.to_owned()];
    let inspected = classifier.inspect(&ClassifyBatch {
        items: &items,
        dimensions: &dimensions,
        level: 0,
    })?;

    Ok(segments
        .iter()
        .zip(inspected)
        .map(|(segment, one)| Classified {
            lines: (
                map_query_line_of(&content.text, segment.start),
                map_query_line_of(&content.text, segment.end.saturating_sub(1)),
            ),
            raw: one.raw,
            composed: one
                .records
                .get(dimension)
                .and_then(|r| r.descriptor.clone()),
        })
        .collect())
}

/// 1-based line of a byte offset. Duplicated from the query crate rather than
/// depended on: the indexer does not otherwise need it.
#[cfg(feature = "llm")]
fn map_query_line_of(text: &str, offset: u32) -> usize {
    text.as_bytes()[..(offset as usize).min(text.len())]
        .iter()
        .filter(|b| **b == b'\n')
        .count()
        + 1
}

/// Read a classifier's `facets` setting into the enforced answer shape.
///
/// `facets = ["purpose", "behaviour", "names[]", "subsystem"]` makes those
/// parts schema fields rather than a request in prose; a `[]` suffix makes the
/// part a list of short strings. Absent means the original single-string shape.
#[cfg(feature = "llm")]
fn facet_set(classifier: &map_format::config::StageRef) -> map_stages::chat::Facets {
    let parts: Vec<String> = classifier
        .settings
        .get("facets")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    map_stages::chat::Facets::parse(&parts)
}

/// The settings that shape what the shared `llm` request asks for.
///
/// Named so the refusal below can say which one diverged, in the spelling the
/// operator wrote in `config.toml`.
#[cfg(feature = "llm")]
const SHARED_LLM_SETTINGS: [&str; 3] = ["prompts", "prompt", "facets"];

/// Every `llm` dimension must configure the same request, because there is only
/// one.
///
/// The group is a single call covering every dimension it owns, so the prompts
/// and facets can only come from one of them — and each dimension's artifact
/// fingerprint asserts *its own* settings. Taking the first dimension's and
/// serving the rest with them stored records keyed to prompts that never ran.
#[cfg(feature = "llm")]
fn llm_group_agrees(config: &Config) -> Result<(), IndexError> {
    let mut first: Option<(&String, &map_format::config::StageRef)> = None;
    for (name, dim) in config.active() {
        let Some(classifier) = dim.classifier.as_ref() else {
            continue;
        };
        if classifier.implementation != "llm" {
            continue;
        }
        let Some((first_name, first_stage)) = first else {
            first = Some((name, classifier));
            continue;
        };
        for setting in SHARED_LLM_SETTINGS {
            if first_stage.settings.get(setting) != classifier.settings.get(setting) {
                return Err(IndexError::DivergentLlmGroup {
                    first: first_name.clone(),
                    second: name.clone(),
                    setting,
                });
            }
        }
    }
    Ok(())
}

/// One prompt table serves the whole LLM group. The group is one call covering
/// every dimension it owns, so a per-dimension prompt would mean either one
/// call each or a prompt that has to describe all of them at once — and
/// [`llm_group_agrees`] refuses the run rather than let one dimension's prompts
/// stand in for another's.
#[cfg(feature = "llm")]
fn build_llm_classifier(
    config: &Config,
    adapter: std::sync::Arc<map_llm::LlmAdapter>,
) -> Result<Option<map_stages::chat::LlmClassifier>, IndexError> {
    llm_group_agrees(config)?;

    let mut prompts: Option<map_stages::chat::PromptSet> = None;
    let mut facets = map_stages::chat::Facets::Prose;
    let mut dimensions: Vec<String> = Vec::new();
    for (name, dim) in config.active() {
        let Some(classifier) = dim.classifier.as_ref() else {
            continue;
        };
        if classifier.implementation != "llm" {
            continue;
        }
        if prompts.is_none() {
            prompts = Some(prompt_set(classifier));
            facets = facet_set(classifier);
        }
        dimensions.push(name.clone());
    }
    if dimensions.is_empty() {
        return Ok(None);
    }
    Ok(Some(map_stages::chat::LlmClassifier::with_facets(
        adapter,
        prompts.unwrap_or_default(),
        dimensions,
        facets,
    )))
}

/// A dimension fabricates with an embedder (vectors to cluster) and a
/// classifier (something to summarize each group into). Missing either, it is
/// skipped.
///
/// The classifier need not be the LLM one: since one call shape serves every
/// level, `structural` and `content` build a tree too — free, offline, and fast
/// enough to iterate tree shape on without billing anything.
#[cfg(feature = "distilled")]
#[allow(clippy::too_many_arguments)]
fn run_fabrication(
    config: &Config,
    shared: &ObjectStore,
    descriptors: &ObjectStore,
    tensor_store: &ObjectStore,
    map_dir: &Path,
    #[cfg(feature = "llm")] adapter: Option<std::sync::Arc<map_llm::LlmAdapter>>,
    embedder: &dyn Embedder,
    prior: Option<&Manifest>,
    manifest: &mut Manifest,
    stats: &mut IndexStats,
) -> Result<(), IndexError> {
    use fabricate::FabParams;

    let cluster_store = ObjectStore::open(map_dir.join("index/cluster/objects"));

    for (name, dim) in config.active() {
        let Some(fab) = dim.fabricator.as_ref() else {
            continue;
        };
        // Both of these were silent `continue`s until `Config::validate` grew
        // `validate_fabricator`, which refuses each at load. They are kept as
        // refusals rather than deleted so that a validation gap shows up as a
        // named dimension instead of a tree that quietly never got built.
        if fab.implementation != "agglomerative" {
            return Err(IndexError::Unfabricatable {
                dimension: name.clone(),
                detail: format!(
                    "its fabricator impl is {:?}, and the only one implemented is \
                     \"agglomerative\"",
                    fab.implementation
                ),
            });
        }
        let Some(embedder_ref) = dim.embedder.as_ref() else {
            return Err(IndexError::Unfabricatable {
                dimension: name.clone(),
                detail: "it declares a fabricator but no embedder, and clustering is over \
                         vectors"
                    .to_owned(),
            });
        };
        let classifier_impl = dim.classifier.as_ref().map(|c| c.implementation.as_str());

        let params = FabParams {
            threshold: setting_f32(fab, "threshold").unwrap_or(0.70),
            min_cluster: setting_usize(fab, "min_cluster").unwrap_or(5),
            max_cluster: setting_usize(fab, "max_cluster").unwrap_or(15),
            max_levels: setting_u16(fab, "max_levels").unwrap_or(3),
            min_remaining: setting_usize(fab, "min_remaining").unwrap_or(8),
        };
        // The dimension's own classifier labels its clusters, one dimension at a
        // time. The segment-pass instance cannot serve: it is built over the
        // whole implementation group, so its schema would demand every facet in
        // that group of a cluster grouped by only one of them.
        let classifier: Box<dyn Classifier> = match classifier_impl {
            #[cfg(feature = "llm")]
            Some("llm") => match adapter.as_ref() {
                Some(adapter) => {
                    let stage = dim.classifier.as_ref().expect("matched on its impl");
                    Box::new(map_stages::chat::LlmClassifier::with_facets(
                        adapter.clone(),
                        prompt_set(stage),
                        vec![name.clone()],
                        facet_set(stage),
                    ))
                }
                // Configured but no endpoint. Reported by the degraded check as
                // a classifier problem, which is where the fix is.
                None => continue,
            },
            Some("structural") => Box::new(StructuralClassifier),
            Some("content") => Box::new(ContentClassifier),
            Some("declaration") => Box::new(map_stages::declaration::DeclarationClassifier),
            // No classifier, or one this build cannot resolve: the degraded
            // check has already reported it.
            _ => continue,
        };

        let dim_fp = dim.artifact_fingerprint()?;
        let level0 = fabricate::gather_level0(
            manifest,
            shared,
            descriptors,
            tensor_store,
            name,
            classifier_impl,
            &embedder_ref.implementation,
            dim_fp,
        )?;
        if level0.is_empty() {
            continue;
        }

        let fab_fp = fab_fingerprint(dim_fp, name, &classifier.fabric_config());

        fabricate::fabricate_dimension(
            name,
            fab_fp,
            level0,
            &params,
            classifier.as_ref(),
            embedder,
            &cluster_store,
            prior,
            manifest,
            stats,
        )?;
    }
    Ok(())
}

/// The key input for a tensor group's objects.
///
/// The fingerprint a descriptor group's stored records are keyed under.
///
/// Named rather than built inline so the string that claims to identify the
/// stored bytes can be quoted by something that also holds the bytes — see the
/// `ledger` module. Re-spelling this format string in a test would make the
/// test its own drift hazard.
fn descriptor_fingerprint_input(
    classifier_config: &str,
    dimensions: &[String],
    segmenter: Fingerprint,
) -> String {
    format!(
        "{classifier_config}|{}|segmenter={segmenter}",
        dimensions.join(",")
    )
}

/// Separated out so the rule it encodes is testable: a tensor is identified by
/// the embedder, the dimensions it covers, **and** whatever produced the text
/// being embedded.
///
/// The segmenter counts as part of that last clause even when
/// `descriptor_identity` is empty. A content-embedding dimension falls through
/// to raw segment text, so its vectors depend on nothing but the file *and how
/// the file was cut* — one tensor per segment means re-segmenting changes both
/// how many there are and what each covers.
fn tensor_fingerprint_input(
    embedder_config: &str,
    dimensions: &[String],
    descriptor_identity: &str,
    segmenter: Fingerprint,
) -> String {
    format!(
        "{embedder_config}|{}|{descriptor_identity}|segmenter={segmenter}",
        dimensions.join(",")
    )
}

/// Identity of the descriptors this embed group will encode.
///
/// Empty when every dimension in the group falls through to raw segment text —
/// which keeps the key stable for a purely content-embedding dimension, whose
/// tensors depend on nothing but the file.
fn descriptor_identity(
    config: &Config,
    classifiers: &[(&dyn Classifier, Tier)],
    group: &map_core::ImplementationGroup,
) -> String {
    let mut out = String::new();
    for dimension in &group.dimensions {
        let Some(dim) = config.dimensions.get(dimension) else {
            continue;
        };
        if !dim.embeds_descriptor() {
            continue;
        }
        let Some(stage) = dim.classifier.as_ref() else {
            continue;
        };
        // The resolved instance's config, not the raw settings: that is what
        // actually shapes the descriptor, and it is what the descriptor's own
        // object key uses.
        let resolved = classifiers
            .iter()
            .find(|(c, _)| c.implementation() == stage.implementation)
            .map(|(c, _)| c.config())
            .unwrap_or_default();
        out.push_str(dimension);
        out.push('=');
        out.push_str(&resolved);
        out.push(';');
    }
    out
}

/// The fingerprint a dimension's clusters are keyed under.
///
/// The name is folded in here rather than left to `artifact_fingerprint`, which
/// deliberately excludes it so that renaming a dimension does not discard its
/// descriptors. Cluster identity needs the opposite: without the name, two
/// dimensions with byte-identical stage config derive identical level-0 merkle
/// ids, and any clustering they happen to share collides on
/// [`ObjectKey::cluster`] — the same key holding records that differ only in
/// `meta.dimension`. `Manifest::insert_object` refuses that as a hard
/// `KeyCollision`, failing the index partway through.
///
/// `fabric_config` is folded in because `artifact_fingerprint` sees only the
/// *declared* config. A dimension that declares no cluster prompt still gets
/// one from the classifier's built-in fallback, and editing that constant would
/// otherwise relabel every tree while reusing the labels it no longer produces.
///
/// Only `run_fabrication` calls this, and that needs both plugins; its own
/// tests cover it regardless of features, the same arrangement `fabricate` uses.
#[cfg_attr(not(all(feature = "llm", feature = "distilled")), allow(dead_code))]
fn fab_fingerprint(dim_fp: Fingerprint, name: &str, fabric_config: &str) -> Fingerprint {
    Fingerprint::of(format!("{dim_fp}|{name}|{fabric_config}").as_bytes())
}

/// Accepts an integer or a float.
#[cfg(feature = "distilled")]
fn setting_f32(stage: &map_format::config::StageRef, key: &str) -> Option<f32> {
    let v = stage.settings.get(key)?;
    v.as_float()
        .map(|f| f as f32)
        .or_else(|| v.as_integer().map(|i| i as f32))
}

#[cfg(feature = "distilled")]
fn setting_usize(stage: &map_format::config::StageRef, key: &str) -> Option<usize> {
    stage
        .settings
        .get(key)?
        .as_integer()
        .and_then(|i| usize::try_from(i).ok())
}

#[cfg(feature = "distilled")]
fn setting_u16(stage: &map_format::config::StageRef, key: &str) -> Option<u16> {
    stage
        .settings
        .get(key)?
        .as_integer()
        .and_then(|i| u16::try_from(i).ok())
}

/// Delegated so the indexer's availability check and the loader can never
/// disagree about where the model lives.
#[cfg(feature = "distilled")]
fn model_cache_dir() -> std::path::PathBuf {
    map_embed::distilled::model_dir()
}

/// The manifest this run must check the objects on disk against.
///
/// `None` only when there is no manifest at all — a `map init` that has never
/// been indexed. An unreadable one is an error rather than a fresh start:
/// treating it as absent would silently drop the integrity baseline and adopt
/// whatever bytes happen to be there.
fn prior_manifest(map_dir: &Path) -> Result<Option<Manifest>, IndexError> {
    let path = map_dir.join("manifest.json");
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(IndexError::Io { path, source: e }),
    };
    match Manifest::from_bytes(&bytes) {
        Ok(manifest) => Ok(Some(manifest)),
        Err(e) => Err(IndexError::UnreadableManifest {
            path,
            detail: e.to_string(),
        }),
    }
}

/// Re-spell a store's integrity failure as an indexing refusal.
///
/// The store reports a hash mismatch; only the indexer knows what the operator
/// can do about it, and that the remedy costs money for an LLM dimension.
pub(crate) fn as_tampered(error: map_format::Error) -> IndexError {
    match error {
        map_format::Error::IntegrityFailure {
            key,
            expected,
            actual,
        } => IndexError::Tampered {
            key,
            expected,
            actual,
        },
        other => IndexError::Format(other),
    }
}

fn write_object(
    store: &ObjectStore,
    manifest: &mut Manifest,
    stats: &mut IndexStats,
    key: ObjectKey,
    bytes: &[u8],
    tier: Tier,
) -> Result<(), IndexError> {
    let already_present = store.contains(key);
    let entry = store.put(key, bytes, tier)?;
    manifest.insert_object(key, entry)?;

    if already_present {
        stats.objects_reused += 1;
    } else {
        stats.objects_written += 1;
    }
    Ok(())
}

/// Temp-and-rename, so an interrupted run never leaves a truncated manifest —
/// which would read as a corrupt index rather than a partial one.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), IndexError> {
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, bytes).map_err(|e| IndexError::Io {
        path: tmp.clone(),
        source: e,
    })?;
    std::fs::rename(&tmp, path).map_err(|e| IndexError::Io {
        path: path.to_path_buf(),
        source: e,
    })
}

/// The two arrays are parallel to the segmentation, so a slot-for-slot zip is
/// the whole join. Restricting to `wanted` keeps a lexical dimension's
/// descriptors out of a structure only the embedder reads.
fn merge_classified(
    classified: &mut [DimensionRecords],
    per_segment: &[DimensionRecords],
    dimensions: &[String],
    wanted: &BTreeSet<String>,
) {
    for (slot, produced) in classified.iter_mut().zip(per_segment) {
        for dimension in dimensions {
            if wanted.contains(dimension) {
                if let Some(record) = produced.get(dimension) {
                    slot.insert(dimension.clone(), record.clone());
                }
            }
        }
    }
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tensor_is_keyed_by_the_descriptor_it_embeds() {
        // The bug this closes: a prompt edit rewrites every descriptor, because
        // the descriptor's own key folds in the classifier config — but the
        // tensor key did not, so `stat` hit and the embedding of the *previous*
        // prose was reused. The dimension then ranked new labels by old vectors
        // and nothing downstream could tell.
        let dims = vec!["descriptive".to_owned()];
        let seg = Fingerprint::of(b"window:lines=40,overlap=8");
        let before = tensor_fingerprint_input(
            "distilled:potion:dim=512",
            &dims,
            "descriptive=chat-completions:model=m:prompt=describe it:dims=descriptive;",
            seg,
        );
        let after = tensor_fingerprint_input(
            "distilled:potion:dim=512",
            &dims,
            "descriptive=chat-completions:model=m:prompt=describe it in detail:dims=descriptive;",
            seg,
        );
        assert_ne!(before, after, "a prompt edit must re-key the tensors");
    }

    #[test]
    fn a_content_only_dimension_keeps_a_stable_tensor_key() {
        // The other half: a dimension whose embedder falls through to raw text
        // depends on nothing but the file, so nothing about a classifier should
        // move its key. Otherwise every unrelated edit would re-embed the
        // corpus for no change in output.
        let dims = vec!["semantic".to_owned()];
        let seg = Fingerprint::of(b"window:lines=40,overlap=8");
        assert_eq!(
            tensor_fingerprint_input("distilled:potion:dim=512", &dims, "", seg),
            tensor_fingerprint_input("distilled:potion:dim=512", &dims, "", seg),
        );
    }

    #[test]
    fn re_segmenting_re_keys_even_a_content_only_dimension() {
        // "Depends on nothing but the file" was too strong: the payload is one
        // tensor per segment, so cutting the same file differently changes both
        // how many there are and what each covers. Reusing them left an index
        // whose spans and vectors described different segmentations of the same
        // bytes — with no error at index time and no way to see it at query
        // time.
        let dims = vec!["semantic".to_owned()];
        assert_ne!(
            tensor_fingerprint_input(
                "distilled:potion:dim=512",
                &dims,
                "",
                Fingerprint::of(b"window:lines=40,overlap=8"),
            ),
            tensor_fingerprint_input(
                "distilled:potion:dim=512",
                &dims,
                "",
                Fingerprint::of(b"window:lines=20,overlap=4"),
            ),
        );
    }

    #[test]
    fn two_dimensions_with_identical_stages_do_not_collide() {
        // Copy a `[dimensions.*]` stanza under a new name and leave both
        // enabled: `artifact_fingerprint` excludes the name, so both dimensions
        // derive the same level-0 merkle ids. If their clusterings coincide,
        // `ObjectKey::cluster` collides and the index dies mid-run on a
        // KeyCollision. Folding the name in is what prevents it.
        let shared = Fingerprint::of(b"identical stage config");
        assert_ne!(
            fab_fingerprint(shared, "semantic", "fab"),
            fab_fingerprint(shared, "descriptive", "fab"),
        );
    }

    #[test]
    fn cluster_identity_still_tracks_the_stage_config() {
        // Folding the name in must not stop a stage edit from re-keying the
        // tree, or a changed prompt would leave stale labels in place.
        assert_ne!(
            fab_fingerprint(Fingerprint::of(b"before"), "descriptive", "fab"),
            fab_fingerprint(Fingerprint::of(b"after"), "descriptive", "fab"),
        );
    }

    /// Two `llm` dimensions, whose stage settings the caller supplies.
    #[cfg(feature = "llm")]
    fn two_llm_dimensions(first: &str, second: &str) -> Config {
        Config::parse(&format!(
            "version = 1\n\
             [dimensions.purpose]\n\
             description = \"what it is for\"\n\
             classifier = {{ impl = \"llm\", {first} }}\n\
             [dimensions.behaviour]\n\
             description = \"what it does\"\n\
             classifier = {{ impl = \"llm\", {second} }}\n"
        ))
        .expect("the fixture config must parse")
    }

    #[test]
    #[cfg(feature = "llm")]
    fn two_llm_dimensions_that_ask_different_questions_are_refused() {
        // One request covers the whole group, so the second dimension's prompts
        // were silently dropped and its records were built from the first's —
        // while its own artifact fingerprint went on asserting the prompts that
        // never ran. Needs no endpoint: this is a config check.
        let config = two_llm_dimensions(
            "prompts = { \"0\" = \"say what it is for\" }",
            "prompts = { \"0\" = \"say what it does\" }",
        );
        let err = llm_group_agrees(&config).unwrap_err();
        let IndexError::DivergentLlmGroup {
            first,
            second,
            setting,
        } = err
        else {
            panic!("expected a divergent-group refusal, got {err:?}");
        };
        assert_eq!(
            (first.as_str(), second.as_str(), setting),
            ("behaviour", "purpose", "prompts",)
        );
    }

    #[test]
    #[cfg(feature = "llm")]
    fn two_llm_dimensions_asking_the_same_question_are_accepted() {
        // The shape the group exists for: several facets of one request.
        let prompts = "prompts = { \"0\" = \"describe it\" }, facets = [\"purpose\"]";
        assert!(llm_group_agrees(&two_llm_dimensions(prompts, prompts)).is_ok());
    }

    #[test]
    #[cfg(feature = "llm")]
    fn llm_dimensions_differing_only_in_facets_are_refused() {
        // Facets shape the enforced answer schema, which is as much the request
        // as the prompt text is.
        let config = two_llm_dimensions(
            "prompts = { \"0\" = \"describe it\" }, facets = [\"purpose\"]",
            "prompts = { \"0\" = \"describe it\" }, facets = [\"purpose\", \"names[]\"]",
        );
        let err = llm_group_agrees(&config).unwrap_err();
        assert!(
            matches!(&err, IndexError::DivergentLlmGroup { setting, .. } if *setting == "facets"),
            "got {err:?}"
        );
    }

    #[test]
    fn a_label_instruction_the_config_never_declared_still_keys_the_tree() {
        // `artifact_fingerprint` sees declared config only. When a dimension
        // declares no cluster prompt it still gets one — the classifier's
        // built-in fallback — and editing that constant used to relabel every
        // tree while reusing the keys of the labels it no longer produced.
        // `fabric_config` carries the resolved ladder, which is what closes it.
        let dim_fp = Fingerprint::of(b"declares no cluster prompt");
        assert_ne!(
            fab_fingerprint(dim_fp, "descriptive", "ladder=fallbackN=name the theme;"),
            fab_fingerprint(dim_fp, "descriptive", "ladder=fallbackN=name the domain;"),
        );
    }
}
