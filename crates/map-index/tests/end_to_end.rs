//! End-to-end: index a small corpus, then query it.
//!
//! Covers the claims MAP rests on — determinism, incrementality, and
//! that retrieval actually finds the right file — against a real filesystem
//! rather than mocks.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use map_format::{Config, Manifest};

struct Corpus(PathBuf);

impl Corpus {
    /// Build a tiny repository with a `.map` already initialized.
    fn new(tag: &str) -> Self {
        let mut root = std::env::temp_dir();
        root.push(format!("map-e2e-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);

        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(
            root.join("src/session.rs"),
            "pub fn refresh_token(session: &mut Session) {\n    session.expiry = now() + TTL;\n}\n",
        )
        .unwrap();
        fs::write(
            root.join("src/logging.rs"),
            "pub fn configure_logger(level: Level) {\n    set_global_level(level);\n}\n",
        )
        .unwrap();
        fs::write(
            root.join("src/geometry.rs"),
            "pub fn bounding_box(points: &[Point]) -> Rect {\n    Rect::containing(points)\n}\n",
        )
        .unwrap();

        let corpus = Corpus(root);
        corpus.init();
        corpus
    }

    fn init(&self) {
        let map = self.0.join(".map");
        for sub in ["index/shared/objects", "index/desc/objects", "cache"] {
            fs::create_dir_all(map.join(sub)).unwrap();
        }
        let config = Config::zero_config();
        fs::write(map.join("config.toml"), config.to_toml().unwrap()).unwrap();
        let manifest = Manifest::new(&config, "test", 0).unwrap();
        fs::write(map.join("manifest.json"), manifest.to_bytes().unwrap()).unwrap();
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Corpus {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn object_bytes(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut out = BTreeMap::new();
    let mut stack = vec![root.join(".map/index")];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else {
                let relative = path.strip_prefix(root).unwrap().to_path_buf();
                out.insert(relative, fs::read(&path).unwrap());
            }
        }
    }
    out
}

fn query(text: &str) -> map_query::Query {
    map_query::Query::from([("lexical".to_owned(), map_query::QueryTerm::new(text))])
}

/// Add a dimension naming a stage no build can resolve.
fn add_unresolvable_dimension(root: &Path) {
    let path = root.join(".map/config.toml");
    let mut toml = fs::read_to_string(&path).unwrap();
    toml.push_str(
        "\n[dimensions.descriptive]\n\
         description = \"what the code does\"\n\
         classifier = { impl = \"no-such-classifier\", prompts = { \"0\" = \"p\" } }\n",
    );
    fs::write(&path, toml).unwrap();
}

/// Widen the structural group so its descriptor objects take new keys.
///
/// The object key folds in the dimension set the call covered, so adding a
/// second dimension to the same implementation group re-keys every descriptor
/// object and strands the previous generation — the same mechanism that left
/// 18.62 MB behind on ripgrep, reproduced in miniature.
fn widen_the_structural_group(root: &Path) {
    let path = root.join(".map/config.toml");
    let mut toml = fs::read_to_string(&path).unwrap();
    toml.push_str(
        "\n[dimensions.other]\n\
         description = \"a second facet sharing the structural classifier\"\n\
         classifier = { impl = \"structural\" }\n",
    );
    fs::write(&path, toml).unwrap();
}

#[test]
fn a_freshly_built_index_has_nothing_to_collect() {
    let corpus = Corpus::new("gc-clean");
    map_index::run(corpus.path()).unwrap();

    let report = map_index::gc::collect(corpus.path(), false).unwrap();
    assert!(report.reachable > 0, "the index should reach something");
    assert!(
        report.unreachable.is_empty(),
        "nothing should be collectable: {:?}",
        report.unreachable
    );
}

#[test]
fn objects_stranded_by_a_config_edit_are_collected() {
    let corpus = Corpus::new("gc-strands");
    map_index::run(corpus.path()).unwrap();
    let before = object_bytes(corpus.path()).len();

    widen_the_structural_group(corpus.path());
    map_index::run(corpus.path()).unwrap();
    assert!(
        object_bytes(corpus.path()).len() > before,
        "re-keying should have added a generation without removing the old one"
    );

    let on_disk = object_bytes(corpus.path()).len();
    let report = map_index::gc::collect(corpus.path(), false).unwrap();
    assert!(
        !report.unreachable.is_empty(),
        "the superseded descriptor objects should be unreachable"
    );
    assert!(report.bytes > 0);

    // Reporting is not collecting — the whole safety model rests on this.
    assert!(!report.pruned);
    assert_eq!(
        object_bytes(corpus.path()).len(),
        on_disk,
        "a report without --prune must not remove anything"
    );
}

#[test]
fn pruning_removes_the_unreachable_and_keeps_the_index_working() {
    let corpus = Corpus::new("gc-prune");
    map_index::run(corpus.path()).unwrap();
    widen_the_structural_group(corpus.path());
    map_index::run(corpus.path()).unwrap();

    let before = map_index::gc::collect(corpus.path(), false).unwrap();
    assert!(!before.unreachable.is_empty());
    let on_disk_before = object_bytes(corpus.path()).len();

    let pruned = map_index::gc::collect(corpus.path(), true).unwrap();
    assert!(pruned.pruned);
    assert_eq!(pruned.unreachable.len(), before.unreachable.len());
    assert_eq!(
        object_bytes(corpus.path()).len(),
        on_disk_before - before.unreachable.len(),
        "exactly the unreachable objects should be gone"
    );

    // Collecting twice is a no-op, and the index still answers.
    let again = map_index::gc::collect(corpus.path(), false).unwrap();
    assert!(again.unreachable.is_empty());
    let index = map_query::Index::open(corpus.path()).unwrap();
    assert!(!index.find(&query("refresh_token"), 5).unwrap().is_empty());
}

fn git(dir: &Path, args: &[&str]) -> bool {
    std::process::Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[test]
fn an_object_another_branch_still_references_is_not_collectable() {
    // The corruption this exists to prevent: prune against the working-tree
    // manifest alone, merge the other branch back, and git resolves
    // delete-versus-unmodified by deleting — leaving a manifest pointing at an
    // object that is gone.
    let corpus = Corpus::new("gc-refs");
    if !git(corpus.path(), &["init", "-q", "-b", "main"]) {
        eprintln!("skipping: git is not available");
        return;
    }
    git(corpus.path(), &["config", "user.name", "gc"]);
    git(corpus.path(), &["config", "user.email", "gc@localhost"]);

    map_index::run(corpus.path()).unwrap();
    assert!(git(corpus.path(), &["add", "-A"]));
    assert!(git(corpus.path(), &["commit", "-q", "-m", "v1"]));
    assert!(git(corpus.path(), &["branch", "old-config"]));

    widen_the_structural_group(corpus.path());
    map_index::run(corpus.path()).unwrap();
    assert!(git(corpus.path(), &["add", "-A"]));
    assert!(git(corpus.path(), &["commit", "-q", "-m", "v2"]));

    let held = map_index::gc::collect(corpus.path(), false).unwrap();
    assert!(
        matches!(held.scope, map_index::GitScope::Refs(_)),
        "expected the refs to be consulted, got {:?}",
        held.scope
    );
    assert!(
        held.unreachable.is_empty(),
        "the old branch still references these: {:?}",
        held.unreachable
    );

    // And the check is load-bearing rather than decorative: drop the branch and
    // the very same object becomes collectable.
    assert!(git(corpus.path(), &["branch", "-D", "old-config"]));
    let freed = map_index::gc::collect(corpus.path(), false).unwrap();
    assert!(
        !freed.unreachable.is_empty(),
        "with no ref holding it, the superseded object should be collectable"
    );
    assert!(freed.reachable < held.reachable);
}

#[test]
fn an_unreadable_manifest_refuses_rather_than_collecting_everything() {
    // The catastrophic misreading: a manifest that will not parse reaches
    // nothing, so treating it as authoritative would delete the whole index.
    let corpus = Corpus::new("gc-bad-manifest");
    map_index::run(corpus.path()).unwrap();
    let before = object_bytes(corpus.path()).len();

    fs::write(corpus.path().join(".map/manifest.json"), b"{ not json").unwrap();

    let err = map_index::gc::collect(corpus.path(), true).unwrap_err();
    assert!(
        matches!(err, map_index::IndexError::GcUnsafe(_)),
        "got {err:?}"
    );
    assert_eq!(
        object_bytes(corpus.path()).len(),
        before,
        "a refused collection must delete nothing"
    );
}

#[test]
fn a_configured_stage_this_build_cannot_run_stops_the_index() {
    // The failure this guards is invisible after the fact: the dimension is
    // dropped, the index still builds, still queries, and still looks right.
    // Worse for a semantic dimension, whose embedder then falls back to raw
    // text and keeps producing tensors that no longer mean what its
    // description claims.
    let corpus = Corpus::new("degraded-abort");
    add_unresolvable_dimension(corpus.path());

    let err = map_index::run(corpus.path()).unwrap_err();
    let map_index::IndexError::Degraded(stages) = err else {
        panic!("expected a Degraded refusal, got {err:?}");
    };
    assert_eq!(stages.len(), 1);
    assert_eq!(stages[0].dimension, "descriptive");
    assert_eq!(stages[0].stage, "classifier");

    // "Nothing was written" is part of the promise, not just the message.
    assert!(
        object_bytes(corpus.path()).is_empty(),
        "a refused run must leave no objects behind"
    );
}

#[test]
fn an_unresolvable_stage_can_be_accepted_explicitly() {
    // Building without a stage is legitimate — it is how a semantic dimension
    // gets configured before its endpoint is live. It just has to be asked for.
    let corpus = Corpus::new("degraded-allow");
    add_unresolvable_dimension(corpus.path());

    let stats =
        map_index::run_with_progress(corpus.path(), map_index::Degraded::Allow, |_| {}).unwrap();
    assert!(stats.indexed > 0);
    // Recorded rather than forgotten: consent is not a reason to stop saying
    // the index is incomplete.
    assert_eq!(stats.degraded.len(), 1);
    assert_eq!(stats.degraded[0].dimension, "descriptive");
}

#[test]
fn the_zero_config_index_is_never_degraded() {
    // The first-run promise: no key, no network, no model, and nothing to ask
    // about. If this ever prompts or refuses, the default experience is broken.
    let corpus = Corpus::new("degraded-zero-config");
    let stats = map_index::run(corpus.path()).unwrap();
    assert!(stats.degraded.is_empty());
}

#[test]
fn index_then_find_locates_the_right_file() {
    let corpus = Corpus::new("find");
    let stats = map_index::run(corpus.path()).unwrap();
    assert_eq!(stats.indexed, 3);
    assert!(stats.segments >= 3);

    let index = map_query::Index::open(corpus.path()).unwrap();
    assert_eq!(index.dimensions(), ["declaration", "lexical"]);

    let hits = index.find(&query("refresh token"), 5).unwrap();
    assert_eq!(
        hits[0].resource,
        "src/session.rs",
        "the session file should win; got {:?}",
        hits.iter().map(|h| &h.resource).collect::<Vec<_>>()
    );

    // A different facet of the same corpus.
    let hits = index.find(&query("bounding box"), 5).unwrap();
    assert_eq!(hits[0].resource, "src/geometry.rs");
}

#[test]
fn a_declared_name_outranks_its_callers_in_the_declaration_dimension() {
    let corpus = Corpus::new("declaration");
    // Three calls in one file: under BM25 over whole content this file
    // outscores the one-line declaration in session.rs.
    fs::write(
        corpus.path().join("src/callers.rs"),
        "pub fn tick(s: &mut Session) {\n    refresh_token(s);\n    refresh_token(s);\n    refresh_token(s);\n}\n",
    )
    .unwrap();
    map_index::run(corpus.path()).unwrap();
    let index = map_query::Index::open(corpus.path()).unwrap();

    let declared = map_query::Query::from([(
        "declaration".to_owned(),
        map_query::QueryTerm::new("refresh_token"),
    )]);
    let hits = index.find(&declared, 5).unwrap();
    assert_eq!(
        hits[0].resource,
        "src/session.rs",
        "the declaring file should win; got {:?}",
        hits.iter().map(|h| &h.resource).collect::<Vec<_>>()
    );
    assert!(
        hits.iter().all(|h| h.resource != "src/callers.rs"),
        "a segment that only calls the name has no record in this dimension"
    );
}

#[test]
fn camel_case_and_snake_case_reach_the_same_code() {
    let corpus = Corpus::new("tokenize");
    map_index::run(corpus.path()).unwrap();
    let index = map_query::Index::open(corpus.path()).unwrap();

    for spelling in ["refresh_token", "refreshToken", "refresh"] {
        let hits = index.find(&query(spelling), 5).unwrap();
        assert_eq!(
            hits.first().map(|h| h.resource.as_str()),
            Some("src/session.rs"),
            "{spelling:?} should reach session.rs"
        );
    }
}

#[test]
fn indexing_twice_is_byte_identical() {
    // Tier A: a default build must be bit-reproducible.
    let corpus = Corpus::new("determinism");
    map_index::run(corpus.path()).unwrap();
    let first = object_bytes(corpus.path());

    map_index::run(corpus.path()).unwrap();
    let second = object_bytes(corpus.path());

    assert_eq!(first, second, "a second index run changed stored bytes");
    assert!(!first.is_empty());
}

#[test]
fn unchanged_resources_do_not_re_run_the_classifier() {
    // The classifier is the expensive stage. Deduplicating on write would
    // still pay for an LLM call and discard the answer.
    let corpus = Corpus::new("incremental");
    let cold = map_index::run(corpus.path()).unwrap();
    // Two offline classifier groups (structural, declaration) per resource.
    assert_eq!(cold.classifier_calls, 6);

    let warm = map_index::run(corpus.path()).unwrap();
    assert_eq!(warm.classifier_calls, 0, "nothing changed; nothing to do");
    assert_eq!(warm.objects_written, 0);

    // Touch exactly one file.
    fs::write(
        corpus.path().join("src/logging.rs"),
        "pub fn configure_logger(level: Level) {\n    set_global_level(level);\n    flush();\n}\n",
    )
    .unwrap();

    let after = map_index::run(corpus.path()).unwrap();
    assert_eq!(
        after.classifier_calls, 2,
        "work must be O(changed), not O(corpus): one resource, two classifier groups"
    );
}

#[test]
fn re_segmenting_reclassifies_rather_than_reusing_the_old_segments_answers() {
    // Caught by indexing a corpus, not by reading the code. Changing
    // `[segmenter]` re-cut every resource, the spans object re-keyed correctly
    // — and the descriptors did not, because their key folded in the classifier
    // config but never the segmenter. `map index` reported "classifier calls: 0"
    // and left an index whose spans described twice as many segments as its
    // descriptors held. Nothing errored, and no query could reveal it.
    let corpus = Corpus::new("resegment");
    let cold = map_index::run(corpus.path()).unwrap();
    // Two offline classifier groups (structural, declaration) per resource.
    assert_eq!(cold.classifier_calls, 6);

    let path = corpus.path().join(".map/config.toml");
    let toml = fs::read_to_string(&path).unwrap();
    let resegmented = toml
        .replace("lines = 40", "lines = 2")
        .replace("overlap = 8", "overlap = 1");
    assert_ne!(
        toml, resegmented,
        "the corpus config must carry a segmenter to edit"
    );
    fs::write(&path, resegmented).unwrap();

    let after = map_index::run(corpus.path()).unwrap();
    assert_eq!(
        after.classifier_calls, 6,
        "every resource was re-cut, so every resource must be re-classified by both groups"
    );
}

/// A `semantic` dimension whose classifier shapes content but stores nothing.
#[cfg(feature = "distilled")]
fn add_pass_through_dimension(root: &Path) {
    let path = root.join(".map/config.toml");
    let mut toml = fs::read_to_string(&path).unwrap();
    toml.push_str(
        "\n[dimensions.semantic]\n\
         description = \"content resembling your target\"\n\
         classifier = { impl = \"content\", persist_output = false }\n\
         embedder = { impl = \"distilled\" }\n",
    );
    fs::write(&path, toml).unwrap();
}

/// Count objects under one tier directory of the index.
#[cfg(feature = "distilled")]
fn objects_under(root: &Path, tier: &str) -> usize {
    let prefix = Path::new(".map/index").join(tier);
    object_bytes(root)
        .keys()
        .filter(|p| p.starts_with(&prefix))
        .count()
}

/// Needs the potion weights on disk; never runs in CI, which builds default
/// features only.
#[test]
#[cfg(feature = "distilled")]
fn changing_the_classifier_re_embeds_rather_than_reusing_stale_tensors() {
    // Swapping `structural` for `content` changes the descriptor the embedder
    // encodes, exactly as editing a prompt would. Before the key folded in the
    // classifier, the descriptors were rewritten and the tensors were *reused*,
    // so the dimension ranked new text by the old text's vectors.
    //
    // Two offline classifiers with different configs stand in for the prompt
    // edit, so this needs no endpoint.
    let corpus = Corpus::new("tensor-rekey");
    let path = corpus.path().join(".map/config.toml");
    let base = fs::read_to_string(&path).unwrap();

    let with = |classifier: &str| {
        format!(
            "{base}\n[dimensions.facet]\n\
             description = \"a dimension whose tensors encode its descriptor\"\n\
             classifier = {{ impl = \"{classifier}\" }}\n\
             embedder = {{ impl = \"distilled\" }}\n"
        )
    };

    fs::write(&path, with("structural")).unwrap();
    map_index::run(corpus.path()).unwrap();
    let first: BTreeMap<PathBuf, Vec<u8>> = object_bytes(corpus.path())
        .into_iter()
        .filter(|(p, _)| p.starts_with(Path::new(".map/index").join("tensor")))
        .collect();
    assert!(!first.is_empty(), "the embedder must have run");

    fs::write(&path, with("content")).unwrap();
    let stats = map_index::run(corpus.path()).unwrap();
    let second: BTreeMap<PathBuf, Vec<u8>> = object_bytes(corpus.path())
        .into_iter()
        .filter(|(p, _)| p.starts_with(Path::new(".map/index").join("tensor")))
        .collect();

    assert!(
        stats.embed_calls > 0,
        "a changed descriptor must be re-embedded, not reused"
    );
    let fresh: Vec<_> = second.keys().filter(|k| !first.contains_key(*k)).collect();
    assert!(
        !fresh.is_empty(),
        "the new descriptors must land under new tensor keys; got none"
    );
}

/// Needs the potion weights on disk; never runs in CI, which builds default
/// features only.
#[test]
#[cfg(feature = "distilled")]
fn a_tree_can_be_built_with_no_llm_at_all() {
    // One call shape serves every level, so `content` labels clusters as
    // readily as the LLM classifier does. That makes tree-shape experiments
    // free and instant instead of a billed reindex, which is the whole point of
    // no longer gating fabrication on an endpoint.
    let corpus = Corpus::new("offline-tree");
    let path = corpus.path().join(".map/config.toml");
    let mut toml = fs::read_to_string(&path).unwrap();
    toml.push_str(
        "\n[dimensions.offline]\n\
         description = \"a tree with no endpoint anywhere\"\n\
         classifier = { impl = \"structural\" }\n\
         embedder = { impl = \"distilled\" }\n\
         fabricator = { impl = \"agglomerative\", threshold = 0.1, min_cluster = 2, max_levels = 1, min_remaining = 1 }\n",
    );
    fs::write(&path, toml).unwrap();

    let stats = map_index::run(corpus.path()).unwrap();
    assert!(
        stats.clusters_written > 0,
        "a tree must be built without an endpoint; wrote {} clusters \
         (segments {}, embed calls {}, label calls {}, unlabeled {}, degraded {:?})",
        stats.clusters_written,
        stats.segments,
        stats.embed_calls,
        stats.label_calls,
        stats.unlabeled_clusters,
        stats.degraded
    );
    assert!(
        stats.unlabeled_clusters == 0,
        "and every cluster got a label"
    );

    // Recorded as built, so the load-time check has something to compare with.
    let manifest =
        Manifest::from_bytes(&fs::read(corpus.path().join(".map/manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest.dimensions["offline"].levels, vec![0, 1]);
}

/// Needs the potion weights on disk; never runs in CI, which builds default
/// features only.
#[test]
#[cfg(feature = "distilled")]
fn a_pass_through_classifier_writes_no_descriptor_object() {
    // The reason `persist_output` exists: naming the stage must not cost a
    // second copy of the corpus in `desc/`.
    let corpus = Corpus::new("pass-through");
    map_index::run(corpus.path()).unwrap();
    let lexical_only = objects_under(corpus.path(), "desc");

    add_pass_through_dimension(corpus.path());
    map_index::run(corpus.path()).unwrap();

    // Positive control first. Without it a missing model would make the
    // descriptor assertion below pass for the wrong reason — nothing was
    // indexed at all.
    assert!(
        objects_under(corpus.path(), "tensor") > 0,
        "the embedder must have run, or this test proves nothing"
    );
    assert_eq!(
        objects_under(corpus.path(), "desc"),
        lexical_only,
        "a pass-through classifier must add no descriptor objects"
    );
}

#[test]
fn a_pack_from_another_format_version_is_rebuilt_not_refused() {
    // Packs are gitignored derived cache, rebuildable from the objects. Reading
    // an old one as a hard error froze the layout by accident: bumping VERSION
    // would have broken every existing cache instead of costing one rebuild.
    let corpus = Corpus::new("pack-version");
    map_index::run(corpus.path()).unwrap();

    let index = map_query::Index::open(corpus.path()).unwrap();
    assert!(!index.find(&query("refresh_token"), 5).unwrap().is_empty());
    // The pack is memory-mapped; Windows refuses to write a file with a mapped
    // section open, so let go of it before doctoring the header.
    drop(index);

    let path = corpus.path().join(".map/cache/lexical.pack");
    let mut bytes = fs::read(&path).unwrap();
    bytes[8..12].copy_from_slice(&99u32.to_le_bytes());
    fs::write(&path, &bytes).unwrap();

    // Proves the doctored byte really is rejected by the reader — without this
    // the test could pass because nothing had changed.
    match map_stages::Pack::open(&path, "lexical") {
        Err(e) => assert!(e.is_rebuildable(), "{e}"),
        Ok(_) => panic!("the doctored version word must be rejected"),
    }

    let reopened = map_query::Index::open(corpus.path())
        .expect("a pack this build cannot read must rebuild, not refuse");
    assert!(
        !reopened
            .find(&query("refresh_token"), 5)
            .unwrap()
            .is_empty(),
        "and the rebuilt index must answer"
    );
}

#[test]
fn identical_files_share_one_object() {
    // Payloads carry no resource identity, so duplicate content dedups
    // instead of colliding.
    let corpus = Corpus::new("dedup");
    fs::write(
        corpus.path().join("src/copy.rs"),
        fs::read(corpus.path().join("src/session.rs")).unwrap(),
    )
    .unwrap();

    let stats = map_index::run(corpus.path()).unwrap();
    assert_eq!(stats.indexed, 4);
    assert!(
        stats.objects_reused > 0,
        "the duplicate file should reuse the original's objects"
    );

    // Both paths must still be independently findable.
    let index = map_query::Index::open(corpus.path()).unwrap();
    let hits = index.find(&query("refresh token"), 10).unwrap();
    let found: Vec<&str> = hits.iter().map(|h| h.resource.as_str()).collect();
    assert!(found.contains(&"src/session.rs"));
    assert!(found.contains(&"src/copy.rs"));
}

#[test]
fn a_partial_query_is_valid_and_an_unknown_dimension_is_not() {
    let corpus = Corpus::new("query-shape");
    map_index::run(corpus.path()).unwrap();
    let index = map_query::Index::open(corpus.path()).unwrap();

    // Empty query field is ignored rather than being an error.
    let empty = map_query::Query::from([("lexical".to_owned(), map_query::QueryTerm::new("   "))]);
    assert!(index.find(&empty, 5).unwrap().is_empty());

    let unknown = map_query::Query::from([(
        "descriptive".to_owned(),
        map_query::QueryTerm::new("anything"),
    )]);
    assert!(index.find(&unknown, 5).is_err());
}

#[test]
fn refresh_makes_an_edit_findable_without_an_explicit_index() {
    // The point of the flag: edit a file, query, get the new content — no
    // "did you remember to run map index" step.
    let corpus = Corpus::new("refresh");
    map_index::run(corpus.path()).unwrap();

    fs::write(
        corpus.path().join("src/logging.rs"),
        "pub fn configure_logger(level: Level) {\n    quiesce_the_reactor();\n}\n",
    )
    .unwrap();

    // Without a refresh the index still describes the old content.
    let stale = map_query::Index::open(corpus.path()).unwrap();
    assert!(stale.find(&query("quiesce reactor"), 5).unwrap().is_empty());

    // Forcing a refresh repairs only what changed, then finds it.
    let stats = map_index::refresh(corpus.path()).unwrap().unwrap();
    // One changed resource, two classifier groups.
    assert_eq!(stats.classifier_calls, 2, "repair must be O(changed)");

    let fresh = map_query::Index::open(corpus.path()).unwrap();
    let hits = fresh.find(&query("quiesce reactor"), 5).unwrap();
    assert_eq!(hits[0].resource, "src/logging.rs");
}

#[test]
fn a_pulled_manifest_is_not_shadowed_by_a_stale_cache_pack() {
    // The pull hazard: a teammate reindexes and commits new objects + manifest;
    // you pull; your gitignored cache packs still describe the old content. The
    // pack marker must catch the mismatch and rebuild, not serve the stale pack.
    let corpus = Corpus::new("stale-cache");
    map_index::run(corpus.path()).unwrap();

    // Snapshot the cache as it stands for the original content (state A). The
    // marker is computed here, while A's manifest is still the one on disk, so
    // it is exactly what the indexer recorded for A's pack.
    let cache = corpus.path().join(".map/cache");
    let map_dir = corpus.path().join(".map");
    let pack_a = fs::read(cache.join("lexical.pack")).unwrap();
    let marker_a = map_stages::cache::pack_marker(&map_dir, "lexical", &pack_a)
        .expect("a locally built pack carries a marker");
    assert!(
        map_stages::cache::PackMarkers::read(&cache).trusts("lexical", &marker_a),
        "the indexer must record a marker for each pack it writes"
    );

    // New committed content (state B): manifest, objects, packs, and marker all
    // advance to describe "quiesce_the_reactor".
    fs::write(
        corpus.path().join("src/logging.rs"),
        "pub fn configure_logger(level: Level) {\n    quiesce_the_reactor();\n}\n",
    )
    .unwrap();
    map_index::run(corpus.path()).unwrap();

    // Simulate the pull: manifest and objects are B's, but the cache pack and
    // its marker are still A's.
    fs::write(cache.join("lexical.pack"), &pack_a).unwrap();
    let mut stale_markers = map_stages::cache::PackMarkers::default();
    stale_markers.insert("lexical", marker_a);
    stale_markers.write(&cache).unwrap();

    // Opening without `-u` must notice the marker disagrees with the manifest
    // and rebuild from the committed objects rather than trust the stale pack.
    let index = map_query::Index::open(corpus.path()).unwrap();
    assert_eq!(
        index.load_path(),
        map_query::LoadPath::Rebuilt,
        "a manifest that disagrees with the pack marker must force a rebuild"
    );
    let hits = index.find(&query("quiesce reactor"), 5).unwrap();
    assert_eq!(
        hits[0].resource, "src/logging.rs",
        "the rebuilt index must reflect the committed objects, not the stale pack"
    );
}

#[test]
fn a_forged_cache_marker_is_rejected_but_a_local_one_is_trusted() {
    // #4: the mmap fast path must not trust a pack a hostile clone force-added.
    // A locally built cache carries a marker keyed with this machine's secret;
    // any marker an attacker could commit (they lack the key) is rejected.
    let corpus = Corpus::new("forged-marker");
    map_index::run(corpus.path()).unwrap();
    let cache = corpus.path().join(".map/cache");

    // Locally built: trusted on the fast path, no rebuild.
    let trusted = map_query::Index::open(corpus.path()).unwrap();
    assert_eq!(
        trusted.load_path(),
        map_query::LoadPath::Mapped,
        "a pack this machine built for the current manifest must be trusted"
    );

    // A well-formed line for the right dimension, carrying a marker not keyed
    // with this machine's secret — the best an attacker shipping a repo can do.
    // Well-formed on purpose: a malformed file would be rejected for merely
    // failing to parse, which would prove nothing about the key.
    let mut forged = map_stages::cache::PackMarkers::default();
    forged.insert("lexical", "forged-by-a-hostile-clone".to_owned());
    forged.write(&cache).unwrap();
    let index = map_query::Index::open(corpus.path()).unwrap();
    assert_eq!(
        index.load_path(),
        map_query::LoadPath::Rebuilt,
        "a marker not keyed with this machine's secret must not be trusted"
    );
    assert!(
        !index.find(&query("refresh token"), 5).unwrap().is_empty(),
        "the rebuilt index still answers, from the verified committed objects"
    );
}

#[test]
fn refresh_repairs_only_what_changed() {
    let corpus = Corpus::new("freshness");
    map_index::run(corpus.path()).unwrap();

    // An untouched tree means there is nothing to do — the stat short-circuit
    // must skip the index pass rather than re-reading every file to discover
    // that.
    assert!(
        map_index::refresh(corpus.path()).unwrap().is_none(),
        "an unchanged corpus must not trigger an index pass"
    );

    // One edited file costs one classifier call, not a full pass.
    fs::write(
        corpus.path().join("src/geometry.rs"),
        "pub fn bounding_box(points: &[Point]) -> Rect {\n    Rect::hull(points)\n}\n",
    )
    .unwrap();
    let did = map_index::refresh(corpus.path()).unwrap();
    // One changed resource, two classifier groups.
    assert_eq!(did.expect("an edit should refresh").classifier_calls, 2);
}

#[test]
fn an_unchanged_corpus_short_circuits_before_reading_anything() {
    // Detecting "nothing changed" by running a full index pass costs
    // O(corpus) — the thing the packed index exists to avoid. The stat
    // snapshot answers it without opening a single resource.
    let corpus = Corpus::new("shortcircuit");
    map_index::run(corpus.path()).unwrap();

    assert!(map_index::refresh(corpus.path()).unwrap().is_none());

    // Deleting a file counts as a change, not just editing one.
    fs::remove_file(corpus.path().join("src/geometry.rs")).unwrap();
    assert!(map_index::refresh(corpus.path()).unwrap().is_some());
}

/// One stored descriptor object as `(path on disk, the key it is stored under)`.
fn a_descriptor_object(root: &Path) -> (PathBuf, String) {
    let relative = object_bytes(root)
        .into_keys()
        .find(|p| p.starts_with(Path::new(".map/index/desc/objects")))
        .expect("the structural classifier must have stored descriptors");
    let shard = relative.parent().unwrap().file_name().unwrap();
    let rest = relative.file_name().unwrap();
    (
        root.join(&relative),
        format!("{}{}", shard.to_string_lossy(), rest.to_string_lossy()),
    )
}

#[test]
fn an_object_altered_since_it_was_recorded_stops_the_run() {
    // Spec §8: a committed index is untrusted data that steers a model's
    // attention. Reuse is a `stat` hit on an input-addressed key, which a
    // tampered descriptor still satisfies — so without checking the recorded
    // content hash the altered bytes would be re-recorded under a freshly
    // computed hash and handed to the next query as if the indexer had written
    // them.
    let corpus = Corpus::new("tampered");
    map_index::run(corpus.path()).unwrap();

    let manifest_path = corpus.path().join(".map/manifest.json");
    let before = fs::read(&manifest_path).unwrap();

    let (path, key) = a_descriptor_object(corpus.path());
    let mut bytes = fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0x20;
    fs::write(&path, &bytes).unwrap();

    let err = map_index::run(corpus.path()).unwrap_err();
    let map_index::IndexError::Tampered { key: named, .. } = &err else {
        panic!("expected a tampering refusal, got {err:?}");
    };
    assert_eq!(*named, key);
    let message = err.to_string();
    assert!(
        message.contains("altered after it was recorded") && message.contains("re-bills"),
        "the message must name the remedy and its cost: {message}"
    );

    assert_eq!(
        fs::read(&manifest_path).unwrap(),
        before,
        "a refused run must leave the previous manifest — the integrity baseline — intact"
    );
}

#[test]
fn an_unreadable_manifest_stops_the_run_rather_than_rebuilding_over_it() {
    // The manifest is the only record of what each object's bytes should hash
    // to. Indexing over one that will not parse would adopt whatever is on disk
    // as correct — the same misreading collection already refuses (spec §9.4).
    let corpus = Corpus::new("unreadable-manifest");
    map_index::run(corpus.path()).unwrap();

    let manifest_path = corpus.path().join(".map/manifest.json");
    fs::write(&manifest_path, b"{ not json").unwrap();

    let err = map_index::run(corpus.path()).unwrap_err();
    let map_index::IndexError::UnreadableManifest { path, .. } = &err else {
        panic!("expected a refusal naming the manifest, got {err:?}");
    };
    // The root is canonicalized on the way in, so compare the tail rather than
    // the whole path — on Windows that is a `\\?\` prefix and a long temp path
    // against the short one this test built.
    assert!(path.ends_with("manifest.json"), "{}", path.display());
}

#[test]
fn an_edited_config_is_stale_even_though_no_resource_moved() {
    // Every object key folds in the config fingerprint, so widening a group
    // re-keys the whole index without touching one mtime. Comparing stats alone
    // reported a clean tree and `-u` short-circuited, leaving the edit unapplied
    // until something happened to touch a file.
    let corpus = Corpus::new("stale-config");
    map_index::run(corpus.path()).unwrap();
    assert!(!map_index::stale(corpus.path()).unwrap().unwrap().is_stale());

    widen_the_structural_group(corpus.path());

    let report = map_index::stale(corpus.path()).unwrap().unwrap();
    assert!(report.config_changed, "{report}");
    assert_eq!(report.changed, 0, "no resource was touched");
    assert!(
        map_index::refresh(corpus.path()).unwrap().is_some(),
        "a config edit must make `-u` do the work rather than short-circuit"
    );
}

#[test]
fn a_degraded_run_records_no_snapshot_to_short_circuit_on() {
    // The index is missing dimensions its config describes, so "nothing has
    // drifted" would be false: the next `-u`, once the stage is available, must
    // do the work rather than compare an unchanged tree and skip it.
    let corpus = Corpus::new("degraded-snapshot");
    add_unresolvable_dimension(corpus.path());

    let stats =
        map_index::run_with_progress(corpus.path(), map_index::Degraded::Allow, |_| {}).unwrap();
    assert!(!stats.degraded.is_empty());

    let snapshot = corpus.path().join(".map/cache/stat.json");
    assert!(
        !snapshot.exists(),
        "a degraded run must leave no staleness snapshot"
    );
    assert!(
        map_index::stale(corpus.path()).unwrap().is_none(),
        "with no snapshot, drift is unknown rather than zero"
    );

    // And a later complete run does record one.
    fs::write(
        corpus.path().join(".map/config.toml"),
        fs::read_to_string(corpus.path().join(".map/config.toml"))
            .unwrap()
            .replace("no-such-classifier", "structural"),
    )
    .unwrap();
    map_index::run(corpus.path()).unwrap();
    assert!(snapshot.exists());
}

#[test]
fn an_edit_during_a_run_is_caught_by_the_next_staleness_check() {
    // The snapshot is taken after discovery and before the first resource is
    // read, so a file edited *while* the run is working is recorded at its
    // pre-edit stat. Stat'ing at the end instead records the post-edit stat
    // against the pre-edit content the run actually indexed, and the edit is
    // then invisible forever — the window that matters is a slow LLM pass.
    let corpus = Corpus::new("edit-during-run");
    map_index::run(corpus.path()).unwrap();

    // Something has to have drifted, or the refresh short-circuits on the stat
    // snapshot and never reaches a resource at all.
    fs::write(
        corpus.path().join("src/logging.rs"),
        "pub fn configure_logger(level: Level) {\n    set_global_level(level);\n    flush();\n}\n",
    )
    .unwrap();

    let edited = corpus.path().join("src/geometry.rs");
    let mut done = 0usize;
    map_index::refresh_with_progress(corpus.path(), map_index::Degraded::Abort, |progress| {
        // Fires before the resource at `done` is read, and discovery is sorted,
        // so by the third callback geometry.rs has already been indexed. No
        // sleep and no clock resolution involved: the edit changes the length.
        done = progress.done;
        if progress.done == 2 {
            fs::write(
                &edited,
                "pub fn bounding_box(points: &[Point]) -> Rect {\n    Rect::hull(points)\n}\n\
                 // landed while the run was still working\n",
            )
            .unwrap();
        }
    })
    .unwrap();
    assert_eq!(done, 2, "the run must have reached the third resource");

    let report = map_index::stale(corpus.path()).unwrap().unwrap();
    assert_eq!(
        report.changed, 1,
        "the edit landed after its file was read, so it must still be pending: {report}"
    );
}

#[test]
fn snippets_resolve_back_to_source() {
    let corpus = Corpus::new("snippet");
    map_index::run(corpus.path()).unwrap();
    let index = map_query::Index::open(corpus.path()).unwrap();

    let hits = index.find(&query("refresh token"), 1).unwrap();
    let (text, line) = index.snippet(&hits[0]).unwrap();
    assert!(text.contains("refresh_token"));
    assert!(line >= 1);
}
