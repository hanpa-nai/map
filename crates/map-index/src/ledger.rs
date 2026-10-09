//! The fingerprint ledger: what each stage's identity claimed, and what it
//! produced.
//!
//! # The failure this closes
//!
//! An edit changes a stage's **behaviour** without moving the **fingerprint**
//! that keys its stored bytes. [`ObjectStore::stat`](map_format::ObjectStore)
//! then hits, the stage never runs, and the index reuses output the live code
//! would no longer produce. Nothing reports an error: `map index` says
//! `classifier calls: 0` and the query path mixes generations silently.
//!
//! It has bitten four times. The tensor key omitted the descriptor it embeds;
//! the level-0 LLM framing was in no fingerprint; `DEFAULT_CLUSTER_PROMPT` hid
//! behind `artifact_fingerprint`; the segmenter was missing from the descriptor
//! and tensor keys. Each was caught by an accident — a prompt comparison, a
//! code reading, a prose corpus that happened to get indexed twice — and none
//! by a standing check. `"structural:v1"` and `"content:v1"` are still
//! literals with no interpolated dependency on the code whose behaviour they
//! claim to identify.
//!
//! # Why a plain golden file would not close it
//!
//! Edit `tokenize`, the output digest moves, the test fails, regenerate the
//! golden, and `"structural:v1"` ships unbumped. The regeneration step launders
//! exactly the bug being hunted. So the assertion is **keyed on the identity**
//! rather than merely stored beside it, and there are only two outcomes:
//!
//! - **identity present, digest differs** — the failure. It fails whether or
//!   not `MAP_LEDGER_APPEND` is set; there is no supported way to rewrite an
//!   existing digest.
//! - **identity absent** — a new key. `MAP_LEDGER_APPEND=1` adds the line.
//!
//! So a deliberate behaviour change is an *appended* line and the bug is a
//! *modified* one, which a reviewer can tell apart at a glance.
//!
//! The append path writes and then still fails, telling you to re-run: a run
//! that produced the ledger never reports success on it.
//!
//! # What is not covered
//!
//! Tier B ([`map_embed`](https://docs.rs/map-embed)'s distilled embedder)
//! needs ~129 MB of weights that CI does not have, so an entry would guard less
//! than it appears to. Tier C's *reply* is nondeterministic by definition — the
//! LLM entries hash the **request**, which is pure and offline and is where all
//! three known LLM fingerprint bugs lived.
//!
//! The level-0 merkle identity was derived twice, in [`crate::fabricate`] and
//! in `map-query`, byte-identically and with no shared helper. Both now call
//! [`map_format::ObjectKey::segment`], so drift is impossible rather than
//! merely detected. `a_cluster_resolves_to_the_spans_the_indexer_keyed` still
//! stands as the end-to-end check, and it needs an embedder to build a tree, so
//! it is `distilled`-gated and **a default build detects nothing here**.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use map_core::{
    CandidateScope, Classifier, ClassifyBatch, Content, DescriptorPayload, Discoverer,
    IndexedRecord, Preprocessor, QueryBundle, QueryField, Resource, ScorerBuilder, Segment,
    Segmenter, SegmentsPayload, Stage,
};
use map_format::codec::canonical_json;
use map_format::{Config, ContentHash, Fingerprint, Manifest, ObjectStore, Record};
use map_stages::{
    ContentClassifier, DeclarationClassifier, FsDiscoverer, PackBuilder, StructuralClassifier,
    TextPreprocessor, WindowSegmenter,
};

/// Generation of the fixture below. Part of every ledger key, so evolving the
/// fixture appends a generation rather than colliding with the old one — and
/// the old digests stay readable as history.
const FIXTURE: &str = "v1";

/// Set to append missing identities to the ledger. Never overwrites one.
const APPEND_VAR: &str = "MAP_LEDGER_APPEND";

// ---------------------------------------------------------------------------
// The ledger file
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct Entry {
    stage: String,
    fixture: String,
    identity: String,
    digest: String,
}

type Key = (String, String, String);

fn key(entry: &Entry) -> Key {
    (
        entry.stage.clone(),
        entry.fixture.clone(),
        entry.identity.clone(),
    )
}

fn ledger_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("ledger.jsonl")
}

fn read_ledger() -> (String, Vec<Entry>) {
    let path = ledger_path();
    let text = std::fs::read_to_string(&path).unwrap_or_default();
    let entries = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("{} is not valid ledger JSONL: {e}", path.display()))
        })
        .collect();
    (text, entries)
}

/// Collects one `(identity, output)` pair per stage.
#[derive(Default)]
struct Observed(Vec<Entry>);

impl Observed {
    fn record(&mut self, stage: &str, identity: String, output: &[u8]) {
        self.0.push(Entry {
            stage: stage.to_owned(),
            fixture: FIXTURE.to_owned(),
            identity,
            digest: ContentHash::of(output).0.to_hex(),
        });
    }
}

/// Split observations against the ledger. Pure, so the rule it encodes is
/// testable without touching the file.
fn compare(stored: &[Entry], seen: &[Entry]) -> (Vec<(Entry, String)>, Vec<Entry>) {
    let by_key: BTreeMap<Key, &str> = stored.iter().map(|e| (key(e), e.digest.as_str())).collect();

    let mut moved = Vec::new();
    let mut missing = Vec::new();
    for entry in seen {
        match by_key.get(&key(entry)) {
            Some(digest) if *digest == entry.digest => {}
            Some(digest) => moved.push((entry.clone(), (*digest).to_owned())),
            None => missing.push(entry.clone()),
        }
    }
    (moved, missing)
}

/// What to do about a comparison. `Ok(Some(_))` is the set to append.
///
/// A moved digest loses to `append` deliberately: the whole point is that the
/// bless path cannot reach an identity that already exists.
fn verdict(
    moved: &[(Entry, String)],
    missing: &[Entry],
    append: bool,
) -> Result<Option<Vec<Entry>>, String> {
    if !moved.is_empty() {
        let mut report = format!(
            "{} stage output(s) moved under an identity that did not:\n\n",
            moved.len()
        );
        for (entry, was) in moved {
            report.push_str(&format!(
                "  stage     {}\n  identity  {}\n  recorded  {}\n  produced  {}\n\n",
                entry.stage, entry.identity, was, entry.digest
            ));
        }
        report.push_str(
            "An identity is the promise that the bytes it keys are what the stage \
             produced. A stored object under an unchanged identity is reused forever, \
             so the fix is to bump the identity — never to edit the ledger. If the \
             output was not meant to change, that is the bug.\n",
        );
        return Err(report);
    }

    if missing.is_empty() {
        return Ok(None);
    }

    if !append {
        let mut report = format!(
            "{} identity/identities are not in the ledger:\n\n",
            missing.len()
        );
        for entry in missing {
            report.push_str(&format!(
                "  {}\n",
                serde_json::to_string(entry).expect("entry serializes")
            ));
        }
        report.push_str(&format!(
            "\nIf the behaviour change was deliberate and the identity moved with it, \
             these are new keys. Re-run with {APPEND_VAR}=1 to append them.\n"
        ));
        return Err(report);
    }

    Ok(Some(missing.to_vec()))
}

/// Append, keeping the file sorted, after checking that re-serializing history
/// does not rewrite it.
fn append(stored: Vec<Entry>, original: &str, missing: Vec<Entry>) {
    let original_lines: Vec<&str> = original
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    for (entry, line) in stored.iter().zip(&original_lines) {
        let round = serde_json::to_string(entry).expect("entry serializes");
        assert_eq!(
            &round, line,
            "re-serializing an existing ledger line changed it — the entry encoding \
             moved, which silently rewrites the context of every recorded digest"
        );
    }

    let mut all = stored;
    all.extend(missing);
    all.sort_by_key(key);
    all.dedup();

    let mut out = String::new();
    for entry in &all {
        out.push_str(&serde_json::to_string(entry).expect("entry serializes"));
        out.push('\n');
    }
    let path = ledger_path();
    std::fs::write(&path, out).unwrap_or_else(|e| panic!("writing {}: {e}", path.display()));
}

// ---------------------------------------------------------------------------
// The fixture
// ---------------------------------------------------------------------------

/// Line terminators are explicit rather than written as literal newlines in
/// this source file.
///
/// A multi-line string literal would carry whatever line endings *this file*
/// was checked out with, so a Windows clone with `core.autocrlf=true` — which
/// is what this repo is developed under — would feed the stages different bytes
/// than a Linux one and every digest below would diverge by platform.
///
/// Verified rather than argued: converting all 968 of this file's line endings
/// to CRLF and re-running the ledger leaves every digest unchanged. Line endings are one of the four normalizations
/// spec §6.1 requires the preprocessor to perform, so the fixture has to be
/// able to state its own bytes exactly — including the file that is
/// deliberately CRLF.
fn joined<S: AsRef<str>>(lines: &[S], terminator: &str) -> String {
    let mut out = String::new();
    for line in lines {
        out.push_str(line.as_ref());
        out.push_str(terminator);
    }
    out
}

/// LF, with a hyphenated word and a contraction — the two things `[alnum_]`
/// word characters split, so a tokenizer reworked for prose moves this digest.
const SESSION: &[&str] = &[
    "pub fn refresh_token(session: &mut Session) {",
    "    // Re-issue at the half-life; don't wait for expiry.",
    "    session.expiry = now() + TTL;",
    "}",
];

/// CRLF. Its normalized text must be byte-identical to the LF form, which is
/// what makes a Tier A index portable.
const CRLF: &[&str] = &[
    "pub fn configure_logger(level: Level) {",
    "    set_global_level(level);",
    "}",
];

/// A BOM and non-ASCII identifiers.
///
/// Measured, not assumed: the preprocessor leaves the BOM in place (78 bytes on
/// disk, 78 after normalization), so it occupies the first three bytes of every
/// offset in this file. Spec §6.1 does not ask for it to be stripped — it names
/// line endings, path separators and case, *filename* NFC, and directory order.
/// Pinning it here means a future decision to strip it cannot be made silently.
const UNICODE: &[&str] = &[
    "\u{feff}pub fn mesure_café(entrée: &Naïve) -> Größe {",
    "    entrée.größe()",
    "}",
];

/// Long enough to exceed one window, so the overlap arithmetic is exercised.
///
/// Generated rather than written out: the only property it is chosen to
/// preserve is its length in lines. 60 lines against the default 40-line
/// window with 8 lines of overlap cuts into two segments — measured, `0..1950`
/// and `1558..2930`, so the second starts inside the first. A one-window
/// fixture cannot reach the overlap arithmetic at all.
fn long_lines() -> Vec<String> {
    (0..60)
        .map(|i| format!("pub fn step_{i:02}(value: u32) -> u32 {{ value + {i} }}"))
        .collect()
}

fn fixture_files() -> Vec<(&'static str, String)> {
    vec![
        ("src/session.rs", joined(SESSION, "\n")),
        ("src/logging.rs", joined(CRLF, "\r\n")),
        ("src/unicode.rs", joined(UNICODE, "\n")),
        ("src/steps.rs", joined(&long_lines(), "\n")),
    ]
}

fn write_fixture(root: &Path) {
    std::fs::create_dir_all(root.join("src")).unwrap();
    for (name, text) in fixture_files() {
        std::fs::write(root.join(name), text.as_bytes()).unwrap();
    }
}

/// Length-prefixed so concatenation is unambiguous — the same reason
/// [`map_format::ObjectKey::derive`] frames its inputs.
fn framed(parts: &BTreeMap<&str, Vec<u8>>) -> Vec<u8> {
    let mut out = Vec::new();
    for (name, bytes) in parts {
        out.extend_from_slice(&(name.len() as u64).to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(bytes);
    }
    out
}

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        // Process id plus a counter keeps parallel tests apart without a `rand`
        // dependency, matching the guards elsewhere in this crate.
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let mut root = std::env::temp_dir();
        root.push(format!(
            "map-ledger-{tag}-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        TempDir(root)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// ---------------------------------------------------------------------------
// Observations — one per stage, identity paired with what it produced
// ---------------------------------------------------------------------------

fn observe_stages(seen: &mut Observed) {
    let dir = TempDir::new("stages");
    write_fixture(dir.path());

    let discoverer = FsDiscoverer::default();
    let resources: Vec<Resource> = discoverer.discover(dir.path()).unwrap();
    // A digest of nothing is perfectly stable, so an empty or shrunken fixture
    // would record a green entry forever. Every observation below is guarded
    // the same way: the ledger may only pin output that exists.
    assert_eq!(
        resources.len(),
        fixture_files().len(),
        "the fixture did not survive discovery: {resources:?}"
    );
    seen.record(
        "discover",
        discoverer.config(),
        &canonical_json(&resources).unwrap(),
    );

    let preprocessor = TextPreprocessor;
    let mut decoded: Vec<(String, Option<String>)> = Vec::new();
    let mut contents: Vec<Content> = Vec::new();
    for resource in &resources {
        let bytes = std::fs::read(dir.path().join(&resource.key)).unwrap();
        let content = preprocessor.preprocess(resource, &bytes).unwrap();
        decoded.push((
            resource.key.clone(),
            content.as_ref().map(|c| c.text.clone()),
        ));
        if let Some(content) = content {
            contents.push(content);
        }
    }
    assert_eq!(
        contents.len(),
        resources.len(),
        "a fixture file failed to decode, so its stages are unmeasured"
    );
    assert!(
        contents.iter().all(|c| !c.text.contains('\r')),
        "the CRLF fixture must normalize to LF, or Tier A is not portable"
    );
    seen.record(
        "preprocess",
        preprocessor.config(),
        &canonical_json(&decoded).unwrap(),
    );

    // From here down each stage is fed input this module owns rather than the
    // stage above it. Otherwise the entries are input-coupled: changing the
    // segmenter would move `structural:v1`'s digest under an unchanged
    // identity, which reads as the bug and is not one — the real descriptor key
    // already folds in `segmenter_fp`, so stored descriptors are correctly
    // re-keyed. Cross-stage coupling belongs to the composite entries below,
    // whose identities name their upstream.
    let segmenter = WindowSegmenter::default();
    let mut spans: BTreeMap<String, SegmentsPayload> = BTreeMap::new();
    for (key, text) in segmenter_inputs() {
        let content = Content {
            key: key.to_owned(),
            text,
        };
        spans.insert(
            key.to_owned(),
            SegmentsPayload {
                segments: segmenter.segment(&content).unwrap(),
            },
        );
    }
    assert!(
        spans["long"].segments.len() > 1,
        "the long input must exceed one window, or overlap is unmeasured"
    );
    seen.record(
        "segment",
        segmenter.config(),
        &canonical_json(&spans).unwrap(),
    );

    let items: Vec<Vec<&str>> = classifier_items().iter().map(|t| vec![*t]).collect();

    let lexical = vec!["lexical".to_owned()];
    let structural_out = DescriptorPayload {
        dimensions: lexical.clone(),
        per_segment: StructuralClassifier
            .classify(&ClassifyBatch {
                items: &items,
                dimensions: &lexical,
                level: 0,
            })
            .unwrap(),
    };
    assert_eq!(
        structural_out.per_segment.len(),
        items.len(),
        "a dropped item would silently shrink what this entry pins"
    );
    assert!(
        structural_out.per_segment[0]["lexical"]
            .descriptor
            .as_deref()
            .is_some_and(|d| d.contains("refresh_token") && d.contains("refresh")),
        "the compound and its sub-terms are the tokenizer behaviour this pins; \
         an empty or truncated descriptor would pin nothing"
    );
    seen.record(
        "structural",
        StructuralClassifier.config(),
        &canonical_json(&structural_out).unwrap(),
    );

    let semantic = vec!["semantic".to_owned()];
    let content_out = DescriptorPayload {
        dimensions: semantic.clone(),
        per_segment: ContentClassifier
            .classify(&ClassifyBatch {
                items: &items,
                dimensions: &semantic,
                level: 0,
            })
            .unwrap(),
    };
    seen.record(
        "content",
        ContentClassifier.config(),
        &canonical_json(&content_out).unwrap(),
    );

    let declaration = vec!["declaration".to_owned()];
    let declaration_out = DescriptorPayload {
        dimensions: declaration.clone(),
        per_segment: DeclarationClassifier
            .classify(&ClassifyBatch {
                items: &items,
                dimensions: &declaration,
                level: 0,
            })
            .unwrap(),
    };
    assert!(
        declaration_out.per_segment[0]["declaration"]
            .descriptor
            .as_deref()
            .is_some_and(|d| d.contains("refresh_token")),
        "the `fn` line must yield its name, or this entry pins nothing"
    );
    assert!(
        !declaration_out.per_segment[2].contains_key("declaration"),
        "prose declares nothing and must produce no record"
    );
    seen.record(
        "declaration",
        DeclarationClassifier.config(),
        &canonical_json(&declaration_out).unwrap(),
    );

    observe_pack(seen, &lexical);
}

/// Content for the segmenter, built here rather than taken from the
/// preprocessor. Already LF, because that is what the preprocessor guarantees
/// and what the segmenter would therefore always see.
fn segmenter_inputs() -> Vec<(&'static str, String)> {
    vec![
        ("empty", String::new()),
        ("short", joined(SESSION, "\n")),
        ("long", joined(&long_lines(), "\n")),
        // No trailing newline: the last window's end offset is the one place
        // an off-by-one in the line walk shows up.
        ("unterminated", joined(SESSION, "\n").trim_end().to_owned()),
    ]
}

/// Texts for the classifiers, chosen for what the tokenizer does to them:
/// a snake_case and a camelCase compound that split into sub-terms, a
/// hyphenated word and a contraction that `[alnum_]` splits, non-ASCII letters,
/// a repeated term for the frequency encoding, and an empty item.
fn classifier_items() -> Vec<&'static str> {
    vec![
        "pub fn refresh_token(session: &mut Session) -> Session",
        "const refreshToken = wellKnown.token;",
        "Re-issue at the half-life; don't wait for expiry.",
        "mesure_caf\u{e9}(entr\u{e9}e: &Na\u{ef}ve) -> Gr\u{f6}\u{df}e",
        "alpha alpha alpha beta",
        "",
    ]
}

/// The pack bytes **and** a fixed query's scores.
///
/// `K1` and `B` are interpolated into the identity, so a tuning change is
/// already visible. The scoring formula around them is not, and it does not
/// reach the pack bytes either — a rewritten `idf` would leave the encoded
/// postings byte-identical. Only the scores catch it.
///
/// Fed literal descriptors rather than the fixture's tokenizer output, which
/// this entry originally did and which was wrong: a tokenizer edit then moved
/// this digest under an unchanged `bm25:` identity, demanding a bump that would
/// have been a lie. The pack is keyed by the manifest fingerprint in real use,
/// which already folds in whatever produced its input. What belongs here is the
/// pack's own behaviour, so its own input is what it gets.
///
/// The four records are shaped to reach the parts of BM25 that a single record
/// cannot: differing lengths for the `b` length-normalization, a term in three
/// of them and a term in one for the `idf` spread, a repeated term for `k1`
/// saturation, and a descriptor-less slot because `push` must still allocate an
/// id for one.
fn observe_pack(seen: &mut Observed, dimensions: &[String]) {
    fn record(descriptor: Option<&str>) -> Record {
        Record {
            descriptor: descriptor.map(str::to_owned),
            tensor: None,
            meta: map_format::RecordMeta {
                kind: map_format::RecordKind::Segment,
                dimension: "lexical".to_owned(),
                level: 0,
                children: Vec::new(),
            },
        }
    }

    let corpus_records = [
        ("src/a.rs", record(Some("alpha:3 beta gamma"))),
        ("src/b.rs", record(Some("alpha beta:2 delta epsilon:4"))),
        ("src/c.rs", record(Some("gamma zeta"))),
        ("src/d.rs", record(None)),
    ];

    fn fill(builder: &mut PackBuilder, entries: &[(&str, Record)]) {
        for (id, (resource, record)) in entries.iter().enumerate() {
            builder.push(&IndexedRecord {
                id: id as u32,
                resource,
                segment: Segment {
                    start: 0,
                    end: 100 + id as u32,
                },
                record,
            });
        }
    }

    let mut builder = PackBuilder::new();
    fill(&mut builder, &corpus_records);
    let pack_bytes = builder.finish();

    let mut builder = PackBuilder::new();
    fill(&mut builder, &corpus_records);
    let identity = builder.config();
    let scorer = Box::new(builder).build().unwrap();

    let query: QueryBundle = [("lexical".to_owned(), QueryField::text("alpha gamma"))]
        .into_iter()
        .collect();
    let corpus = scorer.corpus_stats(&query, dimensions);
    let mut scored = scorer
        .score(CandidateScope::All, &query, dimensions, corpus.as_ref())
        .unwrap();
    assert!(
        scored.iter().any(|(_, s)| s.values().any(|v| *v > 0.0)),
        "a query that scores nothing would pin no arithmetic at all"
    );
    if std::env::var("MAP_LEDGER_SHOW").is_ok() {
        eprintln!("bm25 corpus: {corpus:?}\nbm25 scores: {scored:?}");
    }
    // Sorted because the ledger asks what the scores *are*, not what order they
    // came back in; `ranking_is_deterministic` already covers the ordering.
    scored.sort_by_key(|(id, _)| *id);

    let mut scores = Vec::new();
    for (id, per_dimension) in &scored {
        scores.extend_from_slice(&id.to_le_bytes());
        for (dimension, score) in per_dimension {
            scores.extend_from_slice(dimension.as_bytes());
            scores.extend_from_slice(&score.to_le_bytes());
        }
    }

    seen.record(
        "bm25-pack",
        identity,
        &framed(&BTreeMap::from([
            ("1-pack", pack_bytes),
            ("2-scores", scores),
        ])),
    );
}

/// The composite keys — the class that has actually bitten.
///
/// Identity is the fingerprint-input string the driver builds; the digest is
/// the bytes it stored under the key that string derives. Hashing the key
/// itself would be circular: the key *is* a function of the identity, so it
/// could never disagree with it.
fn observe_driver(seen: &mut Observed) {
    let dir = TempDir::new("driver");
    write_fixture(dir.path());

    let map = dir.path().join(".map");
    for sub in ["index/shared/objects", "index/desc/objects", "cache"] {
        std::fs::create_dir_all(map.join(sub)).unwrap();
    }
    let config = Config::zero_config();
    std::fs::write(map.join("config.toml"), config.to_toml().unwrap()).unwrap();
    let manifest = Manifest::new(&config, "ledger", 0).unwrap();
    std::fs::write(map.join("manifest.json"), manifest.to_bytes().unwrap()).unwrap();

    crate::run(dir.path()).unwrap();

    let manifest =
        Manifest::from_bytes(&std::fs::read(map.join("manifest.json")).unwrap()).unwrap();
    let shared = ObjectStore::open(map.join("index/shared/objects"));
    let descriptors = ObjectStore::open(map.join("index/desc/objects"));

    let mut segment_objects: BTreeMap<&str, Vec<u8>> = BTreeMap::new();
    let mut descriptor_objects: BTreeMap<&str, Vec<u8>> = BTreeMap::new();
    for (resource, root) in &manifest.roots {
        segment_objects.insert(resource, shared.get(root.segments).unwrap());
        if let Some(key) = root.descriptors.get("structural") {
            descriptor_objects.insert(resource, descriptors.get(*key).unwrap());
        }
    }
    assert!(
        !descriptor_objects.is_empty(),
        "the driver stored no descriptor objects, so this observation proves nothing"
    );

    // Resolved from the config the run above actually used, not from
    // `WindowSegmenter::default()`. The two agree today, and an entry that
    // described a run it did not perform would be worse than no entry.
    let segmenter = crate::segmenter_for(&config);
    let segmenter_fp = Fingerprint::of(segmenter.config().as_bytes());

    seen.record(
        "segments-object",
        segmenter.config(),
        &framed(&segment_objects),
    );
    seen.record(
        "descriptor-object",
        crate::descriptor_fingerprint_input(
            &StructuralClassifier.config(),
            &["lexical".to_owned()],
            segmenter_fp,
        ),
        &framed(&descriptor_objects),
    );
}

/// The LLM request, not the reply.
///
/// The reply is Tier C and has no reproducible bytes; the request is pure and
/// offline, and all three known LLM fingerprint bugs — the level-0 framing, the
/// cluster framing, the fallback cluster prompt — were bugs in *the request*
/// being invisible to *the identity*. The adapter points at a closed port: it
/// is never called, and `config()` reads only the model name from it.
#[cfg(feature = "llm")]
fn observe_llm(seen: &mut Observed) {
    use map_stages::chat::{
        build_messages, build_schema, parse_response, Facets, LlmClassifier, PromptSet,
    };

    const PROMPT: &str = "Describe what this does.";
    const PROSE_REPLY: &str = r#"{"segments":[{"descriptive":"first"},{"descriptive":"second"}]}"#;
    const PARTS_REPLY: &str = r#"{"segments":[{"descriptive":{"description":"first","identifiers":["alpha"]}},{"descriptive":{"description":"second","identifiers":["beta"]}}]}"#;

    fn adapter() -> std::sync::Arc<map_llm::LlmAdapter> {
        std::sync::Arc::new(map_llm::LlmAdapter::new(map_llm::Connection {
            endpoint: "http://127.0.0.1:1/v1".into(),
            model: "ledger-model".into(),
            api_key: None,
            protocol: map_llm::Protocol::OpenAiChat,
        }))
    }

    let dimensions = vec!["descriptive".to_owned()];
    let items = ["alpha beta", "gamma delta"];

    let prose_facets = Facets::Prose;
    let parts_facets = Facets::parse(&["description".to_owned(), "identifiers[8]".to_owned()]);

    let prose = LlmClassifier::new(adapter(), PromptSet::uniform(PROMPT), dimensions.clone());
    let parts = LlmClassifier::with_facets(
        adapter(),
        PromptSet::uniform(PROMPT),
        dimensions.clone(),
        parts_facets.clone(),
    );

    // The shipped shape is the four-part one, and a cluster's identity is
    // separate from a segment's — both framings need an entry, or the half that
    // has already been wrong twice is the half left uncovered.
    let cases: [(&str, u16, String, &Facets, &str); 4] = [
        (
            "llm-request-prose",
            0,
            prose.config(),
            &prose_facets,
            PROSE_REPLY,
        ),
        (
            "llm-request-prose-cluster",
            1,
            prose.fabric_config(),
            &prose_facets,
            PROSE_REPLY,
        ),
        (
            "llm-request-parts",
            0,
            parts.config(),
            &parts_facets,
            PARTS_REPLY,
        ),
        (
            "llm-request-parts-cluster",
            1,
            parts.fabric_config(),
            &parts_facets,
            PARTS_REPLY,
        ),
    ];

    for (stage, level, identity, facets, reply) in cases {
        let (system, user) = build_messages(PROMPT, &dimensions, &items, level, facets);
        let schema = build_schema(&dimensions, facets);
        let per_segment = parse_response(reply, &dimensions, items.len(), level, facets).unwrap();
        let records = DescriptorPayload {
            dimensions: dimensions.clone(),
            per_segment,
        };
        seen.record(
            stage,
            identity,
            &framed(&BTreeMap::from([
                ("1-system", system.into_bytes()),
                ("2-user", user.into_bytes()),
                ("3-schema", serde_json::to_vec(&schema).unwrap()),
                ("4-records", canonical_json(&records).unwrap()),
            ])),
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn a_stage_output_may_not_move_under_an_identity_that_did_not() {
    let mut seen = Observed::default();
    observe_stages(&mut seen);
    observe_driver(&mut seen);
    #[cfg(feature = "llm")]
    observe_llm(&mut seen);

    let (original, stored) = read_ledger();
    let (moved, missing) = compare(&stored, &seen.0);
    let append_requested = std::env::var(APPEND_VAR).is_ok_and(|v| v != "0" && !v.is_empty());

    match verdict(&moved, &missing, append_requested) {
        Ok(None) => {}
        Ok(Some(to_add)) => {
            let count = to_add.len();
            append(stored, &original, to_add);
            panic!(
                "appended {count} identity/identities to {}. Re-run without \
                 {APPEND_VAR} — a run that wrote the ledger does not get to \
                 report success on it.",
                ledger_path().display()
            );
        }
        Err(report) => panic!("{report}"),
    }
}

/// The fixture corpus with one dimension that builds a tree over it.
#[cfg(feature = "distilled")]
fn write_tree_fixture(root: &Path) {
    write_fixture(root);

    let map = root.join(".map");
    for sub in ["index/shared/objects", "index/desc/objects", "cache"] {
        std::fs::create_dir_all(map.join(sub)).unwrap();
    }
    let config = Config::zero_config();
    std::fs::write(map.join("config.toml"), config.to_toml().unwrap()).unwrap();
    let manifest = Manifest::new(&config, "ledger", 0).unwrap();
    std::fs::write(map.join("manifest.json"), manifest.to_bytes().unwrap()).unwrap();

    let mut toml = std::fs::read_to_string(map.join("config.toml")).unwrap();
    toml.push_str(
        "
[dimensions.tree]
         description = \"a tree to resolve members through\"
         classifier = { impl = \"structural\" }
         embedder = { impl = \"distilled\" }
         fabricator = { impl = \"agglomerative\", threshold = 0.1, min_cluster = 2, max_levels = 1, min_remaining = 1 }
",
    );
    std::fs::write(map.join("config.toml"), toml).unwrap();
}

/// The level-0 ids the fabricator would cluster, in manifest order.
#[cfg(feature = "distilled")]
fn level0_ids(root: &Path) -> Vec<map_format::ObjectKey> {
    let map_dir = root.join(".map");
    let config =
        Config::parse(&std::fs::read_to_string(map_dir.join("config.toml")).unwrap()).unwrap();
    let manifest =
        Manifest::from_bytes(&std::fs::read(map_dir.join("manifest.json")).unwrap()).unwrap();

    crate::fabricate::gather_level0(
        &manifest,
        &ObjectStore::open(map_dir.join("index/shared/objects")),
        &ObjectStore::open(map_dir.join("index/desc/objects")),
        &ObjectStore::open(map_dir.join("index/tensor/objects")),
        "tree",
        Some("structural"),
        "distilled",
        config.dimensions["tree"].artifact_fingerprint().unwrap(),
    )
    .unwrap()
    .into_iter()
    .map(|(identity, _, _)| identity)
    .collect()
}

/// How many segments one resource was cut into, as the index records it.
#[cfg(feature = "distilled")]
fn recorded_segments(root: &Path, resource: &str) -> usize {
    let map_dir = root.join(".map");
    let manifest =
        Manifest::from_bytes(&std::fs::read(map_dir.join("manifest.json")).unwrap()).unwrap();
    let shared = ObjectStore::open(map_dir.join("index/shared/objects"));
    let payload: SegmentsPayload =
        serde_json::from_slice(&shared.get(manifest.roots[resource].segments).unwrap()).unwrap();
    payload.segments.len()
}

/// The cluster keys the index recorded for the `tree` dimension.
#[cfg(feature = "distilled")]
fn cluster_keys(root: &Path) -> Vec<map_format::ObjectKey> {
    let manifest =
        Manifest::from_bytes(&std::fs::read(root.join(".map/manifest.json")).unwrap()).unwrap();
    manifest.clusters.get("tree").cloned().unwrap_or_default()
}

/// Needs the potion weights on disk; CI does not run it.
#[test]
#[cfg(feature = "distilled")]
fn an_edit_that_moves_no_span_still_re_keys_that_file_s_leaves() {
    // Leaf identity used to be `(resource, start, end)` under the dimension
    // fingerprint, which says nothing about content. An edit that preserves
    // every line length therefore left every leaf id — and so every cluster key
    // above them — untouched, and the fabricator reused labels describing text
    // that was no longer there. Spec §3.1 promises the reverse: it is the
    // *unchanged* subtree that keeps its identity.
    let dir = TempDir::new("content-sensitive-leaves");
    write_tree_fixture(dir.path());

    let stats = crate::run(dir.path()).unwrap();
    assert!(
        stats.clusters_written > 0,
        "no tree was built, so nothing here is tested: {stats:?}"
    );
    let before = level0_ids(dir.path());
    let clusters_before = cluster_keys(dir.path());
    assert!(!before.is_empty() && !clusters_before.is_empty());

    // Same bytes in the same places: every span start and end survives.
    let path = dir.path().join("src/steps.rs");
    let original = std::fs::read_to_string(&path).unwrap();
    let edited = original.replace("value + ", "value * ");
    assert_eq!(
        edited.len(),
        original.len(),
        "the edit must preserve length"
    );
    assert_ne!(edited, original, "the edit must change something");
    std::fs::write(&path, &edited).unwrap();

    crate::run(dir.path()).unwrap();
    let after = level0_ids(dir.path());
    let edited_segments = recorded_segments(dir.path(), "src/steps.rs");

    assert!(edited_segments > 0);
    assert_eq!(before.len(), after.len(), "the segmentation is unchanged");
    let survived = before.iter().filter(|id| after.contains(id)).count();
    assert_eq!(
        survived,
        before.len() - edited_segments,
        "exactly the edited file's leaves must be re-keyed — every other file's \
         must survive, or an unrelated edit would rebuild the whole tree"
    );
    assert_ne!(
        clusters_before,
        cluster_keys(dir.path()),
        "a cluster over a re-keyed leaf must be re-keyed rather than reused"
    );
}

/// The indexer and the retriever must derive the same level-0 merkle ids.
///
/// Both now call [`map_format::ObjectKey::segment`] — `map-index`'s
/// `fabricate::gather_level0` and `map-query`'s cluster resolution — which is
/// what makes them agree by construction rather than by two copies staying in
/// step. `Index::cluster_members` resolves a cluster's leaf ids and
/// `filter_map`s misses away, so a drift returns `Some(vec![])` — a cluster
/// that stands for nothing, reported as success. That silence is the reason
/// this asserts non-empty rather than merely `is_some`.
///
/// Needs the potion weights on disk: a tree needs an embedder, so there is no
/// default-feature path to this check. CI does not run it.
#[test]
#[cfg(feature = "distilled")]
fn a_cluster_resolves_to_the_spans_the_indexer_keyed() {
    let dir = TempDir::new("merkle");
    write_tree_fixture(dir.path());

    let stats = crate::run(dir.path()).unwrap();
    assert!(
        stats.clusters_written > 0,
        "no tree was built, so nothing here is tested: {stats:?}"
    );

    let index = map_query::Index::open(dir.path()).unwrap();
    let query: map_query::Query = [(
        "tree".to_owned(),
        map_query::QueryTerm::new("refresh token session"),
    )]
    .into_iter()
    .collect();
    let hits = index
        .find_at(&query, 5, map_query::LevelFilter::Clusters)
        .unwrap();
    assert!(!hits.is_empty(), "no cluster was retrievable: {stats:?}");

    let members = index
        .cluster_members(&hits[0])
        .expect("a cluster hit resolves");
    assert!(
        !members.is_empty(),
        "a cluster resolved to no spans — the indexer and the retriever derive          level-0 merkle ids in two places and they have drifted apart"
    );

    let known: std::collections::BTreeSet<&str> =
        fixture_files().iter().map(|(name, _)| *name).collect();
    for member in &members {
        assert!(
            known.contains(member.resource.as_str()),
            "member {member:?} names no fixture resource"
        );
    }
}

/// Tests of the check itself. A ledger that has never been seen to fail is not
/// evidence, and every determinism test in this workspace compares a run
/// against itself — which cannot fail for this bug by construction.
mod harness {
    use super::*;

    fn entry(stage: &str, identity: &str, digest: &str) -> Entry {
        Entry {
            stage: stage.to_owned(),
            fixture: FIXTURE.to_owned(),
            identity: identity.to_owned(),
            digest: digest.to_owned(),
        }
    }

    #[test]
    fn a_changed_output_under_an_unchanged_identity_is_rejected() {
        // The whole point of the ledger. Without this the check above could be
        // silently unable to fail, which is the state the project was already
        // in — every determinism test compares a run against itself.
        let stored = vec![entry("structural", "structural:v1", "aaaa")];
        let seen = vec![entry("structural", "structural:v1", "bbbb")];

        let (moved, missing) = compare(&stored, &seen);
        assert_eq!(moved.len(), 1, "a moved digest must be detected");
        assert!(missing.is_empty());

        let report = verdict(&moved, &missing, false).unwrap_err();
        assert!(report.contains("structural:v1"), "{report}");
        assert!(report.contains("bump the identity"), "{report}");
    }

    #[test]
    fn appending_may_not_overwrite_an_existing_identity() {
        // The bless path is the obvious way to launder the bug: regenerate,
        // commit, ship. It has to lose to a moved digest.
        let stored = vec![entry("structural", "structural:v1", "aaaa")];
        let seen = vec![entry("structural", "structural:v1", "bbbb")];

        let (moved, missing) = compare(&stored, &seen);
        assert!(
            verdict(&moved, &missing, true).is_err(),
            "{APPEND_VAR} must not reach an identity that already exists"
        );
    }

    #[test]
    fn a_bumped_identity_is_a_new_key_rather_than_an_edit() {
        // The remedy the failure message names has to actually work, or the
        // check is a wall with no door.
        let stored = vec![entry("structural", "structural:v1", "aaaa")];
        let seen = vec![entry("structural", "structural:v2", "bbbb")];

        let (moved, missing) = compare(&stored, &seen);
        assert!(moved.is_empty(), "a bumped identity is not a moved digest");
        assert_eq!(missing.len(), 1);

        let to_add = verdict(&moved, &missing, true).unwrap().unwrap();
        assert_eq!(to_add, seen, "the new key is what gets appended");
        assert!(
            verdict(&moved, &missing, false).is_err(),
            "without the env var, a new key still fails rather than passing quietly"
        );
    }

    #[test]
    fn an_unchanged_run_is_accepted() {
        let stored = vec![entry("structural", "structural:v1", "aaaa")];
        let (moved, missing) = compare(&stored, &stored);
        assert!(verdict(&moved, &missing, false).unwrap().is_none());
    }

    #[test]
    fn an_entry_recorded_under_another_feature_is_not_treated_as_missing() {
        // A default build never observes the `llm` entries. They are history,
        // not a failure — otherwise the ledger could not hold both.
        let stored = vec![
            entry("structural", "structural:v1", "aaaa"),
            entry("llm-request-prose", "chat-completions:model=m", "cccc"),
        ];
        let seen = vec![entry("structural", "structural:v1", "aaaa")];

        let (moved, missing) = compare(&stored, &seen);
        assert!(moved.is_empty());
        assert!(missing.is_empty());
    }

    /// Prints what the ledger digests summarize.
    ///
    /// Ignored by default because it asserts nothing — it exists because a
    /// digest is two layers of indirection away from the behaviour it pins, and
    /// the raw descriptors are what say whether the fixture reaches the
    /// tokenizer at all. Run it before trusting a new entry:
    /// `cargo test -p map-index --lib inspect_the_fixture -- --ignored --nocapture`
    #[test]
    #[ignore = "prints for inspection; asserts nothing"]
    fn inspect_the_fixture() {
        let dir = TempDir::new("dump");
        write_fixture(dir.path());
        let resources = FsDiscoverer::default().discover(dir.path()).unwrap();
        eprintln!("-- discover + preprocess --");
        for resource in &resources {
            let bytes = std::fs::read(dir.path().join(&resource.key)).unwrap();
            let content = TextPreprocessor.preprocess(resource, &bytes).unwrap();
            let text = content.as_ref().map(|c| c.text.as_str()).unwrap_or("");
            eprintln!(
                "{:16} raw={:5} norm={:5} cr={} bom={}",
                resource.key,
                bytes.len(),
                text.len(),
                text.matches('\r').count(),
                text.starts_with('\u{feff}'),
            );
        }

        eprintln!("\n-- segment --");
        for (key, text) in segmenter_inputs() {
            let content = Content {
                key: key.to_owned(),
                text,
            };
            eprintln!(
                "{key:16} {:?}",
                WindowSegmenter::default().segment(&content).unwrap()
            );
        }

        eprintln!("\n-- structural --");
        let items: Vec<Vec<&str>> = classifier_items().iter().map(|t| vec![*t]).collect();
        let out = StructuralClassifier
            .classify(&ClassifyBatch {
                items: &items,
                dimensions: &["lexical".to_owned()],
                level: 0,
            })
            .unwrap();
        for (item, records) in classifier_items().iter().zip(&out) {
            eprintln!(
                "{item:?}\n  -> {:?}",
                records.get("lexical").and_then(|r| r.descriptor.as_deref())
            );
        }
    }

    #[test]
    fn the_fixture_states_its_own_line_endings() {
        // If this file were checked out with CRLF, a multi-line literal would
        // carry it and every digest would be platform-dependent. The fixture
        // builds its terminators explicitly; this asserts the LF file has no
        // CR at all and the CRLF one has one per line.
        let files: BTreeMap<&str, String> = fixture_files().into_iter().collect();
        assert!(!files["src/session.rs"].contains('\r'));
        assert_eq!(
            files["src/logging.rs"].matches("\r\n").count(),
            CRLF.len(),
            "the CRLF fixture must actually be CRLF"
        );
        assert!(files["src/unicode.rs"].starts_with('\u{feff}'));
    }
}
