//! `map init` — create a `.map` directory.
//!
//! The generated config enables the two offline dimensions: `lexical`
//! (structural classification plus BM25) and `declaration` (the names a
//! segment declares, also scored by BM25). Neither needs an API key, network,
//! or model download, so `map init && map index && map find` succeeds on a
//! fresh machine with nothing configured. Semantic dimensions are emitted
//! commented out, one edit away.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use map_format::{Config, Manifest};

/// Committed config template.
///
/// Hand-authored rather than serialized from [`Config::zero_config`] because
/// the comments are the point — this file is the first thing a user reads.
/// A test asserts it parses to exactly [`Config::zero_config`], so the two
/// cannot drift.
const CONFIG_TEMPLATE: &str = r##"# .map/config.toml
#
# This file is committed. It is the input the index is built from, so editing
# a dimension changes its fingerprint and invalidates exactly the objects that
# dimension produced -- no more, and no less than has to be recomputed.

version = 1

# Nothing updates the index on its own. `map find` never does -- pass -u when
# you want it to, since a configured dense dimension can make an update slow
# and a query should not surprise you with one. An interactive `map find` tells
# you when the index has drifted; that notice is suppressed for non-interactive
# callers, where it would just be noise in a model's context.

[storage]
# How much derived data enters the repository, and where it lives.
#
# Only the values written below are implemented. The others are defined in the
# format but not yet honoured by the writer, so setting one is an error rather
# than a setting that quietly does nothing:
#
#   commit    binary | descriptors-only | none   not implemented
#   location  ref (an orphan ref, out of the tree)  not implemented
commit = "full"
location = "working-tree"

# ---------------------------------------------------------------------------
# Segmentation
#
# How a resource is cut up before any dimension sees it. Shared by every
# dimension on purpose: segment identity is the join key that makes a hit in
# one dimension comparable to a hit in another.
#
# Line windows are the only implementation. They are deliberately
# structure-blind -- cutting on syntax would need a grammar per format, and this
# has to run on whatever text a root holds. The cost is that a boundary falls
# where the line count lands rather than at a natural seam, which is what
# `overlap` is for.
#
# The defaults below were measured on source code. Prose, transcripts and
# exported records have different natural units, and nothing here knows which
# you have -- so treat them as a starting point, and change them by measuring
# rather than by reasoning. Editing them re-segments the corpus, which re-keys
# every object derived from it.
#
# [segmenter]
# impl    = "window"
# lines   = 40
# overlap = 8

# ---------------------------------------------------------------------------
# Dimensions
#
# A dimension is one field of the query the model fills in. Its `description`
# is what the model reads when deciding what to put there, so write it for
# that audience rather than as internal documentation.
#
# The descriptions below are written for a root of unspecified content, because
# that is all a fresh `map init` can know. If your root holds one kind of thing
# -- source code, meeting notes, support tickets, a documentation set -- say so
# here. A description is model-facing prose and is deliberately excluded from
# the artifact fingerprint, so sharpening one costs nothing and invalidates
# nothing.
#
# Every dimension must produce a descriptor (text), a tensor (numeric), or
# both. A dimension producing neither is unsearchable and is rejected. Which
# scorer runs follows from that: BM25 over the descriptor, cosine over the
# tensor. There is no `scorer` key -- it would be a setting nothing could
# contradict.
# ---------------------------------------------------------------------------

[dimensions.lexical]
description = "Exact words, names, and literals as they appear in the text. Use this for anything you would otherwise search for verbatim."
enabled = true
classifier = { impl = "structural" }

# Only the names a segment declares (`fn`, `class`, `struct`, `def`, `type`,
# ...), so a name put here ranks the declaring segment above every segment
# that merely uses it. BM25 over whole content does the opposite: callers
# mention a name more often than its one declaration does. Offline, free, and
# a line scan rather than a parser, so it runs on any text.
[dimensions.declaration]
description = "The exact name of the thing whose declaration you want -- a function, type, class, module, or constant. A name here ranks the place that declares it above the places that use it."
enabled = true
classifier = { impl = "declaration" }

# ---------------------------------------------------------------------------
# Embedding dimensions
#
# A dimension is named for what the caller puts in that query field, not for the
# technique behind it: `lexical` takes keywords, `semantic` takes content
# resembling your target, `descriptive` takes a description of it.
#
# Nothing below is required. The lexical dimension above works offline, with
# no key and no model download.
#
# `distilled` needs the model in ~/.map/models and a build with
# --features distilled; build with --features auto-distilled instead and
# `map model fetch` downloads it. `llm` needs an OpenAI-compatible endpoint
# and a build with --features llm; run `map llm login` to set it.
# The connection lives in ~/.map/llm.toml, never here -- this file is committed
# and a secret must not appear in it.
# ---------------------------------------------------------------------------

# Free and offline: static distilled embeddings of the content itself, no LLM
# and no key. Matches meaning rather than exact terms.
#
# `content` shapes the segment for the embedder and stores nothing --
# persist_output = false, because the descriptor would be a second copy of the
# corpus to save a string join.
# [dimensions.semantic]
# description = "Content resembling what you are looking for -- paste or paraphrase the material itself."
# classifier = { impl = "content", persist_output = false }
# embedder   = { impl = "distilled" }

# An LLM writes a prose descriptor of each segment, the distilled embedder
# embeds that prose, and the fabricator clusters it into a labeled tree.
#
# The prompt is where a corpus says what it holds. The built-in framing around
# it says only how many segments there are and what shape to answer in -- it
# names no subject matter, so an unedited prompt indexes prose and source code
# on the same terms. Both the prompt and that framing are fingerprinted:
# changing either re-classifies rather than reusing answers to a question no
# longer being asked.
# [dimensions.descriptive]
# description = "A description of what you are looking for, in your own words -- what it does, what it is about, what happens when it runs."
# classifier = { impl = "llm", prompts = { "0" = "Describe what this is and what it does, in one concise sentence of plain language.", "1" = "Name the theme these descriptions share, as a short noun phrase." } }
# embedder   = { impl = "distilled" }
# fabricator = { impl = "agglomerative", threshold = 0.70, min_cluster = 5, max_cluster = 15 }
"##;

/// `.map/.gitignore` — everything derived and disposable.
const GITIGNORE: &str = "\
# Derived and disposable. Deleting anything here costs time, never information.
cache/
";

/// `.map/.gitattributes`.
///
/// `linguist-generated=true` collapses index files in GitHub pull request
/// diffs by default while keeping them **expandable**. Deliberately no `-diff`:
/// a committed index steers a model's attention, so index changes must stay
/// reviewable. See spec §8.
///
/// `manifest.json merge=map` names the manifest merge driver. Two branches
/// that each edit a different file and re-index conflict on this one file, and
/// a textual merge cannot tell a legitimate union of object tables from a
/// same-key/different-bytes collision.
///
/// `.gitattributes` can only *name* a driver -- git refuses to execute a
/// command a repository distributes -- so the command itself is registered in
/// local config by `map init` and `map index`. Naming one the local machine
/// has not registered was previously avoided on the grounds that it falls back
/// to a binary merge. Measured with git 2.55: it does not. An unregistered
/// driver falls back to the ordinary text merge -- "Auto-merging" followed by a
/// content conflict with markers -- which is exactly what would happen with no
/// attribute at all. So the attribute costs a machine without the driver
/// nothing and buys every machine with it a resolved merge.
const GITATTRIBUTES: &str = "\
# Collapsed in pull request diffs, but still expandable and still diffable.
# A committed index steers a model's attention; it must remain reviewable.
index/** linguist-generated=true
manifest.json linguist-generated=true

# Resolved by `map merge`, registered in local git config by `map init` and
# `map index`. A machine without it gets git's ordinary text merge, the same
# as it would with no attribute here.
manifest.json merge=map
";

/// Outcome of an init run.
#[derive(Debug)]
pub(crate) struct Created {
    pub root: PathBuf,
    /// Relative to `root`.
    pub files: Vec<String>,
}

/// Errors from `map init`.
#[derive(Debug, thiserror::Error)]
pub(crate) enum InitError {
    #[error("{0} already exists; pass --force to overwrite its config")]
    Exists(PathBuf),
    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error(transparent)]
    Format(#[from] map_format::Error),
}

fn write(path: &Path, contents: &str) -> Result<(), InitError> {
    fs::write(path, contents).map_err(|e| InitError::Io {
        path: path.to_owned(),
        source: e,
    })
}

fn mkdir(path: &Path) -> Result<(), InitError> {
    fs::create_dir_all(path).map_err(|e| InitError::Io {
        path: path.to_owned(),
        source: e,
    })
}

/// Create a `.map` directory under `target`.
pub(crate) fn run(target: &Path, force: bool) -> Result<Created, InitError> {
    let root = target.join(".map");
    if root.exists() && !force {
        return Err(InitError::Exists(root));
    }

    // Object directories. `desc` is separate because a descriptor group spans
    // every dimension one classifier call covered. `tensor` and `cluster` are
    // created on first write by the indexer rather than here, since a lexical
    // index never produces either.
    for sub in ["index/shared/objects", "index/desc/objects", "cache"] {
        mkdir(&root.join(sub))?;
    }

    write(&root.join("config.toml"), CONFIG_TEMPLATE)?;
    write(&root.join(".gitignore"), GITIGNORE)?;
    write(&root.join(".gitattributes"), GITATTRIBUTES)?;

    // Parse back the template rather than trusting it, so a malformed default
    // fails here instead of at the user's first index.
    let config = Config::parse(CONFIG_TEMPLATE)?;

    let generated_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let manifest = Manifest::new(&config, env!("CARGO_PKG_VERSION"), generated_at)?;
    let bytes = manifest.to_bytes()?;
    fs::write(root.join("manifest.json"), &bytes).map_err(|e| InitError::Io {
        path: root.join("manifest.json"),
        source: e,
    })?;

    Ok(Created {
        root,
        files: vec![
            "config.toml".into(),
            "manifest.json".into(),
            ".gitignore".into(),
            ".gitattributes".into(),
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let mut p = std::env::temp_dir();
            p.push(format!("map-init-test-{}-{}", tag, std::process::id()));
            let _ = fs::remove_dir_all(&p);
            fs::create_dir_all(&p).unwrap();
            TempDir(p)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn template_matches_zero_config() {
        // The template is hand-authored for its comments; this is what stops
        // it drifting from the canonical default.
        assert_eq!(
            Config::parse(CONFIG_TEMPLATE).unwrap(),
            Config::zero_config()
        );
    }

    #[test]
    fn template_enables_only_the_offline_dimensions() {
        let cfg = Config::parse(CONFIG_TEMPLATE).unwrap();
        let active: Vec<_> = cfg.active().map(|(n, _)| n.as_str()).collect();
        assert_eq!(active, vec!["declaration", "lexical"]);
    }

    #[test]
    fn creates_expected_layout() {
        let dir = TempDir::new("layout");
        let created = run(&dir.0, false).unwrap();

        for f in &created.files {
            assert!(created.root.join(f).is_file(), "missing {f}");
        }
        assert!(created.root.join("index/shared/objects").is_dir());
        assert!(created.root.join("index/desc/objects").is_dir());
        assert!(created.root.join("cache").is_dir());
    }

    #[test]
    fn gitattributes_collapses_but_does_not_hide() {
        // -diff would make committed index changes unreviewable, which is
        // exactly the prompt-injection surface spec §8 refuses to create.
        assert!(GITATTRIBUTES.contains("linguist-generated=true"));
        assert!(!GITATTRIBUTES.contains("-diff"));
    }

    #[test]
    fn gitattributes_declares_the_manifest_merge_driver() {
        // The attribute travels with the clone; the driver command cannot,
        // because git will not run a command a repository distributes. A
        // machine that has not registered `map merge` falls back to git's
        // ordinary text merge with conflict markers -- measured with git 2.55,
        // and the same outcome as omitting the attribute -- so declaring it
        // costs that machine nothing.
        assert!(GITATTRIBUTES.contains("manifest.json merge=map"));
    }

    #[test]
    fn manifest_is_valid_and_empty() {
        let dir = TempDir::new("manifest");
        let created = run(&dir.0, false).unwrap();
        let bytes = fs::read(created.root.join("manifest.json")).unwrap();
        let m = map_format::Manifest::from_bytes(&bytes).unwrap();

        assert!(m.objects.is_empty());
        assert!(m.roots.is_empty());
        assert!(m.dimensions.contains_key("lexical"));
    }

    #[test]
    fn refuses_to_clobber_without_force() {
        let dir = TempDir::new("clobber");
        run(&dir.0, false).unwrap();
        assert!(matches!(run(&dir.0, false), Err(InitError::Exists(_))));
        assert!(run(&dir.0, true).is_ok());
    }
}
