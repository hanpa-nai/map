//! `map init` — create a `.map` directory.
//!
//! The generated config enables the two offline dimensions: `lexical`
//! (structural classification plus BM25) and `declaration` (the names a
//! segment declares, also scored by BM25). Neither needs an API key, network,
//! or model download, so `map init && map index && map find` succeeds on a
//! fresh machine with nothing configured. The embedding dimensions are
//! emitted commented out, one edit away.

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
# You commit this file. MAP builds the index from it. Thus an edit to a
# dimension changes its fingerprint, and MAP builds only the objects of that
# dimension again.

version = 1

# No operation updates the index automatically. `map find` updates it only when
# you use `-u`, because an embedding dimension or an LLM dimension can make an
# update slow. An interactive `map find` tells you when the index is stale. A
# run with no terminal does not print that notice, because the notice does not
# help a model.

[storage]
# The quantity of derived data in the repository, and its location.
#
# MAP implements only the two values in this section. It rejects the other
# values of the format:
#
#   commit    binary | descriptors-only | none   not implemented
#   location  ref (a ref with no parent, not in the tree)  not implemented
commit = "full"
location = "working-tree"

# ---------------------------------------------------------------------------
# Segmenter
#
# The segmenter cuts each resource into segments before a dimension reads it.
# All dimensions use the same segments. The segment identity connects a hit in
# one dimension to a hit in a different dimension.
#
# Line windows are the only implementation. They do not use the structure of
# the text. A cut by syntax uses a grammar for each format, and MAP must run on
# all text in a root. Thus the edge of a segment is at a line count, not at a
# point that the content gives. The `overlap` setting decreases the effect of
# this.
#
# The author measured the defaults on source code. Prose, transcripts, and
# records from other systems have different units, and MAP does not know which
# type you have. Use the defaults as a start, and measure before you change
# them. A change cuts the corpus again, and it gives new keys to all objects
# that come from the segments.
#
# [segmenter]
# impl    = "window"
# lines   = 40
# overlap = 8

# ---------------------------------------------------------------------------
# Dimensions
#
# A dimension is one field of the query, and the model writes text in it. The
# model reads the `description` to select the text for the field. Write the
# description for the model, not as documentation for a person.
#
# The descriptions in this file are for a root with content of an unknown type.
# If your root contains one type of content, write that type in the
# description. Examples are source code, notes, tickets, and a documentation
# set. The artifact fingerprint does not include the description. Thus an edit
# to a description has no cost and invalidates no objects.
#
# Each dimension must make a descriptor (text), a tensor (numeric), or the two.
# MAP rejects a dimension that makes no payload, because MAP cannot search it.
# The payload selects the scorer: BM25 for a descriptor, and cosine for a
# tensor. There is no `scorer` key.
# ---------------------------------------------------------------------------

[dimensions.lexical]
description = "Exact words, names, and literals as they appear in the text. Use this for anything you would otherwise search for verbatim."
enabled = true
classifier = { impl = "structural" }

# This dimension contains only the names that a segment declares (`fn`,
# `class`, `struct`, `def`, `type`, ...). Thus a name in this field ranks the
# segment that declares it above the segments that only use it. BM25 on full
# content does the opposite, because the callers contain a name more times than
# its one declaration does. The classifier is offline and has no cost. It is a
# line scan, not a parser. Thus it runs on all text.
[dimensions.declaration]
description = "The exact name of the thing whose declaration you want -- a function, type, class, module, or constant. A name here ranks the place that declares it above the places that use it."
enabled = true
classifier = { impl = "declaration" }

# ---------------------------------------------------------------------------
# Embedding dimensions
#
# The name of a dimension tells you the type of text that the caller puts in
# that query field. It does not give the method that calculates the score.
# `lexical` gets keywords. `semantic` gets an example of the content that you
# want. `descriptive` gets a description of that content.
#
# The dimensions in this part are optional. The `lexical` and `declaration`
# dimensions are offline, and no key and no model download are necessary for
# them.
#
# For `distilled`, the model must be in `~/.map/models`, and the binary must
# have the `distilled` feature. A binary with the `auto-distilled` feature can
# download the model with `map model fetch`. For `llm`, an endpoint that is
# compatible with the OpenAI API is necessary, and the binary must have the
# `llm` feature. Run `map llm login` to set the endpoint.
# The connection is in `~/.map/llm.toml`. It is not in this file, because you
# commit this file and it must not contain a secret.
# ---------------------------------------------------------------------------

# Offline, with no cost: static distilled embeddings of the content, with no
# LLM and no key. It finds content about the same subject, not only content
# with the same words.
#
# `content` prepares the segment for the embedder and stores no data. With
# `persist_output = false`, MAP does not store a second copy of the corpus.
# [dimensions.semantic]
# description = "Content resembling what you are looking for -- paste or paraphrase the material itself."
# classifier = { impl = "content", persist_output = false }
# embedder   = { impl = "distilled" }

# An LLM writes a prose descriptor of each segment. The distilled embedder
# makes an embedding from that prose. The fabricator makes a tree of clusters
# with labels.
#
# The prompt tells the model the type of content in the corpus. MAP puts frame
# text around the prompt. The frame text gives only the number of segments and
# the shape of the answer. It gives no subject. Thus the default prompt applies
# the same instructions to prose and to source code.
#
# The fingerprint includes the prompt and the frame text. A change to one of
# them makes MAP classify the corpus again.
# [dimensions.descriptive]
# description = "A description of what you are looking for, in your own words -- what it does, what it is about, what happens when it runs."
# classifier = { impl = "llm", prompts = { "0" = "Describe what this is and what it does, in one concise sentence of plain language.", "1" = "Name the theme these descriptions share, as a short noun phrase." } }
# embedder   = { impl = "distilled" }
# fabricator = { impl = "agglomerative", threshold = 0.70, min_cluster = 5, max_cluster = 15 }
"##;

/// `.map/.gitignore` — everything derived and disposable.
const GITIGNORE: &str = "\
# Derived data. If you delete it, MAP builds it again. That uses time and
# removes no information.
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
# A pull request shows these files closed, and a reviewer can open them.
# A committed index controls the text that a model reads. Thus it must be
# possible to read each change in a review.
index/** linguist-generated=true
manifest.json linguist-generated=true

# `map merge` resolves this file. `map init` and `map index` add the driver
# command to the local git configuration. A machine without the driver gets
# the usual git text merge, the same as with no attribute.
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
