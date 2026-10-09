//! Collecting objects nothing can reach any more.
//!
//! Object keys are input-addressed — `hash(input + stage_config_fingerprint)`
//! (spec §4) — so editing anything that feeds a fingerprint produces a *new*
//! key, which is a new file. The old one is neither overwritten nor removed,
//! so an index accumulates a stranded generation per edit until something
//! collects them. On ripgrep, two config edits are enough to strand a third of
//! the index.
//!
//! Which edits strand a generation follows from `artifact_fingerprint`:
//! `classifier`, `embedder`, `fabricator`, `dims`, `quant`. Editing a
//! `description` does not, deliberately, so model-facing prose stays free to
//! tune.
//!
//! # Why this is not `rm`
//!
//! A committed index commits its manifest, so **every ref has its own reachable
//! set**. Pruning against the working tree alone deletes objects another branch
//! still references; merging that branch back is a delete-versus-unmodified
//! merge, which git resolves by deleting — leaving a manifest pointing at an
//! object that is gone. Silent corruption assembled from two individually
//! correct operations.
//!
//! So reachability is the union over the working-tree manifest and every
//! manifest reachable from a git ref. Git is used **if it is there** and is
//! never required: with no repository, or with `.map` untracked, no other
//! manifest can exist and the working tree's is already the whole truth. The
//! one case that refuses to prune is a repository whose git cannot be run —
//! then the answer is unknown rather than empty, and guessing is what this
//! module exists to prevent.
//!
//! # What it cannot do
//!
//! Deleting a blob removes it from future checkouts; it stays in every packfile
//! forever and cloners still download it. Collection bounds *future* growth and
//! shrinks a checkout — garbage already committed is sunk. That makes this an
//! ordering constraint rather than a cleanup task: it wants to exist before
//! anyone is advised to commit an index, which is the same conclusion spec §9.1
//! reaches from re-fabrication cost.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use map_format::Manifest;

use crate::IndexError;

/// Which manifests contributed to the reachable set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GitScope {
    /// No repository above the index. The working tree holds the only manifest
    /// there is, so tracing it is complete.
    NoRepository,
    /// A repository, but `.map/manifest.json` is not tracked — same conclusion:
    /// no commit holds a different one.
    Untracked,
    /// Manifests were read from these refs as well as the working tree.
    Refs(Vec<String>),
    /// A repository whose git could not be run, so other refs' manifests cannot
    /// be read. Reachability is unknown and pruning is refused.
    Unavailable(String),
}

impl std::fmt::Display for GitScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GitScope::NoRepository => {
                write!(f, "no git repository; working tree is the only state")
            }
            GitScope::Untracked => write!(f, ".map is untracked; working tree is the only state"),
            GitScope::Refs(refs) => write!(f, "working tree plus {} git ref(s)", refs.len()),
            GitScope::Unavailable(why) => write!(f, "git could not be run: {why}"),
        }
    }
}

/// One object no manifest reaches.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnreachableObject {
    /// The object store it sits in: `shared`, `desc`, `tensor`, or `cluster`.
    pub store: String,
    pub key: String,
    pub len: u64,
    path: PathBuf,
}

/// What a collection found, and did.
#[derive(Clone, Debug)]
pub struct GcReport {
    pub scope: GitScope,
    /// Distinct object keys any consulted manifest declares.
    pub reachable: usize,
    pub unreachable: Vec<UnreachableObject>,
    /// Bytes held by [`GcReport::unreachable`].
    pub bytes: u64,
    pub pruned: bool,
}

impl GcReport {
    /// Unreachable objects grouped by store, in canonical order.
    pub fn by_store(&self) -> std::collections::BTreeMap<&str, (usize, u64)> {
        let mut out: std::collections::BTreeMap<&str, (usize, u64)> = Default::default();
        for object in &self.unreachable {
            let entry = out.entry(object.store.as_str()).or_default();
            entry.0 += 1;
            entry.1 += object.len;
        }
        out
    }
}

/// Find unreachable objects under `root`'s index, and optionally delete them.
///
/// Reports without `prune`; that is the whole safety model. An orphaned
/// descriptor is LLM output that costs money to recreate, and reverting a
/// config edit can make a whole stranded generation reachable again — git
/// survives the same hazard with a reflog grace period, which this format has
/// no equivalent of.
pub fn collect(start: &Path, prune: bool) -> Result<GcReport, IndexError> {
    let root = map_core::find_map_root(start)
        .ok_or_else(|| IndexError::NotInitialized(start.to_path_buf()))?;
    let map_dir = root.join(".map");

    let (scope, mut manifests) = git_manifests(&root, &map_dir)?;

    // The working tree's manifest is always a root.
    let path = map_dir.join("manifest.json");
    let bytes = std::fs::read(&path).map_err(|e| IndexError::Io {
        path: path.clone(),
        source: e,
    })?;
    manifests.push(parse_manifest(&bytes, "the working tree")?);

    // The object table, not the resource roots. It is the superset — every
    // object the index declares, including clusters, which hang off
    // `manifest.clusters` rather than off any `ResourceRoot`. Tracing roots
    // would collect strictly more and is the stricter thing to do later; a
    // collector's first version should err toward keeping.
    let mut reachable: BTreeSet<String> = BTreeSet::new();
    for manifest in &manifests {
        reachable.extend(manifest.objects.keys().map(|k| k.to_string()));
    }

    let mut unreachable = Vec::new();
    let mut bytes_held = 0u64;
    for (store, dir) in object_stores(&map_dir) {
        for (key, path, len) in objects_in(&dir)? {
            if reachable.contains(&key) {
                continue;
            }
            bytes_held += len;
            unreachable.push(UnreachableObject {
                store: store.clone(),
                key,
                len,
                path,
            });
        }
    }
    unreachable.sort_by(|a, b| a.store.cmp(&b.store).then(a.key.cmp(&b.key)));

    let mut report = GcReport {
        scope,
        reachable: reachable.len(),
        unreachable,
        bytes: bytes_held,
        pruned: false,
    };

    if prune {
        if let GitScope::Unavailable(why) = &report.scope {
            return Err(IndexError::GcUnsafe(format!(
                "this is a git repository but git could not be run ({why}), so the manifests \
                 on other refs cannot be read. Objects another branch still references would \
                 be deleted. Nothing was removed."
            )));
        }
        for object in &report.unreachable {
            std::fs::remove_file(&object.path).map_err(|e| IndexError::Io {
                path: object.path.clone(),
                source: e,
            })?;
        }
        prune_empty_shards(&map_dir);
        report.pruned = true;
    }

    Ok(report)
}

fn parse_manifest(bytes: &[u8], origin: &str) -> Result<Manifest, IndexError> {
    Manifest::from_bytes(bytes).map_err(|e| {
        // Refuse rather than treat an unreadable manifest as reaching nothing:
        // that reading would delete everything it declares.
        IndexError::GcUnsafe(format!(
            "the manifest at {origin} could not be parsed ({e}), so what it reaches is unknown"
        ))
    })
}

/// Manifests held by git refs, and which refs those were.
///
/// Never fails for want of git — only for a repository whose git will not run,
/// and even then it reports rather than errors, so `map gc` still lists what it
/// found and only pruning is refused.
fn git_manifests(root: &Path, map_dir: &Path) -> Result<(GitScope, Vec<Manifest>), IndexError> {
    if !has_git_dir(root) {
        return Ok((GitScope::NoRepository, Vec::new()));
    }

    // Ask git for the repository-relative path rather than deriving it. The
    // root here has been through `canonicalize`, which on Windows yields a
    // `\\?\` extended-length prefix that no amount of `strip_prefix` against
    // git's own toplevel will match — and a mangled path does not fail loudly,
    // it makes `git show` print something else that happens to succeed.
    //
    // This also separates the two answers cleanly: `ls-files` exits zero with
    // no output when the file is untracked, and non-zero only when git itself
    // could not run.
    let Some(listed_path) = git(map_dir, &["ls-files", "--full-name", "--", "manifest.json"])
    else {
        return Ok((
            GitScope::Unavailable(
                "git is not installed or could not read this repository".to_owned(),
            ),
            Vec::new(),
        ));
    };
    let relative = String::from_utf8_lossy(&listed_path).trim().to_owned();
    if relative.is_empty() {
        return Ok((GitScope::Untracked, Vec::new()));
    }

    let Some(listed) = git(map_dir, &["for-each-ref", "--format=%(refname)"]) else {
        return Ok((
            GitScope::Unavailable("`git for-each-ref` failed".to_owned()),
            Vec::new(),
        ));
    };

    let mut names: Vec<String> = String::from_utf8_lossy(&listed)
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_owned)
        .collect();
    names.push("HEAD".to_owned());

    let mut manifests = Vec::new();
    let mut consulted = Vec::new();
    for name in names {
        // A ref older than the index has no manifest; that is not an error.
        let Some(bytes) = git(map_dir, &["show", &format!("{name}:{relative}")]) else {
            continue;
        };
        manifests.push(parse_manifest(&bytes, &name)?);
        consulted.push(name);
    }

    Ok((GitScope::Refs(consulted), manifests))
}

/// A `.git` entry above `root` — a directory normally, a file in a worktree.
///
/// Checked without invoking anything, so "is this a repository at all" is
/// answerable even when git is missing. That is what makes the refusal to prune
/// possible rather than a silent working-tree-only guess.
fn has_git_dir(root: &Path) -> bool {
    let mut here = root;
    loop {
        if here.join(".git").exists() {
            return true;
        }
        match here.parent() {
            Some(parent) => here = parent,
            None => return false,
        }
    }
}

/// Run git, returning stdout on success and `None` on any failure.
fn git(dir: &Path, args: &[&str]) -> Option<Vec<u8>> {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .ok()?;
    out.status.success().then_some(out.stdout)
}

fn object_stores(map_dir: &Path) -> Vec<(String, PathBuf)> {
    let index = map_dir.join("index");
    let Ok(entries) = std::fs::read_dir(&index) else {
        return Vec::new();
    };
    let mut out: Vec<(String, PathBuf)> = entries
        .flatten()
        .filter(|e| e.path().is_dir())
        .map(|e| {
            (
                e.file_name().to_string_lossy().into_owned(),
                e.path().join("objects"),
            )
        })
        .filter(|(_, dir)| dir.is_dir())
        .collect();
    out.sort();
    out
}

/// Every object in a two-hex-shard store as `(key, path, len)`.
fn objects_in(dir: &Path) -> Result<Vec<(String, PathBuf, u64)>, IndexError> {
    let mut out = Vec::new();
    let Ok(shards) = std::fs::read_dir(dir) else {
        return Ok(out);
    };
    for shard in shards.flatten() {
        let shard_path = shard.path();
        if !shard_path.is_dir() {
            continue;
        }
        let prefix = shard.file_name().to_string_lossy().into_owned();
        let Ok(files) = std::fs::read_dir(&shard_path) else {
            continue;
        };
        for file in files.flatten() {
            let path = file.path();
            if !path.is_file() {
                continue;
            }
            let len = file.metadata().map(|m| m.len()).unwrap_or(0);
            let key = format!("{prefix}{}", file.file_name().to_string_lossy());
            out.push((key, path, len));
        }
    }
    Ok(out)
}

/// Remove shard directories emptied by pruning. Best effort — a shard that
/// will not delete is not a failed collection.
fn prune_empty_shards(map_dir: &Path) {
    for (_, dir) in object_stores(map_dir) {
        let Ok(shards) = std::fs::read_dir(&dir) else {
            continue;
        };
        for shard in shards.flatten() {
            let path = shard.path();
            if path.is_dir() && std::fs::read_dir(&path).is_ok_and(|mut e| e.next().is_none()) {
                let _ = std::fs::remove_dir(&path);
            }
        }
    }
}
