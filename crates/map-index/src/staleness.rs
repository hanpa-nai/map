//! Cheap staleness detection.
//!
//! Repairing before a query is only usable if *detecting that nothing changed*
//! is cheap. Running a full incremental index to discover that costs
//! O(corpus): every resource is read, decoded, and segmented before the
//! content-derived key can be compared. Measured on ripgrep that is ~220 ms —
//! twenty times the query it was meant to precede, and it grows with the
//! corpus, which is precisely the scaling problem the packed index removed.
//!
//! So the fast path never reads file contents. It walks the tree, stats each
//! resource, and compares size and modification time against a snapshot taken
//! at index time. Unchanged corpus means no index pass at all.
//!
//! The snapshot lives in gitignored `.map/cache/`, so a fresh clone simply has
//! none and pays one full pass — during which every object is reused and the
//! classifier never runs.
//!
//! # What this deliberately does not do
//!
//! Size and mtime can miss an edit that preserves both within the
//! filesystem's timestamp resolution. That is the same trade `make` and
//! `cargo` accept, and the escape hatch is explicit: `map index` always does
//! the real content-addressed comparison. This is a *skip-work* heuristic,
//! never the thing that decides what an object's key is.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::UNIX_EPOCH;

use map_core::Resource;
use serde::{Deserialize, Serialize};

/// Size and modification time for one resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stat {
    pub size: u64,
    /// Nanoseconds since the Unix epoch.
    pub mtime_ns: i128,
}

/// What the tree looked like when the index was last written.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Snapshot {
    /// Canonical resource key to its stat.
    pub entries: BTreeMap<String, Stat>,
    /// Fingerprint of the `config.toml` the index was built from.
    ///
    /// An edited config changes what the index should hold without touching a
    /// single resource, so stats alone report a corpus that has not drifted
    /// while every object key has moved. `None` is a snapshot written before
    /// this field existed, which cannot be confirmed and so reads as stale.
    #[serde(default)]
    pub config_fingerprint: Option<String>,
    /// Content hash of the manifest this run wrote.
    ///
    /// Catches the index being replaced under an unchanged tree — a pull, a
    /// branch switch, a `map gc --prune`. The snapshot then describes an index
    /// that is no longer there.
    #[serde(default)]
    pub manifest_hash: Option<String>,
}

impl Snapshot {
    /// Stat every resource under `root`.
    ///
    /// The identity fields are filled in by the caller that knows them — the
    /// indexer, once it has written the manifest.
    pub fn take(root: &Path, resources: &[Resource]) -> Self {
        let mut entries = BTreeMap::new();
        for resource in resources {
            if let Some(stat) = stat_of(&root.join(&resource.key)) {
                entries.insert(resource.key.clone(), stat);
            }
        }
        Snapshot {
            entries,
            config_fingerprint: None,
            manifest_hash: None,
        }
    }

    /// Load a snapshot, or `None` if absent or unreadable.
    ///
    /// An unreadable snapshot is treated as absent rather than as an error: it
    /// is a cache, and the only cost of ignoring it is doing the work.
    pub fn load(path: &Path) -> Option<Self> {
        let bytes = std::fs::read(path).ok()?;
        serde_json::from_slice(&bytes).ok()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }

    /// Whether the tree and the index identity still match this snapshot.
    pub fn matches(
        &self,
        root: &Path,
        resources: &[Resource],
        config_fingerprint: Option<&str>,
        manifest_hash: Option<&str>,
    ) -> bool {
        !self
            .compare(root, resources, config_fingerprint, manifest_hash)
            .is_stale()
    }

    /// What changed since this snapshot was taken.
    ///
    /// Compares the resource *set* as well as each stat, so an added or
    /// deleted file counts even when every surviving file is untouched — and
    /// the index's own identity, because a config edit or a pulled manifest
    /// changes what the index should hold while every mtime stays put.
    ///
    /// A current value that could not be read is `None`, which counts as
    /// changed: "cannot confirm" and "confirmed different" both mean the only
    /// safe answer is to do the work.
    pub fn compare(
        &self,
        root: &Path,
        resources: &[Resource],
        config_fingerprint: Option<&str>,
        manifest_hash: Option<&str>,
    ) -> StaleReport {
        let mut report = StaleReport {
            config_changed: !confirms(self.config_fingerprint.as_deref(), config_fingerprint),
            index_replaced: !confirms(self.manifest_hash.as_deref(), manifest_hash),
            ..StaleReport::default()
        };

        for resource in resources {
            match self.entries.get(&resource.key) {
                None => report.added += 1,
                Some(recorded) => {
                    if stat_of(&root.join(&resource.key)) != Some(*recorded) {
                        report.changed += 1;
                    }
                }
            }
        }

        let present: std::collections::BTreeSet<&str> =
            resources.iter().map(|r| r.key.as_str()).collect();
        report.removed = self
            .entries
            .keys()
            .filter(|key| !present.contains(key.as_str()))
            .count();

        report
    }
}

/// Both sides known and equal. Anything else — either side unknown, or a real
/// difference — is not a confirmation.
fn confirms(recorded: Option<&str>, current: Option<&str>) -> bool {
    matches!((recorded, current), (Some(a), Some(b)) if a == b)
}

/// How far the tree has drifted from the index.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StaleReport {
    /// Present in both, but modified.
    pub changed: usize,
    pub added: usize,
    pub removed: usize,
    /// `config.toml` no longer fingerprints to what the index was built from,
    /// so every object key it feeds has moved.
    pub config_changed: bool,
    /// The manifest on disk is not the one this snapshot was written beside.
    pub index_replaced: bool,
}

impl StaleReport {
    pub fn is_stale(&self) -> bool {
        self.total() > 0
    }

    pub fn total(&self) -> usize {
        self.changed
            + self.added
            + self.removed
            + usize::from(self.config_changed)
            + usize::from(self.index_replaced)
    }
}

impl std::fmt::Display for StaleReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut parts = Vec::new();
        if self.changed > 0 {
            parts.push(format!("{} changed", self.changed));
        }
        if self.added > 0 {
            parts.push(format!("{} added", self.added));
        }
        if self.removed > 0 {
            parts.push(format!("{} removed", self.removed));
        }
        if self.config_changed {
            parts.push("config changed".to_owned());
        }
        if self.index_replaced {
            parts.push("index replaced".to_owned());
        }
        f.write_str(&parts.join(", "))
    }
}

fn stat_of(path: &Path) -> Option<Stat> {
    let metadata = std::fs::metadata(path).ok()?;
    let mtime = metadata.modified().ok()?;
    let mtime_ns = match mtime.duration_since(UNIX_EPOCH) {
        Ok(d) => d.as_nanos() as i128,
        // Pre-epoch timestamps are rare but legal.
        Err(e) => -(e.duration().as_nanos() as i128),
    };
    Some(Stat {
        size: metadata.len(),
        mtime_ns,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Dir(std::path::PathBuf);

    impl Dir {
        fn new(tag: &str) -> Self {
            let mut p = std::env::temp_dir();
            p.push(format!("map-stale-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).unwrap();
            Dir(p)
        }
        fn write(&self, name: &str, text: &str) {
            std::fs::write(self.0.join(name), text).unwrap();
        }
    }

    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    const CONFIG: &str = "the config fingerprint at index time";
    const MANIFEST: &str = "the manifest hash at index time";

    /// A snapshot as the indexer writes one: the stats plus the identity of the
    /// index it was written beside.
    fn snapshot(root: &Path, list: &[Resource]) -> Snapshot {
        let mut snapshot = Snapshot::take(root, list);
        snapshot.config_fingerprint = Some(CONFIG.to_owned());
        snapshot.manifest_hash = Some(MANIFEST.to_owned());
        snapshot
    }

    /// Compare against an unchanged index identity, so a test that is about the
    /// tree says nothing about the config or the manifest.
    fn tree_matches(snapshot: &Snapshot, root: &Path, list: &[Resource]) -> bool {
        snapshot.matches(root, list, Some(CONFIG), Some(MANIFEST))
    }

    fn resources(names: &[&str]) -> Vec<Resource> {
        names
            .iter()
            .map(|n| Resource {
                key: (*n).to_owned(),
                size: 0,
            })
            .collect()
    }

    #[test]
    fn an_untouched_tree_matches() {
        let dir = Dir::new("same");
        dir.write("a.rs", "fn a() {}");
        dir.write("b.rs", "fn b() {}");
        let list = resources(&["a.rs", "b.rs"]);

        let snapshot = snapshot(&dir.0, &list);
        assert!(tree_matches(&snapshot, &dir.0, &list));
    }

    #[test]
    fn an_edit_is_detected() {
        let dir = Dir::new("edit");
        dir.write("a.rs", "fn a() {}");
        let list = resources(&["a.rs"]);
        let snapshot = snapshot(&dir.0, &list);

        // Different length, so this is caught regardless of clock resolution.
        dir.write("a.rs", "fn a() { changed(); }");
        assert!(!tree_matches(&snapshot, &dir.0, &list));
    }

    #[test]
    fn an_added_file_is_detected() {
        let dir = Dir::new("add");
        dir.write("a.rs", "fn a() {}");
        let snapshot = snapshot(&dir.0, &resources(&["a.rs"]));

        dir.write("b.rs", "fn b() {}");
        assert!(!tree_matches(
            &snapshot,
            &dir.0,
            &resources(&["a.rs", "b.rs"])
        ));
    }

    #[test]
    fn a_deleted_file_is_detected() {
        let dir = Dir::new("delete");
        dir.write("a.rs", "fn a() {}");
        dir.write("b.rs", "fn b() {}");
        let snapshot = snapshot(&dir.0, &resources(&["a.rs", "b.rs"]));

        std::fs::remove_file(dir.0.join("b.rs")).unwrap();
        assert!(!tree_matches(&snapshot, &dir.0, &resources(&["a.rs"])));
    }

    #[test]
    fn roundtrips() {
        let dir = Dir::new("roundtrip");
        dir.write("a.rs", "fn a() {}");
        let snapshot = snapshot(&dir.0, &resources(&["a.rs"]));

        let path = dir.0.join("snap.json");
        std::fs::write(&path, snapshot.to_bytes()).unwrap();
        assert_eq!(Snapshot::load(&path), Some(snapshot));
    }

    #[test]
    fn an_edited_config_is_stale_though_no_file_moved() {
        // The gap this closes: object keys fold in the config fingerprint, so
        // editing a prompt, a segmenter, or a dimension set moves every key
        // while leaving every mtime exactly where it was. Stats alone reported
        // a clean tree and `-u` short-circuited, so the edit never took effect.
        let dir = Dir::new("config-edit");
        dir.write("a.rs", "fn a() {}");
        let list = resources(&["a.rs"]);
        let snapshot = snapshot(&dir.0, &list);

        let report = snapshot.compare(&dir.0, &list, Some("a different config"), Some(MANIFEST));
        assert!(report.config_changed);
        assert_eq!(report.changed, 0, "no file moved");
        assert!(report.is_stale());
        assert_eq!(report.to_string(), "config changed");
    }

    #[test]
    fn a_replaced_manifest_is_stale_though_no_file_moved() {
        // A pull, a branch switch, or a prune swaps the index out from under an
        // untouched tree.
        let dir = Dir::new("manifest-swap");
        dir.write("a.rs", "fn a() {}");
        let list = resources(&["a.rs"]);
        let snapshot = snapshot(&dir.0, &list);

        let report = snapshot.compare(&dir.0, &list, Some(CONFIG), Some("someone else's index"));
        assert!(report.index_replaced);
        assert!(report.is_stale());
        assert_eq!(report.to_string(), "index replaced");
    }

    #[test]
    fn a_snapshot_predating_the_identity_fields_is_stale() {
        // An old `stat.json` records neither, and unknown is not confirmation:
        // it would otherwise claim a clean tree for an index whose config it
        // has never seen.
        let dir = Dir::new("old-snapshot");
        dir.write("a.rs", "fn a() {}");
        let list = resources(&["a.rs"]);
        let old = Snapshot::take(&dir.0, &list);

        assert!(!tree_matches(&old, &dir.0, &list));
        // And it still deserializes, rather than being discarded as corrupt.
        let round: Snapshot = serde_json::from_slice(&old.to_bytes()).unwrap();
        assert_eq!(round, old);
    }

    #[test]
    fn an_identity_that_cannot_be_read_is_not_a_confirmation() {
        let dir = Dir::new("unreadable-identity");
        dir.write("a.rs", "fn a() {}");
        let list = resources(&["a.rs"]);
        let snapshot = snapshot(&dir.0, &list);

        let report = snapshot.compare(&dir.0, &list, None, None);
        assert!(report.config_changed && report.index_replaced);
        assert_eq!(report.total(), 2);
    }

    #[test]
    fn a_missing_or_corrupt_snapshot_is_just_absent() {
        let dir = Dir::new("corrupt");
        assert_eq!(Snapshot::load(&dir.0.join("nope.json")), None);
        dir.write("bad.json", "not json at all");
        assert_eq!(Snapshot::load(&dir.0.join("bad.json")), None);
    }
}
