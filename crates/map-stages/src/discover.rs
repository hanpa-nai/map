//! Filesystem discoverer.

use std::path::Path;

use map_core::{Discoverer, Error, Resource, Result, Stage};
use unicode_normalization::UnicodeNormalization;

/// Walks a directory tree, honouring `.gitignore` and skipping `.map` itself.
#[derive(Clone, Debug)]
pub struct FsDiscoverer {
    /// Skip files larger than this. Defaults to 4 MiB.
    pub max_bytes: u64,
}

/// Default size cap. Not `#[derive(Default)]`: a zero cap skips *every* file,
/// so a derived default would discover nothing and report it as an empty corpus.
const DEFAULT_MAX_BYTES: u64 = 4 * 1024 * 1024;

impl Default for FsDiscoverer {
    fn default() -> Self {
        FsDiscoverer {
            max_bytes: DEFAULT_MAX_BYTES,
        }
    }
}

impl FsDiscoverer {
    /// A discoverer with the default size cap.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Convert a path to a canonical resource key (spec §6.1).
///
/// Root-relative, forward slashes, NFC-normalized. macOS hands back
/// NFD-decomposed filenames where Linux returns NFC, so without this the same
/// file yields a different key per platform and Tier A determinism breaks.
pub fn canonical_key(root: &Path, path: &Path) -> Option<String> {
    let rel = path.strip_prefix(root).ok()?;
    let mut key = String::new();
    for part in rel.components() {
        let text = part.as_os_str().to_str()?;
        if !key.is_empty() {
            key.push('/');
        }
        key.extend(text.nfc());
    }
    (!key.is_empty()).then_some(key)
}

/// Reject a resource key that could escape the index root or corrupt a terminal.
///
/// Keys travel inside a committed index, so a cloned repository can carry
/// hostile ones. Checked here rather than at use sites so there is one place to
/// get it right. Beyond path traversal, control characters are rejected: a key
/// is printed to the terminal on every hit, and an embedded ESC or CR could
/// rewrite the line or spoof output for whoever (or whatever) reads it.
pub fn check_resource_key(key: &str) -> Result<()> {
    // A `:` in the first component is a Windows drive prefix (`C:/x`, `c:x`),
    // and `Path::join` replaces the root with such a path instead of appending
    // it. The index is committed, so the rule holds on every platform, not
    // only where the prefix would bite.
    let drive_prefix = key
        .split('/')
        .next()
        .is_some_and(|first| first.contains(':'));
    let unsafe_key = key.is_empty()
        || key.starts_with('/')
        || key.contains('\\')
        || drive_prefix
        || key.chars().any(char::is_control) // NUL, ESC, CR, LF, DEL, C1…
        || key
            .split('/')
            .any(|c| c == ".." || c == "." || c.is_empty());

    if unsafe_key {
        return Err(Error::UnsafeResourceKey(key.to_owned()));
    }
    Ok(())
}

impl Stage for FsDiscoverer {
    fn implementation(&self) -> &str {
        "fs"
    }

    fn config(&self) -> String {
        // The size cap changes which resources exist at all, so it belongs in
        // the fingerprint even though discovery writes no object of its own.
        format!("fs:max_bytes={}", self.max_bytes)
    }
}

impl Discoverer for FsDiscoverer {
    fn discover(&self, root: &Path) -> Result<Vec<Resource>> {
        let mut found = Vec::new();

        let walker = ignore::WalkBuilder::new(root)
            .hidden(false)
            .git_ignore(true)
            .git_global(false)
            .filter_entry(|e| e.file_name() != ".map" && e.file_name() != ".git")
            .build();

        for entry in walker.flatten() {
            if !entry.file_type().is_some_and(|t| t.is_file()) {
                continue;
            }
            let Some(key) = canonical_key(root, entry.path()) else {
                continue; // non-UTF-8 path; nothing downstream could address it
            };
            if check_resource_key(&key).is_err() {
                continue;
            }
            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
            if size == 0 || size > self.max_bytes {
                continue;
            }
            found.push(Resource { key, size });
        }

        // Enumeration order is unstable across platforms and runs; everything
        // downstream inherits this order, so sorting is load-bearing for
        // Tier A determinism, not cosmetic.
        found.sort_by(|a, b| a.key.cmp(&b.key));
        Ok(found)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_use_forward_slashes() {
        let root = Path::new("/repo");
        let key = canonical_key(root, &Path::new("/repo").join("src").join("main.rs"));
        assert_eq!(key.as_deref(), Some("src/main.rs"));
    }

    #[test]
    fn traversing_keys_are_rejected() {
        for bad in [
            "../secrets",
            "/etc/passwd",
            "a/../../b",
            "",
            "a//b",
            "a\\b",
            "./x",
            "src/a\x1b[2K.rs",        // ESC — terminal escape injection
            "src/a\rb.rs",            // carriage return
            "a\nb",                   // newline
            "a\0b",                   // NUL
            "C:/Users/x/.ssh/id_rsa", // drive prefix: `join` would drop the root
            "c:secret.txt",           // drive-relative, same escape
            "D:",
        ] {
            assert!(
                check_resource_key(bad).is_err(),
                "{bad:?} should be rejected"
            );
        }
    }

    #[test]
    fn ordinary_keys_are_accepted() {
        for good in ["src/main.rs", "a", "crates/map-core/src/lib.rs"] {
            check_resource_key(good).unwrap();
        }
    }
}
