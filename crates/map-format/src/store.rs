//! The loose object store.
//!
//! Objects are immutable and fan out two hex characters deep, exactly like
//! git's loose objects:
//!
//! ```text
//! index/shared/objects/ab/cdef0123…
//! ```
//!
//! Writes are atomic — an interrupted index leaves either a complete object
//! or none, never a half-written one that would later fail integrity checking
//! for the wrong reason. Landing an object is an atomic exclusive create
//! rather than a rename; see [`ObjectStore::put`] for why that distinction
//! matters.
//!
//! # What loose objects cost
//!
//! A churn spike over 200 commits at 1% churn on NTFS measured:
//!
//! | Form | Files in tree | History growth | Clone | Checkout |
//! |---|---|---|---|---|
//! | loose | 5,000 | 35.2 MB | 20.5 s | 5.1 s |
//! | packed | 16 | 34.6 MB | 0.94 s | 0.13 s |
//!
//! Loose objects do **not** win on history size — git's per-object tree
//! overhead cancels the dedup advantage — while costing ~22× on clone and
//! ~39× on checkout. They are used here anyway because packing needs
//! a deterministic shard-membership scheme to preserve clean merges, and that
//! is a harder problem than the storage saving justifies so far. The read/write
//! API is deliberately narrow so the backing layout can change underneath it.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{Error, Result};
use crate::hash::{ContentHash, ObjectKey};
use crate::manifest::{ObjectEntry, Tier};

/// Distinguishes concurrent temp files within one process.
static TEMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// A directory of immutable, input-addressed objects.
#[derive(Clone, Debug)]
pub struct ObjectStore {
    root: PathBuf,
}

impl ObjectStore {
    /// Open a store rooted at `root`. The directory need not exist yet.
    pub fn open(root: impl Into<PathBuf>) -> Self {
        ObjectStore { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn path_for(&self, key: ObjectKey) -> PathBuf {
        self.root.join(key.shard()).join(key.rest())
    }

    pub fn contains(&self, key: ObjectKey) -> bool {
        self.path_for(key).is_file()
    }

    /// Write an object and return its manifest entry.
    ///
    /// Objects are immutable, so writing a key that already exists is a no-op
    /// when the bytes match. When they differ this returns
    /// [`Error::KeyCollision`] rather than overwriting: under input-addressing
    /// two producers can derive the same key while writing different bytes,
    /// and silently keeping one side would corrupt the index invisibly.
    ///
    /// There is deliberately no replacing counterpart. A producer never
    /// re-runs a stage for a key already on disk — the driver stats the store
    /// first and reuses — so the same-key/different-bytes case only arises
    /// between *different* producers, where it is a collision to report rather
    /// than a write to land.
    ///
    /// # Concurrency
    ///
    /// The landing step is an atomic exclusive create (a hard link), not a
    /// rename — `fs::rename` replaces the destination on both Unix and
    /// Windows, so a check-then-rename would let two processes writing
    /// different bytes at one key silently pick a winner and never report the
    /// collision.
    pub fn put(&self, key: ObjectKey, bytes: &[u8], tier: Tier) -> Result<ObjectEntry> {
        self.write(key, bytes, tier)
    }

    fn write(&self, key: ObjectKey, bytes: &[u8], tier: Tier) -> Result<ObjectEntry> {
        let content_hash = ContentHash::of(bytes);
        let entry = ObjectEntry {
            content_hash,
            len: bytes.len() as u64,
            tier,
        };

        let path = self.path_for(key);
        if self.reconcile(key, &path, content_hash)? {
            return Ok(entry);
        }

        let dir = path.parent().expect("object path always has a parent");
        fs::create_dir_all(dir).map_err(|e| Error::io(dir, e))?;

        // Unique per process *and* per call: a parallel indexer has many
        // threads in this function at once, and a shared temp path would let
        // one thread truncate another's in-flight write.
        let seq = TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
        let tmp = dir.join(format!(".{}.{}.{seq}.tmp", key.rest(), std::process::id()));

        let result = (|| -> Result<()> {
            let mut f = fs::File::create(&tmp).map_err(|e| Error::io(&tmp, e))?;
            f.write_all(bytes).map_err(|e| Error::io(&tmp, e))?;
            f.sync_all().map_err(|e| Error::io(&tmp, e))?;
            drop(f);

            // Hard-linking fails if the destination exists, which makes landing
            // an object an atomic test-and-set. Rename cannot do this — it
            // replaces silently.
            match fs::hard_link(&tmp, &path) {
                Ok(()) => Ok(()),
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
                    // Lost the race. Whoever won must have written the same
                    // bytes, or this is a genuine collision.
                    self.reconcile(key, &path, content_hash)?;
                    Ok(())
                }
                Err(_) => {
                    // Filesystem without hard-link support. Fall back to
                    // rename, re-checking first — narrower than the hard-link
                    // path, but still reports the common case.
                    self.reconcile(key, &path, content_hash)?;
                    fs::rename(&tmp, &path).map_err(|e| Error::io(&path, e))
                }
            }
        })();

        let _ = fs::remove_file(&tmp);
        result.map(|()| entry)
    }

    /// Compare an already-present object against bytes we are about to write.
    ///
    /// `Ok(true)` — present and identical, nothing to do.
    /// `Ok(false)` — absent.
    /// `Err(KeyCollision)` — present with different bytes.
    fn reconcile(&self, key: ObjectKey, path: &Path, incoming: ContentHash) -> Result<bool> {
        if !path.is_file() {
            return Ok(false);
        }
        let existing = fs::read(path).map_err(|e| Error::io(path, e))?;
        let existing_hash = ContentHash::of(&existing);
        if existing_hash == incoming {
            return Ok(true);
        }
        Err(Error::KeyCollision {
            key: key.to_string(),
            a: existing_hash.to_string(),
            b: incoming.to_string(),
        })
    }

    /// Manifest entry for an object already in the store, without needing the
    /// bytes that would produce it.
    ///
    /// This is what makes incremental indexing cheap for expensive stages: the
    /// driver derives an object's key from its inputs, finds the object
    /// already present, and re-records it in the manifest **without running
    /// the stage**. Deduplicating at write time is too late — by then an LLM
    /// classifier has already been paid for.
    ///
    /// # Why `prior` is not optional in practice
    ///
    /// Whatever this reports goes straight into the next manifest. So on its
    /// own, reuse *launders* tampering: an attacker edits a committed
    /// descriptor object (spec §8, the prompt-injection case), the next
    /// `map index` skips the stage because the key is present, and re-records
    /// the injected bytes under a freshly computed hash that matches them
    /// perfectly. The integrity check of [`ObjectStore::get_verified`] then
    /// passes forever after, because the manifest now attests to the attacker's
    /// bytes.
    ///
    /// Pass the entry the *previous* manifest recorded for this key whenever
    /// there is one, and reuse is refused with the same
    /// [`Error::IntegrityFailure`] a verified read would give. `None` means
    /// genuinely nothing to compare against — a key this index has never
    /// recorded — and stats as before.
    pub fn stat(
        &self,
        key: ObjectKey,
        tier: Tier,
        prior: Option<&ObjectEntry>,
    ) -> Result<Option<ObjectEntry>> {
        let path = self.path_for(key);
        if !path.is_file() {
            return Ok(None);
        }
        let bytes = fs::read(&path).map_err(|e| Error::io(&path, e))?;
        let content_hash = ContentHash::of(&bytes);
        let len = bytes.len() as u64;

        if let Some(prior) = prior {
            if prior.content_hash != content_hash || prior.len != len {
                return Err(Error::IntegrityFailure {
                    key: key.to_string(),
                    expected: format!("{} ({} bytes)", prior.content_hash, prior.len),
                    actual: format!("{content_hash} ({len} bytes)"),
                });
            }
        }

        Ok(Some(ObjectEntry {
            content_hash,
            len,
            tier,
        }))
    }

    /// Read an object's raw bytes without verifying them.
    pub fn get(&self, key: ObjectKey) -> Result<Vec<u8>> {
        let path = self.path_for(key);
        fs::read(&path).map_err(|e| Error::io(&path, e))
    }

    /// Read an object and verify it against its manifest entry.
    ///
    /// This is the only integrity check the format has. Object keys are
    /// input-addressed, so bytes cannot be self-verified (spec §4) — a
    /// tampered descriptor still lands at a perfectly valid key. Prefer this
    /// over [`ObjectStore::get`] on any path that feeds model context.
    pub fn get_verified(&self, key: ObjectKey, entry: &ObjectEntry) -> Result<Vec<u8>> {
        let path = self.path_for(key);

        // Check the recorded length before reading. A tampered object can be
        // arbitrarily large, and this is a path that runs on untrusted
        // committed data.
        let meta = fs::metadata(&path).map_err(|e| Error::io(&path, e))?;
        if meta.len() != entry.len {
            return Err(Error::IntegrityFailure {
                key: key.to_string(),
                expected: format!("{} ({} bytes)", entry.content_hash, entry.len),
                actual: format!("{} bytes", meta.len()),
            });
        }

        let bytes = self.get(key)?;
        let actual = ContentHash::of(&bytes);
        if actual != entry.content_hash {
            return Err(Error::IntegrityFailure {
                key: key.to_string(),
                expected: entry.content_hash.to_string(),
                actual: actual.to_string(),
            });
        }
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::Fingerprint;

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let mut p = std::env::temp_dir();
            p.push(format!("map-store-test-{}-{}", tag, std::process::id()));
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

    fn key(seed: &[u8]) -> ObjectKey {
        ObjectKey::derive(&[seed], Fingerprint::of(b"cfg"))
    }

    #[test]
    fn put_then_get_roundtrips() {
        let dir = TempDir::new("roundtrip");
        let store = ObjectStore::open(&dir.0);
        let k = key(b"a");

        let entry = store.put(k, b"hello", Tier::A).unwrap();
        assert_eq!(entry.len, 5);
        assert!(store.contains(k));
        assert_eq!(store.get(k).unwrap(), b"hello");
        assert_eq!(store.get_verified(k, &entry).unwrap(), b"hello");
    }

    #[test]
    fn fans_out_two_hex_characters() {
        let dir = TempDir::new("fanout");
        let store = ObjectStore::open(&dir.0);
        let k = key(b"a");
        store.put(k, b"x", Tier::A).unwrap();

        let expected = dir.0.join(k.shard()).join(k.rest());
        assert!(expected.is_file());
    }

    #[test]
    fn rewriting_identical_bytes_is_a_noop() {
        let dir = TempDir::new("idempotent");
        let store = ObjectStore::open(&dir.0);
        let k = key(b"a");
        store.put(k, b"same", Tier::A).unwrap();
        assert!(store.put(k, b"same", Tier::A).is_ok());
    }

    #[test]
    fn same_key_different_bytes_refuses_to_overwrite() {
        let dir = TempDir::new("collision");
        let store = ObjectStore::open(&dir.0);
        let k = key(b"a");
        store.put(k, b"first", Tier::B).unwrap();

        let err = store.put(k, b"second", Tier::B).unwrap_err();
        assert!(matches!(err, Error::KeyCollision { .. }));
        // The original must survive untouched.
        assert_eq!(store.get(k).unwrap(), b"first");
    }

    #[test]
    fn tampering_is_detected() {
        let dir = TempDir::new("tamper");
        let store = ObjectStore::open(&dir.0);
        let k = key(b"a");
        let entry = store.put(k, b"trusted descriptor", Tier::C).unwrap();

        // Simulate a malicious edit to a committed descriptor object.
        fs::write(store.path_for(k), b"injected instructions").unwrap();

        assert!(store.get(k).is_ok(), "raw read still succeeds");
        assert!(
            matches!(
                store.get_verified(k, &entry),
                Err(Error::IntegrityFailure { .. })
            ),
            "verified read must reject it"
        );
    }

    #[test]
    fn concurrent_writers_cannot_silently_pick_a_winner() {
        // The failure this guards: check-then-rename lets both threads pass
        // the existence check and both rename, so one set of bytes vanishes
        // and no collision is ever reported.
        use std::sync::{Arc, Barrier};

        let dir = TempDir::new("race");
        let store = Arc::new(ObjectStore::open(&dir.0));
        let k = key(b"contested");
        let barrier = Arc::new(Barrier::new(8));

        let handles: Vec<_> = (0..8)
            .map(|i| {
                let store = Arc::clone(&store);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    store.put(k, format!("bytes from thread {i}").as_bytes(), Tier::B)
                })
            })
            .collect();

        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let winners = results.iter().filter(|r| r.is_ok()).count();
        let collisions = results
            .iter()
            .filter(|r| matches!(r, Err(Error::KeyCollision { .. })))
            .count();

        assert_eq!(winners, 1, "exactly one writer should land the object");
        assert_eq!(
            collisions, 7,
            "every loser must see a KeyCollision, not a silent overwrite"
        );
    }

    #[test]
    fn parallel_writes_of_identical_bytes_all_succeed() {
        // The common case: many threads deriving the same key from the same
        // input. All must succeed — this is dedup, not contention.
        use std::sync::{Arc, Barrier};

        let dir = TempDir::new("dedup");
        let store = Arc::new(ObjectStore::open(&dir.0));
        let k = key(b"shared");
        let barrier = Arc::new(Barrier::new(8));

        let handles: Vec<_> = (0..8)
            .map(|_| {
                let store = Arc::clone(&store);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    store.put(k, b"identical bytes", Tier::A)
                })
            })
            .collect();

        for h in handles {
            h.join()
                .unwrap()
                .expect("identical bytes must never collide");
        }
        assert_eq!(store.get(k).unwrap(), b"identical bytes");
    }

    #[test]
    fn truncated_object_is_rejected_before_hashing() {
        let dir = TempDir::new("length");
        let store = ObjectStore::open(&dir.0);
        let k = key(b"a");
        let entry = store.put(k, b"the full object", Tier::A).unwrap();

        fs::write(store.path_for(k), b"short").unwrap();
        assert!(matches!(
            store.get_verified(k, &entry),
            Err(Error::IntegrityFailure { .. })
        ));
    }

    #[test]
    fn reuse_of_untouched_bytes_matches_the_prior_entry() {
        let dir = TempDir::new("reuse");
        let store = ObjectStore::open(&dir.0);
        let k = key(b"a");
        let entry = store.put(k, b"trusted descriptor", Tier::C).unwrap();

        let reused = store.stat(k, Tier::C, Some(&entry)).unwrap().unwrap();
        assert_eq!(reused, entry);
    }

    #[test]
    fn a_tampered_object_is_refused_on_reuse() {
        // Without the prior entry the indexer re-records whatever is on disk,
        // so the injected bytes get a fresh matching hash in the next manifest
        // and every later integrity check passes (spec §8).
        let dir = TempDir::new("launder");
        let store = ObjectStore::open(&dir.0);
        let k = key(b"a");
        let entry = store.put(k, b"trusted descriptor", Tier::C).unwrap();

        fs::write(store.path_for(k), b"injected instructions").unwrap();

        assert!(matches!(
            store.stat(k, Tier::C, Some(&entry)),
            Err(Error::IntegrityFailure { .. })
        ));
    }

    #[test]
    fn a_length_preserving_edit_is_still_refused_on_reuse() {
        // Length alone is not the check; equal-length bytes must fail on hash.
        let dir = TempDir::new("samelen");
        let store = ObjectStore::open(&dir.0);
        let k = key(b"a");
        let entry = store.put(k, b"aaaaaaaa", Tier::C).unwrap();

        fs::write(store.path_for(k), b"bbbbbbbb").unwrap();
        assert!(matches!(
            store.stat(k, Tier::C, Some(&entry)),
            Err(Error::IntegrityFailure { .. })
        ));
    }

    #[test]
    fn a_key_never_recorded_before_stats_without_a_comparison() {
        let dir = TempDir::new("first-sight");
        let store = ObjectStore::open(&dir.0);
        let k = key(b"a");
        assert!(store.stat(k, Tier::A, None).unwrap().is_none());

        store.put(k, b"hello", Tier::A).unwrap();
        let seen = store.stat(k, Tier::A, None).unwrap().unwrap();
        assert_eq!(seen.len, 5);
        assert_eq!(seen.content_hash, ContentHash::of(b"hello"));
    }

    #[test]
    fn no_temp_files_survive_a_successful_write() {
        let dir = TempDir::new("clean");
        let store = ObjectStore::open(&dir.0);
        let k = key(b"a");
        store.put(k, b"x", Tier::A).unwrap();

        let shard = dir.0.join(k.shard());
        let leftovers: Vec<_> = fs::read_dir(&shard)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty());
    }
}
