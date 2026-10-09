//! Rebuilding the derived cache packs from committed objects.
//!
//! This is the **single** definition of what a cache pack contains and in what
//! id order. Two callers use it, and it matters that they use the *same* one:
//!
//! - the indexer, after it has written the committed objects (`map index`,
//!   `map find -u`), builds every pack from them;
//! - the retriever, on a cache miss (a fresh clone), rebuilds the missing packs
//!   and writes them through.
//!
//! If these were two implementations they would drift — a change to the pack
//! layout or the id assignment in one and not the other would make a
//! query-rebuilt cache disagree with an index-built one, silently. One function
//! makes that impossible.
//!
//! # Trust
//!
//! `verify` decides whether each object is hash-checked against the manifest
//! before use. The retriever passes `true`: a committed index is untrusted data
//! that steers an agent, so its bytes are verified. The indexer passes `false`:
//! it wrote those objects microseconds ago and has no reason to re-hash them.
//!
//! The packs themselves are never verified that way — they are derived, not
//! committed, and have no entry in the manifest. Instead each carries a
//! [`pack_marker`]: a per-machine keyed hash over the manifest fingerprint, the
//! dimension name, and the pack bytes, recorded in [`PackMarkers`]. Anything
//! that does not match is not a pack this machine built, and is rebuilt from
//! the verified objects rather than mapped. The pack bytes are in that hash
//! because `cache/` is gitignored and git overwrites ignored files on pull
//! without warning, so a hostile clone can drop its own `lexical.pack` into a
//! checkout whose manifest is untouched.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use map_core::{DescriptorPayload, IndexedRecord, ScorerBuilder, Segment, SegmentsPayload};
use map_format::{ContentHash, Manifest, ObjectKey, ObjectStore, Record, RecordKind, RecordMeta};

use crate::discover::check_resource_key;
use crate::{DensePackBuilder, PackBuilder};

/// Errors from rebuilding the cache.
#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// Includes object integrity failures.
    #[error(transparent)]
    Format(#[from] map_format::Error),
    #[error("could not decode stored payload: {0}")]
    Decode(#[from] serde_json::Error),
    #[error(transparent)]
    Stage(#[from] map_core::Error),
    #[error("index contains an unsafe resource key: {0:?}")]
    UnsafeResourceKey(String),
}

/// Build cache-pack bytes for `wanted` dimensions from the committed objects
/// under `map_dir`.
///
/// `dense` names which of the dimensions are tensor-backed (their pack is a
/// [`crate::DensePack`]); the rest are descriptor-backed ([`crate::Pack`]).
/// Returns one blob of pack bytes per built dimension, ready to write or open.
///
/// Ids are assigned per dimension in resource-then-segment order. Manifest
/// roots iterate sorted by key, which is the same order the discoverer walks,
/// so the ids here match the ones the indexer assigned — the packs come out
/// byte-identical regardless of which caller built them.
pub fn rebuild_packs(
    map_dir: &Path,
    wanted: &[String],
    dense: &BTreeSet<String>,
    verify: bool,
) -> Result<BTreeMap<String, Vec<u8>>, CacheError> {
    let manifest_path = map_dir.join("manifest.json");
    let manifest_bytes = std::fs::read(&manifest_path).map_err(|e| CacheError::Io {
        path: manifest_path,
        source: e,
    })?;
    let manifest = Manifest::from_bytes(&manifest_bytes)?;

    let shared = ObjectStore::open(map_dir.join("index/shared/objects"));
    let descriptors = ObjectStore::open(map_dir.join("index/desc/objects"));
    let tensors = ObjectStore::open(map_dir.join("index/tensor/objects"));

    let read = |store: &ObjectStore, key: ObjectKey| -> Result<Option<Vec<u8>>, CacheError> {
        let Some(entry) = manifest.object(key) else {
            return Ok(None);
        };
        let bytes = if verify {
            store.get_verified(key, entry)?
        } else {
            store.get(key)?
        };
        Ok(Some(bytes))
    };

    let mut lexical: BTreeMap<String, PackBuilder> = wanted
        .iter()
        .filter(|d| !dense.contains(*d))
        .map(|d| (d.clone(), PackBuilder::new()))
        .collect();
    let mut dense_builders: BTreeMap<String, DensePackBuilder> = wanted
        .iter()
        .filter(|d| dense.contains(*d))
        .map(|d| (d.clone(), DensePackBuilder::new()))
        .collect();
    let mut next_id: BTreeMap<String, u32> = BTreeMap::new();

    for (resource, root) in &manifest.roots {
        // A cloned repository can carry a hostile key; reject a traversing one
        // here rather than where it would be resolved to a file.
        check_resource_key(resource)
            .map_err(|_| CacheError::UnsafeResourceKey(resource.clone()))?;

        let Some(bytes) = read(&shared, root.segments)? else {
            continue;
        };
        let segments: SegmentsPayload = serde_json::from_slice(&bytes)?;

        // Descriptor objects → the lexical (BM25) packs.
        for key in root.descriptors.values() {
            let Some(bytes) = read(&descriptors, *key)? else {
                continue;
            };
            let payload: DescriptorPayload = serde_json::from_slice(&bytes)?;
            for (segment, produced) in segments.segments.iter().zip(&payload.per_segment) {
                for (dimension, record) in produced {
                    if let Some(builder) = lexical.get_mut(dimension) {
                        let id = next_id.entry(dimension.clone()).or_insert(0);
                        builder.push(&IndexedRecord {
                            id: *id,
                            resource,
                            segment: *segment,
                            record,
                        });
                        *id += 1;
                    }
                }
            }
        }

        // Tensor objects → the dense (cosine) packs.
        for key in root.tensors.values() {
            let Some(bytes) = read(&tensors, *key)? else {
                continue;
            };
            let payload = map_core::decode_tensor_payload(&bytes)?;
            for (segment, produced) in segments.segments.iter().zip(&payload.per_segment) {
                for (dimension, tensor) in produced {
                    if let Some(builder) = dense_builders.get_mut(dimension) {
                        let record = Record {
                            descriptor: None,
                            tensor: Some(tensor.clone()),
                            meta: RecordMeta {
                                kind: RecordKind::Segment,
                                dimension: dimension.clone(),
                                level: 0,
                                children: Vec::new(),
                            },
                        };
                        let id = next_id.entry(dimension.clone()).or_insert(0);
                        builder.push(&IndexedRecord {
                            id: *id,
                            resource,
                            segment: *segment,
                            record: &record,
                        });
                        *id += 1;
                    }
                }
            }
        }
    }

    // Cluster records → the same packs as their segments, appended after them.
    // A cluster carries its own descriptor, tensor, and level; the retriever
    // tells it apart from a segment by that level and nothing else, so it needs
    // no separate structure. Ids continue past the segments, staying positional.
    // Its location is its own object key — clusters span no resource, and a
    // unique per-cluster key is what keeps the query's location-keyed fusion
    // from collapsing them together.
    let clusters = ObjectStore::open(map_dir.join("index/cluster/objects"));
    for (dimension, keys) in &manifest.clusters {
        let is_dense = dense_builders.contains_key(dimension);
        let is_lexical = lexical.contains_key(dimension);
        if !is_dense && !is_lexical {
            continue;
        }
        for key in keys {
            let Some(bytes) = read(&clusters, *key)? else {
                continue;
            };
            let record: Record = serde_json::from_slice(&bytes)?;
            let hex = key.to_string();
            let id = next_id.entry(dimension.clone()).or_insert(0);
            let indexed = IndexedRecord {
                id: *id,
                resource: &hex,
                segment: Segment { start: 0, end: 0 },
                record: &record,
            };
            if is_dense {
                dense_builders.get_mut(dimension).unwrap().push(&indexed);
            } else {
                lexical.get_mut(dimension).unwrap().push(&indexed);
            }
            *id += 1;
        }
    }

    let mut out: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for (dimension, builder) in lexical {
        out.insert(dimension, builder.finish());
    }
    for (dimension, builder) in dense_builders {
        out.insert(dimension, builder.finish());
    }
    Ok(out)
}

/// Filename, in the cache dir, of the pack trust markers. Gitignored like the
/// packs and the stat snapshot it sits beside.
const PACK_MARKER: &str = "packs.manifest";

/// Whether a committed manifest exists to rebuild from.
pub fn has_manifest(map_dir: &Path) -> bool {
    map_dir.join("manifest.json").is_file()
}

/// Fingerprint of the committed manifest the cache packs must match.
///
/// The manifest enumerates every committed object — input-addressed keys and
/// their content hashes — so its bytes change whenever the indexed content does.
/// Hashing them yields a stable "which index is this?" tag. `None` when no
/// manifest is present (a bare `map init` before the first `map index`).
///
/// A reindex that changed nothing still rewrites the manifest timestamp, so a
/// pulled timestamp-only manifest can trigger one needless rebuild — cheap, and
/// far simpler than trying to hash only the content-bearing fields.
fn manifest_fingerprint(map_dir: &Path) -> Option<String> {
    let bytes = std::fs::read(map_dir.join("manifest.json")).ok()?;
    Some(ContentHash::of(&bytes).to_string())
}

/// The marker one dimension's pack must carry to be trusted:
/// `keyed_hash(machine_key, manifest_fingerprint ‖ dimension ‖ pack_bytes)`.
///
/// It asserts *this machine built these exact pack bytes for this exact
/// manifest*, and each of the three clauses buys something:
///
/// - the **manifest fingerprint**, so a pulled index that swapped the manifest
///   under a gitignored cache no longer matches (the stale-shadow bug);
/// - the **dimension name**, so a pack cannot be moved from one dimension's
///   file to another's and still validate;
/// - the **pack bytes**, because `cache/` is gitignored and git overwrites
///   ignored files on pull without a word. Binding only the manifest left the
///   pack itself unhashed, so a hostile clone that force-committed
///   `cache/lexical.pack` under an unchanged manifest had it trusted and
///   memory-mapped on the next query.
///
/// Keying with a per-machine secret is what makes the marker unforgeable: the
/// attacker ships bytes but cannot compute a marker for them.
///
/// Fields are length-prefixed so no field boundary is ambiguous — otherwise a
/// crafted dimension name could absorb the start of the pack bytes and two
/// different inputs would hash alike.
///
/// `None` when there is no manifest to bind to or no machine key to vouch with;
/// the retriever reads `None` as "cannot confirm this pack", and rebuilds from
/// the verified committed objects.
pub fn pack_marker(map_dir: &Path, dimension: &str, pack_bytes: &[u8]) -> Option<String> {
    let fingerprint = manifest_fingerprint(map_dir)?;
    let key = machine_key()?;

    let mut input = Vec::with_capacity(fingerprint.len() + dimension.len() + pack_bytes.len() + 24);
    for field in [fingerprint.as_bytes(), dimension.as_bytes(), pack_bytes] {
        input.extend_from_slice(&(field.len() as u64).to_le_bytes());
        input.extend_from_slice(field);
    }
    Some(map_format::keyed_hex(&key, &input))
}

/// The per-dimension pack markers recorded beside the cache packs.
///
/// One marker per pack rather than one per cache directory: the markers bind
/// pack bytes, so they can only be checked — and replaced — a pack at a time.
/// Serialized as `dimension marker` lines, so a `BTreeMap` keeps the file
/// byte-stable across writes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PackMarkers(BTreeMap<String, String>);

impl PackMarkers {
    /// The markers recorded in `cache_dir`.
    ///
    /// A missing or unparseable file reads as empty, which trusts nothing and
    /// costs a rebuild — the safe direction for a file that is derived,
    /// gitignored, and writable by anything that can write the cache.
    pub fn read(cache_dir: &Path) -> PackMarkers {
        let Ok(text) = std::fs::read_to_string(cache_dir.join(PACK_MARKER)) else {
            return PackMarkers::default();
        };
        let mut markers = BTreeMap::new();
        for line in text.lines() {
            let mut fields = line.split_whitespace();
            if let (Some(dimension), Some(marker), None) =
                (fields.next(), fields.next(), fields.next())
            {
                markers.insert(dimension.to_owned(), marker.to_owned());
            }
        }
        PackMarkers(markers)
    }

    /// Record these markers in `cache_dir`.
    ///
    /// Temp file then rename, so a query reading the file concurrently with a
    /// rebuild sees either the old set or the new one, never a half-written
    /// line that would silently drop a dimension's marker.
    pub fn write(&self, cache_dir: &Path) -> std::io::Result<()> {
        let mut text = String::new();
        for (dimension, marker) in &self.0 {
            text.push_str(dimension);
            text.push(' ');
            text.push_str(marker);
            text.push('\n');
        }
        let path = cache_dir.join(PACK_MARKER);
        let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
        std::fs::write(&tmp, text)?;
        std::fs::rename(&tmp, &path)
    }

    /// Whether `dimension`'s recorded marker is exactly `expected`.
    pub fn trusts(&self, dimension: &str, expected: &str) -> bool {
        self.0.get(dimension).is_some_and(|m| m == expected)
    }

    pub fn insert(&mut self, dimension: &str, marker: String) {
        self.0.insert(dimension.to_owned(), marker);
    }

    pub fn remove(&mut self, dimension: &str) {
        self.0.remove(dimension);
    }
}

/// This machine's secret cache-trust key, from `~/.map/machine.key`, generated
/// once and persisted. `None` if no home directory resolves or the key cannot be
/// established — the caller then treats packs as unverifiable and rebuilds.
///
/// Lives user-global, never inside a repository, so nothing machine-specific
/// travels with a committed index. Process-cached so every caller in one run
/// agrees even if the file is racing into existence under parallel processes.
fn machine_key() -> Option<[u8; 32]> {
    static KEY: std::sync::OnceLock<Option<[u8; 32]>> = std::sync::OnceLock::new();
    *KEY.get_or_init(load_or_create_machine_key)
}

fn load_or_create_machine_key() -> Option<[u8; 32]> {
    let home = std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(|h| PathBuf::from(h).join(".map"))?;
    let path = home.join("machine.key");

    if let Some(key) = read_key(&path) {
        return Some(key);
    }

    let mut key = [0u8; 32];
    getrandom::getrandom(&mut key).ok()?;
    std::fs::create_dir_all(&home).ok()?;

    // Atomic create-with-content: write a full temp, then hard-link it into
    // place. The final path never exists half-written, so a process that loses
    // the race still reads 32 complete bytes rather than a partial key.
    let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
    std::fs::write(&tmp, key).ok()?;
    set_owner_only(&tmp);
    let result = match std::fs::hard_link(&tmp, &path) {
        Ok(()) => Some(key),
        // Another process created it first — read theirs so we all agree.
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => read_key(&path),
        Err(_) => None,
    };
    let _ = std::fs::remove_file(&tmp);
    result
}

fn read_key(path: &Path) -> Option<[u8; 32]> {
    let bytes = std::fs::read(path).ok()?;
    <[u8; 32]>::try_from(bytes.as_slice()).ok()
}

#[cfg(unix)]
fn set_owner_only(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
}

#[cfg(not(unix))]
fn set_owner_only(_path: &Path) {}

/// Persist pack bytes to `path` via temp-and-rename.
///
/// A unique temp name per process then an atomic rename, so a concurrent query
/// never mmaps a half-written file — the immutability the pack readers assume.
/// Two processes rebuilding the same clone write byte-identical packs, so
/// last-writer-wins is harmless.
pub fn write_pack(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("cache.pack");
    let tmp = path.with_file_name(format!("{name}.tmp.{}", std::process::id()));
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A private directory that removes itself, so these tests need no
    /// dev-dependency to get a scratch path.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str) -> Scratch {
            static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!(
                "map-stages-cache-{name}-{}-{n}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Scratch(dir)
        }

        fn with_manifest(name: &str, manifest: &[u8]) -> Scratch {
            let scratch = Scratch::new(name);
            std::fs::write(scratch.0.join("manifest.json"), manifest).unwrap();
            scratch
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn marker(dir: &Scratch, dimension: &str, bytes: &[u8]) -> String {
        pack_marker(&dir.0, dimension, bytes)
            .expect("a manifest and a machine key are both available in tests")
    }

    #[test]
    fn a_pack_marker_binds_the_pack_bytes() {
        // The whole point of the redesign: git overwrites gitignored files on
        // pull, so a marker that ignored the pack bytes would vouch for a pack
        // a hostile clone substituted under an unchanged manifest.
        let dir = Scratch::with_manifest("bytes", b"{\"roots\":{}}");
        assert_ne!(
            marker(&dir, "lexical", b"MAPPACK\0original"),
            marker(&dir, "lexical", b"MAPPACK\0substituted")
        );
    }

    #[test]
    fn a_pack_marker_binds_the_manifest() {
        // A pulled index that swapped the manifest must not be trusted by the
        // cache the previous manifest left behind.
        let before = Scratch::with_manifest("manifest-a", b"{\"roots\":{}}");
        let after = Scratch::with_manifest("manifest-b", b"{\"roots\":{\"a\":1}}");
        assert_ne!(
            marker(&before, "lexical", b"pack"),
            marker(&after, "lexical", b"pack")
        );
    }

    #[test]
    fn a_pack_marker_binds_the_dimension() {
        // Otherwise one dimension's trusted pack could be copied over another's
        // file and still validate.
        let dir = Scratch::with_manifest("dimension", b"{\"roots\":{}}");
        assert_ne!(
            marker(&dir, "lexical", b"pack"),
            marker(&dir, "semantic", b"pack")
        );
    }

    #[test]
    fn length_prefixing_keeps_adjacent_fields_distinct() {
        // Concatenating without lengths would let a dimension name swallow the
        // first bytes of the pack and make two different inputs hash alike.
        let dir = Scratch::with_manifest("prefix", b"{\"roots\":{}}");
        assert_ne!(
            marker(&dir, "lex", b"icalpack"),
            marker(&dir, "lexical", b"pack")
        );
    }

    #[test]
    fn markers_round_trip_through_the_cache_file() {
        let dir = Scratch::new("roundtrip");
        let mut written = PackMarkers::default();
        written.insert("lexical", "aaaa".to_owned());
        written.insert("semantic", "bbbb".to_owned());
        written.write(&dir.0).unwrap();

        assert_eq!(PackMarkers::read(&dir.0), written);
        assert!(PackMarkers::read(&dir.0).trusts("lexical", "aaaa"));
    }

    #[test]
    fn a_removed_marker_no_longer_round_trips() {
        let dir = Scratch::new("removed");
        let mut markers = PackMarkers::default();
        markers.insert("lexical", "aaaa".to_owned());
        markers.insert("semantic", "bbbb".to_owned());
        markers.remove("semantic");
        markers.write(&dir.0).unwrap();

        let read = PackMarkers::read(&dir.0);
        assert!(read.trusts("lexical", "aaaa"));
        assert!(!read.trusts("semantic", "bbbb"));
    }

    #[test]
    fn a_missing_or_unparseable_file_trusts_nothing() {
        // Failing closed costs a rebuild; failing open maps an unvouched pack.
        let dir = Scratch::new("missing");
        assert_eq!(PackMarkers::read(&dir.0), PackMarkers::default());

        std::fs::write(dir.0.join(PACK_MARKER), "garbage with too many fields\n").unwrap();
        assert_eq!(PackMarkers::read(&dir.0), PackMarkers::default());
    }

    #[test]
    fn an_unrecorded_dimension_is_not_trusted() {
        let dir = Scratch::new("unknown");
        let mut markers = PackMarkers::default();
        markers.insert("lexical", "aaaa".to_owned());
        markers.write(&dir.0).unwrap();

        let read = PackMarkers::read(&dir.0);
        assert!(!read.trusts("semantic", "aaaa"));
        assert!(!read.trusts("lexical", "bbbb"));
    }

    #[test]
    fn a_pack_marker_is_none_without_a_manifest() {
        // A bare `map init` has nothing to bind to; the caller reads that as
        // "cannot confirm", not as "trusted".
        let dir = Scratch::new("no-manifest");
        assert!(pack_marker(&dir.0, "lexical", b"pack").is_none());
    }
}
