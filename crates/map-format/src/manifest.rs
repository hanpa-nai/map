//! `.map/manifest.json` — the index's table of contents and its only
//! integrity mechanism.
//!
//! Object keys are input-addressed, so stored bytes cannot be verified by
//! hashing them (spec §4). The manifest therefore records a content hash per
//! object. Everything that depends on knowing whether an index is intact —
//! integrity checking on read, collision detection, merge resolution — reads it
//! from here.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::codec::canonical_json;
use crate::config::Config;
use crate::error::{Error, Result};
use crate::hash::{ContentHash, Digest, Fingerprint, ObjectKey, HASH_ALGO};

/// Current manifest schema version.
pub const MANIFEST_VERSION: u32 = 1;

/// Which determinism guarantee an object falls under (spec §6).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    /// Bit-identical everywhere, including across platforms.
    A,
    /// Producer-authoritative. Consumers verify; they never re-derive.
    B,
    /// Authored. No reproducibility expected; provenance is the control.
    C,
}

/// What the manifest knows about one stored object.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObjectEntry {
    /// Hash of the stored bytes. The only way to detect tampering.
    pub content_hash: ContentHash,
    pub len: u64,
    pub tier: Tier,
}

/// Identity of a dimension as `name + fingerprint` (spec §5).
///
/// Federation compares this, never the bare name, so two `descriptive`
/// dimensions built with different prompts or embedders stay distinct instead
/// of being silently fused into incomparable scores.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DimensionIdentity {
    /// Artifact fingerprint — see
    /// [`DimensionConfig::artifact_fingerprint`](crate::config::DimensionConfig::artifact_fingerprint).
    ///
    /// Deliberately *not* the full-config fingerprint: editing a dimension's
    /// model-facing `description` must not make two indices incomparable.
    pub fingerprint: Fingerprint,
    pub descriptor: bool,
    pub tensor: bool,
    /// Fabric levels this dimension actually holds records at, ascending.
    ///
    /// Config declares what a dimension *searches*; this records what was
    /// *built*, so the two can be compared at load. Without it, lowering
    /// `max_levels` or dropping a fabricator makes every cluster the LLM was
    /// billed for go quiet — still on disk, still referenced here, never
    /// returned, and nothing says so.
    ///
    /// Empty means "unknown": a manifest written before this field existed.
    /// That reads as no information rather than as "level 0 only", so an old
    /// index opens instead of failing closed on a fact it never recorded.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub levels: Vec<u16>,
}

/// Entry point into the object graph for one resource.
///
/// The spec calls these "object roots" (§5). Without them the object table is
/// a flat bag of hashes with no way in: the retriever cannot enumerate what to
/// search, and `map gc` cannot trace what is still reachable.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceRoot {
    /// Object holding this resource's segment spans.
    pub segments: ObjectKey,
    /// Descriptor-group objects, keyed by the classifier implementation that
    /// produced them.
    ///
    /// Keyed by implementation rather than by dimension because that is the
    /// invocation unit: one classifier call covering several dimensions writes
    /// one object.
    #[serde(default)]
    pub descriptors: BTreeMap<String, ObjectKey>,
    /// Tensor-group objects, keyed by the embedder implementation that produced
    /// them — the dense analog of `descriptors`. One embed call covering
    /// several dimensions writes one object.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub tensors: BTreeMap<String, ObjectKey>,
}

/// Who produced this index and with what.
///
/// **Tier C.** Contains a wall-clock timestamp and therefore is not
/// byte-reproducible. It is excluded from
/// [`Manifest::determinism_digest`] for exactly that reason.
///
/// Per-stage identity — implementation, model, revision — is deliberately not
/// here. It was declared and never written, and a provenance record no producer
/// fills is indistinguishable from one that had nothing to report. It belongs
/// back only alongside a consumer that reads it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    pub map_version: String,
    /// Unix seconds.
    pub generated_at: u64,
}

/// The index manifest.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub format_version: u32,

    /// Hash algorithm used for every digest in this index.
    ///
    /// Recorded rather than assumed, so the choice stays changeable while the
    /// format is DRAFT.
    pub hash_algo: String,

    /// Fingerprint of the config this index was built from.
    pub config_fingerprint: Fingerprint,

    /// Dimension identities, keyed by name.
    pub dimensions: BTreeMap<String, DimensionIdentity>,

    /// Every stored object, keyed by hex object key.
    ///
    /// `BTreeMap` so serialization order is canonical.
    pub objects: BTreeMap<String, ObjectEntry>,

    /// Entry points into the object graph, keyed by canonical resource key.
    ///
    /// This is what makes the object table traversable — see [`ResourceRoot`].
    #[serde(default)]
    pub roots: BTreeMap<String, ResourceRoot>,

    /// Cluster records the fabricator emitted, keyed by dimension.
    ///
    /// Clusters are not owned by any one resource — they group records across
    /// the whole corpus — so they hang off the manifest rather than off a
    /// [`ResourceRoot`]. Stored sorted so the manifest serializes identically
    /// regardless of the order fabrication emitted them.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub clusters: BTreeMap<String, Vec<ObjectKey>>,

    /// Producer identity. Tier C; excluded from the determinism digest.
    pub provenance: Provenance,
}

impl Manifest {
    pub fn new(config: &Config, map_version: impl Into<String>, generated_at: u64) -> Result<Self> {
        let mut dimensions = BTreeMap::new();
        for (name, dim) in config.active() {
            dimensions.insert(
                name.clone(),
                DimensionIdentity {
                    fingerprint: dim.artifact_fingerprint()?,
                    descriptor: dim.emits_descriptor(),
                    tensor: dim.emits_tensor(),
                    // Every dimension holds its segments; the fabricator raises
                    // this as it writes each level.
                    levels: vec![0],
                },
            );
        }

        Ok(Manifest {
            format_version: MANIFEST_VERSION,
            hash_algo: HASH_ALGO.to_owned(),
            config_fingerprint: config.fingerprint()?,
            dimensions,
            objects: BTreeMap::new(),
            roots: BTreeMap::new(),
            clusters: BTreeMap::new(),
            provenance: Provenance {
                map_version: map_version.into(),
                generated_at,
            },
        })
    }

    /// Record an object, rejecting a same-key/different-bytes collision.
    ///
    /// Under input-addressing two producers can legitimately derive the same
    /// key from the same inputs while writing different bytes — nondeterministic
    /// stages guarantee it. Union is not a valid resolution, so this surfaces
    /// the collision instead of silently keeping one side (spec §4).
    pub fn insert_object(&mut self, key: ObjectKey, entry: ObjectEntry) -> Result<()> {
        let hex = key.to_string();
        if let Some(existing) = self.objects.get(&hex) {
            if existing.content_hash != entry.content_hash {
                return Err(Error::KeyCollision {
                    key: hex,
                    a: existing.content_hash.to_string(),
                    b: entry.content_hash.to_string(),
                });
            }
            return Ok(());
        }
        self.objects.insert(hex, entry);
        Ok(())
    }

    /// Look up an object entry.
    pub fn object(&self, key: ObjectKey) -> Option<&ObjectEntry> {
        self.objects.get(&key.to_string())
    }

    /// Digest over everything that is expected to be reproducible.
    ///
    /// Excludes [`Provenance`], which carries a wall clock. This is what the
    /// cross-platform Tier A comparison in CI compares.
    pub fn determinism_digest(&self) -> Result<Digest> {
        #[derive(Serialize)]
        struct Reproducible<'a> {
            format_version: u32,
            hash_algo: &'a str,
            config_fingerprint: &'a Fingerprint,
            dimensions: &'a BTreeMap<String, DimensionIdentity>,
            objects: &'a BTreeMap<String, ObjectEntry>,
            roots: &'a BTreeMap<String, ResourceRoot>,
            clusters: &'a BTreeMap<String, Vec<ObjectKey>>,
        }

        let view = Reproducible {
            format_version: self.format_version,
            hash_algo: &self.hash_algo,
            config_fingerprint: &self.config_fingerprint,
            dimensions: &self.dimensions,
            objects: &self.objects,
            roots: &self.roots,
            clusters: &self.clusters,
        };
        Ok(ContentHash::of(&canonical_json(&view)?).0)
    }

    /// Serialize canonically, one entry per line.
    ///
    /// Still exactly one byte string per manifest — sorted map keys,
    /// declaration-ordered struct fields, compact values — so Tier A holds.
    /// What changes is the layout: every top-level key, and every entry of
    /// `dimensions`, `objects`, `roots` and `clusters`, gets its own line.
    ///
    /// Why: emitted as one line, a ripgrep-sized manifest is 265 KB of text
    /// that git can only treat as a single atom. Two people re-indexing
    /// disjoint parts of a repository then produce a conflict on that one line
    /// every time, and the review diff says nothing beyond "it changed". One
    /// entry per line makes disjoint edits merge textually and makes the diff
    /// name the objects that actually moved. [`Manifest::merge`] is for what
    /// the textual merge cannot resolve.
    ///
    /// [`Manifest::from_bytes`] needs no change — this is ordinary JSON — and
    /// neither does [`Manifest::determinism_digest`], which hashes its own
    /// canonical view rather than these bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        /// One `"key": <compact value>` entry per line, in sorted key order.
        fn block<V: Serialize>(
            out: &mut Vec<u8>,
            name: &str,
            map: &BTreeMap<String, V>,
        ) -> Result<()> {
            out.extend_from_slice(b"  ");
            out.extend_from_slice(&canonical_json(&name)?);
            out.extend_from_slice(b": ");
            if map.is_empty() {
                out.extend_from_slice(b"{}");
                return Ok(());
            }
            out.extend_from_slice(b"{\n");
            for (i, (key, value)) in map.iter().enumerate() {
                if i > 0 {
                    out.extend_from_slice(b",\n");
                }
                out.extend_from_slice(b"    ");
                out.extend_from_slice(&canonical_json(key)?);
                out.extend_from_slice(b": ");
                out.extend_from_slice(&canonical_json(value)?);
            }
            out.extend_from_slice(b"\n  }");
            Ok(())
        }

        fn scalar<V: Serialize>(out: &mut Vec<u8>, name: &str, value: &V) -> Result<()> {
            out.extend_from_slice(b"  ");
            out.extend_from_slice(&canonical_json(&name)?);
            out.extend_from_slice(b": ");
            out.extend_from_slice(&canonical_json(value)?);
            Ok(())
        }

        // Key order mirrors the struct's declaration order, so the hand-written
        // writer and the derived `Serialize` describe the same document.
        let mut out = Vec::new();
        out.extend_from_slice(b"{\n");
        scalar(&mut out, "format_version", &self.format_version)?;
        out.extend_from_slice(b",\n");
        scalar(&mut out, "hash_algo", &self.hash_algo)?;
        out.extend_from_slice(b",\n");
        scalar(&mut out, "config_fingerprint", &self.config_fingerprint)?;
        out.extend_from_slice(b",\n");
        block(&mut out, "dimensions", &self.dimensions)?;
        out.extend_from_slice(b",\n");
        block(&mut out, "objects", &self.objects)?;
        out.extend_from_slice(b",\n");
        block(&mut out, "roots", &self.roots)?;
        out.extend_from_slice(b",\n");
        // Omitted when empty, matching the field's `skip_serializing_if`.
        if !self.clusters.is_empty() {
            block(&mut out, "clusters", &self.clusters)?;
            out.extend_from_slice(b",\n");
        }
        scalar(&mut out, "provenance", &self.provenance)?;
        out.extend_from_slice(b"\n}\n");
        Ok(out)
    }

    /// Parse from canonical JSON, rejecting anything this build cannot read.
    ///
    /// Without these checks an older binary would silently misinterpret a
    /// newer index — reading fields that moved and ignoring ones it does not
    /// know — instead of saying "this index needs a newer map".
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let manifest: Manifest = serde_json::from_slice(bytes)?;

        if manifest.format_version != MANIFEST_VERSION {
            return Err(Error::UnsupportedVersion {
                kind: "manifest",
                found: manifest.format_version,
                expected: MANIFEST_VERSION,
            });
        }
        if manifest.hash_algo != HASH_ALGO {
            return Err(Error::UnsupportedHashAlgo {
                found: manifest.hash_algo.clone(),
                expected: HASH_ALGO,
            });
        }
        Ok(manifest)
    }

    /// Three-way merge of two manifests against their common ancestor.
    ///
    /// Line-oriented [`Manifest::to_bytes`] lets git merge the disjoint case on
    /// its own. This is for the rest: a textual merge cannot tell a legitimate
    /// union of object tables from a same-key/different-bytes collision, and it
    /// cannot know that two re-fabricated cluster trees describe neither side's
    /// object set.
    ///
    /// "Three-way" on a single value means: equal on both sides, take it;
    /// unchanged on our side, take theirs; unchanged on theirs, take ours;
    /// otherwise a conflict. A key missing on one side is a deletion when
    /// `base` had it and the other side left it alone, and an addition when
    /// `base` lacked it. `base` of `None` is an empty ancestor.
    ///
    /// Every conflict is collected — resolving a manifest one rejected merge at
    /// a time is worse than seeing the whole list.
    pub fn merge(
        base: Option<&Manifest>,
        ours: &Manifest,
        theirs: &Manifest,
    ) -> std::result::Result<Merged, Vec<MergeConflict>> {
        let mut conflicts = Vec::new();
        let mut notes = Vec::new();

        // Not three-way: a manifest this build cannot read on one side is not
        // something to resolve field by field.
        if ours.format_version != theirs.format_version || ours.hash_algo != theirs.hash_algo {
            conflicts.push(MergeConflict::Format);
        }

        let config_fingerprint = match three_way(
            base.map(|b| &b.config_fingerprint),
            Some(&ours.config_fingerprint),
            Some(&theirs.config_fingerprint),
        )
        .flatten()
        {
            Some(fingerprint) => fingerprint,
            None => {
                conflicts.push(MergeConflict::ConfigFingerprint);
                ours.config_fingerprint
            }
        };

        let mut dimensions = BTreeMap::new();
        for name in union_keys(
            base.map(|b| &b.dimensions),
            &ours.dimensions,
            &theirs.dimensions,
        ) {
            match three_way(
                base.and_then(|b| b.dimensions.get(name)),
                ours.dimensions.get(name),
                theirs.dimensions.get(name),
            ) {
                Some(Some(identity)) => {
                    dimensions.insert(name.clone(), identity);
                }
                Some(None) => {}
                None => conflicts.push(MergeConflict::Dimension { name: name.clone() }),
            }
        }

        // Union, not three-way: objects are immutable and input-addressed, so
        // two sides adding different objects is the ordinary case. The same key
        // with different bytes is the one thing union cannot express (spec §4).
        let mut objects = ours.objects.clone();
        for (key, theirs_entry) in &theirs.objects {
            match objects.get(key) {
                Some(ours_entry) if ours_entry.content_hash != theirs_entry.content_hash => {
                    conflicts.push(MergeConflict::Object {
                        key: key.clone(),
                        ours: ours_entry.content_hash,
                        theirs: theirs_entry.content_hash,
                    });
                }
                Some(_) => {}
                None => {
                    objects.insert(key.clone(), theirs_entry.clone());
                }
            }
        }

        let mut roots = BTreeMap::new();
        for resource in union_keys(base.map(|b| &b.roots), &ours.roots, &theirs.roots) {
            match three_way(
                base.and_then(|b| b.roots.get(resource)),
                ours.roots.get(resource),
                theirs.roots.get(resource),
            ) {
                Some(Some(root)) => {
                    roots.insert(resource.clone(), root);
                }
                Some(None) => {}
                None => conflicts.push(MergeConflict::Root {
                    resource: resource.clone(),
                }),
            }
        }

        // Runs after `dimensions` because dropping a tree rewrites the built
        // levels that describe it.
        let mut clusters = BTreeMap::new();
        for name in union_keys(base.map(|b| &b.clusters), &ours.clusters, &theirs.clusters) {
            match three_way(
                base.and_then(|b| b.clusters.get(name)),
                ours.clusters.get(name),
                theirs.clusters.get(name),
            ) {
                Some(Some(tree)) => {
                    clusters.insert(name.clone(), tree);
                }
                Some(None) => {}
                None => {
                    // Deliberately not a conflict. Clusters are derived from the
                    // union of the object set, so when both sides re-fabricated,
                    // neither tree describes what the merge produced and picking
                    // one would be wrong rather than merely arbitrary. The
                    // cluster objects stay on disk, so re-fabricating reuses
                    // every unchanged subtree.
                    if let Some(identity) = dimensions.get_mut(name) {
                        identity.levels = vec![0];
                    }
                    notes.push(format!(
                        "clusters for dimension {name:?} were dropped: both sides re-fabricated \
                         and neither tree describes the merged object set. Run `map index` to \
                         rebuild it; unchanged subtrees are reused."
                    ));
                }
            }
        }

        // Tier C and wall-clock; the later run is the one that describes the
        // objects, and its version is what produced them.
        let provenance = if theirs.provenance.generated_at > ours.provenance.generated_at {
            theirs.provenance.clone()
        } else {
            ours.provenance.clone()
        };

        if !conflicts.is_empty() {
            return Err(conflicts);
        }

        Ok(Merged {
            manifest: Manifest {
                format_version: ours.format_version,
                hash_algo: ours.hash_algo.clone(),
                config_fingerprint,
                dimensions,
                objects,
                roots,
                clusters,
                provenance,
            },
            notes,
        })
    }
}

/// A merge that resolved, plus anything the resolution silently changed.
///
/// `notes` is not decoration: a dropped cluster tree leaves the index queryable
/// but flat, and nothing else would say so.
#[derive(Clone, Debug)]
pub struct Merged {
    pub manifest: Manifest,
    pub notes: Vec<String>,
}

/// Something [`Manifest::merge`] refuses to decide on the operator's behalf.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MergeConflict {
    /// One key, two different byte strings. Union is not a valid resolution
    /// (spec §4) and neither side is authoritative.
    Object {
        key: String,
        ours: ContentHash,
        theirs: ContentHash,
    },
    /// Both sides re-indexed the same resource into different objects.
    Root { resource: String },
    /// Both sides changed a dimension's identity — its fingerprint, what it
    /// emits, or the levels it holds.
    Dimension { name: String },
    /// Both sides built from a different `config.toml`.
    ConfigFingerprint,
    /// Different format versions or hash algorithms; not a field-by-field
    /// question.
    Format,
}

impl std::fmt::Display for MergeConflict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MergeConflict::Object { key, ours, theirs } => write!(
                f,
                "object {key}: ours holds {ours}, theirs holds {theirs}; \
                 same key, different content, and union is not a valid resolution"
            ),
            MergeConflict::Root { resource } => write!(
                f,
                "resource {resource:?}: both sides re-indexed it into different objects"
            ),
            MergeConflict::Dimension { name } => {
                write!(f, "dimension {name:?}: both sides changed its identity")
            }
            MergeConflict::ConfigFingerprint => f.write_str(
                "both sides built from a different config.toml; merge the config first, \
                 then re-index",
            ),
            MergeConflict::Format => {
                f.write_str("the two manifests disagree on format version or hash algorithm")
            }
        }
    }
}

/// Resolve one value three ways. `None` is a conflict; `Some(None)` is a
/// resolved deletion.
fn three_way<T: Clone + PartialEq>(
    base: Option<&T>,
    ours: Option<&T>,
    theirs: Option<&T>,
) -> Option<Option<T>> {
    if ours == theirs {
        Some(ours.cloned())
    } else if ours == base {
        Some(theirs.cloned())
    } else if theirs == base {
        Some(ours.cloned())
    } else {
        None
    }
}

/// Every key any of the three sides holds, in canonical order.
fn union_keys<'a, V>(
    base: Option<&'a BTreeMap<String, V>>,
    ours: &'a BTreeMap<String, V>,
    theirs: &'a BTreeMap<String, V>,
) -> BTreeSet<&'a String> {
    base.into_iter()
        .flat_map(|b| b.keys())
        .chain(ours.keys())
        .chain(theirs.keys())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::Fingerprint;

    fn manifest() -> Manifest {
        Manifest::new(&Config::zero_config(), "0.0.0", 1_700_000_000).unwrap()
    }

    fn entry(bytes: &[u8], tier: Tier) -> ObjectEntry {
        ObjectEntry {
            content_hash: ContentHash::of(bytes),
            len: bytes.len() as u64,
            tier,
        }
    }

    fn key(seed: &[u8]) -> ObjectKey {
        ObjectKey::derive(&[seed], Fingerprint::of(b"cfg"))
    }

    #[test]
    fn roundtrips_through_json() {
        let m = manifest();
        assert_eq!(Manifest::from_bytes(&m.to_bytes().unwrap()).unwrap(), m);
    }

    #[test]
    fn records_dimension_identity_not_just_name() {
        let m = manifest();
        let lex = &m.dimensions["lexical"];
        assert!(lex.descriptor);
        assert!(!lex.tensor);
        // Fingerprint must match what the config computes independently.
        let cfg = Config::zero_config();
        assert_eq!(
            lex.fingerprint,
            cfg.dimensions["lexical"].artifact_fingerprint().unwrap()
        );
        // Every dimension holds its segments before any fabricator runs.
        assert_eq!(lex.levels, vec![0]);
    }

    #[test]
    fn built_levels_are_inside_the_determinism_digest() {
        // The digest view serializes DimensionIdentity whole, so this holds for
        // free — and would break silently if anyone projected the fields by
        // hand. Two indices differing only in tree height are not reproducible
        // copies of one another.
        let mut flat = manifest();
        let mut tall = manifest();
        tall.dimensions.get_mut("lexical").unwrap().levels = vec![0, 1, 2];
        assert_ne!(
            flat.determinism_digest().unwrap(),
            tall.determinism_digest().unwrap()
        );
        // And the check can fail: identical levels must agree.
        flat.dimensions.get_mut("lexical").unwrap().levels = vec![0, 1, 2];
        assert_eq!(
            flat.determinism_digest().unwrap(),
            tall.determinism_digest().unwrap()
        );
    }

    #[test]
    fn reinserting_identical_bytes_is_idempotent() {
        let mut m = manifest();
        let k = key(b"a");
        m.insert_object(k, entry(b"same", Tier::A)).unwrap();
        m.insert_object(k, entry(b"same", Tier::A)).unwrap();
        assert_eq!(m.objects.len(), 1);
    }

    #[test]
    fn same_key_different_bytes_is_a_collision_not_a_union() {
        // The exact case two branches editing the same file produce.
        let mut m = manifest();
        let k = key(b"a");
        m.insert_object(k, entry(b"branch-one", Tier::C)).unwrap();
        let err = m
            .insert_object(k, entry(b"branch-two", Tier::C))
            .unwrap_err();
        assert!(matches!(err, Error::KeyCollision { .. }));
    }

    #[test]
    fn determinism_digest_ignores_provenance() {
        // Two runs a day apart, same content, must agree.
        let a = Manifest::new(&Config::zero_config(), "0.0.0", 1_700_000_000).unwrap();
        let b = Manifest::new(&Config::zero_config(), "0.0.0", 1_700_086_400).unwrap();
        assert_ne!(a.provenance, b.provenance);
        assert_eq!(
            a.determinism_digest().unwrap(),
            b.determinism_digest().unwrap()
        );
    }

    #[test]
    fn determinism_digest_tracks_objects() {
        let a = manifest();
        let mut b = a.clone();
        b.insert_object(key(b"x"), entry(b"content", Tier::A))
            .unwrap();
        assert_ne!(
            a.determinism_digest().unwrap(),
            b.determinism_digest().unwrap()
        );
    }

    #[test]
    fn future_format_version_is_rejected() {
        let mut m = manifest();
        m.format_version = MANIFEST_VERSION + 1;
        let err = Manifest::from_bytes(&m.to_bytes().unwrap()).unwrap_err();
        assert!(matches!(
            err,
            Error::UnsupportedVersion {
                kind: "manifest",
                ..
            }
        ));
    }

    #[test]
    fn foreign_hash_algorithm_is_rejected() {
        let mut m = manifest();
        m.hash_algo = "md5".into();
        let err = Manifest::from_bytes(&m.to_bytes().unwrap()).unwrap_err();
        assert!(matches!(err, Error::UnsupportedHashAlgo { .. }));
    }

    /// A manifest with something in every section, so layout tests exercise
    /// more than the empty case.
    fn populated() -> Manifest {
        let mut m = manifest();
        for i in 0..4u8 {
            m.insert_object(key(&[i]), entry(&[i], Tier::A)).unwrap();
        }
        m.roots.insert(
            "src/lib.rs".to_owned(),
            ResourceRoot {
                segments: key(b"segments"),
                descriptors: BTreeMap::from([("structural".to_owned(), key(b"desc"))]),
                tensors: BTreeMap::new(),
            },
        );
        m.clusters
            .insert("lexical".to_owned(), vec![key(b"c1"), key(b"c2")]);
        m
    }

    #[test]
    fn every_object_entry_gets_a_line_to_itself() {
        // One 265 KB line makes every concurrent re-index a git conflict and
        // every review diff unreadable.
        let m = populated();
        let text = String::from_utf8(m.to_bytes().unwrap()).unwrap();
        let keys: Vec<&str> = m.objects.keys().map(String::as_str).collect();

        for line in text.lines() {
            let on_this_line = keys.iter().filter(|k| line.contains(**k)).count();
            assert!(
                on_this_line <= 1,
                "two object keys share a line, so a disjoint edit cannot merge: {line}"
            );
        }
        for k in &keys {
            let lines = text.lines().filter(|l| l.contains(k)).count();
            assert_eq!(lines, 1, "object {k} appears on {lines} lines");
        }
        assert!(!text.contains('\r'), "line endings must be LF");
        assert!(text.ends_with("}\n"), "must end with a trailing newline");
    }

    #[test]
    fn every_section_entry_gets_a_line_to_itself() {
        let m = populated();
        let text = String::from_utf8(m.to_bytes().unwrap()).unwrap();
        for section in ["dimensions", "objects", "roots", "clusters", "provenance"] {
            let lines = text
                .lines()
                .filter(|l| l.trim_start().starts_with(&format!("\"{section}\"")))
                .count();
            assert_eq!(lines, 1, "{section} does not start its own line");
        }
        assert!(text.contains("\n    \"src/lib.rs\": {\"segments\":"));
        assert!(text.contains("\n    \"lexical\": ["));
    }

    #[test]
    fn the_line_oriented_layout_still_parses() {
        let m = populated();
        assert_eq!(Manifest::from_bytes(&m.to_bytes().unwrap()).unwrap(), m);
    }

    #[test]
    fn the_determinism_digest_does_not_depend_on_the_serialized_layout() {
        // Built independently rather than compared against a stored hex string:
        // the digest hashes its own canonical view, so reshaping `to_bytes`
        // must not move it, and a hard-coded constant would only prove that
        // *some* bytes stayed the same.
        #[derive(Serialize)]
        struct Reproducible<'a> {
            format_version: u32,
            hash_algo: &'a str,
            config_fingerprint: &'a Fingerprint,
            dimensions: &'a BTreeMap<String, DimensionIdentity>,
            objects: &'a BTreeMap<String, ObjectEntry>,
            roots: &'a BTreeMap<String, ResourceRoot>,
            clusters: &'a BTreeMap<String, Vec<ObjectKey>>,
        }

        let m = populated();
        let view = Reproducible {
            format_version: m.format_version,
            hash_algo: &m.hash_algo,
            config_fingerprint: &m.config_fingerprint,
            dimensions: &m.dimensions,
            objects: &m.objects,
            roots: &m.roots,
            clusters: &m.clusters,
        };
        assert_eq!(
            m.determinism_digest().unwrap(),
            ContentHash::of(&canonical_json(&view).unwrap()).0
        );
    }

    fn root(seed: &[u8]) -> ResourceRoot {
        ResourceRoot {
            segments: key(seed),
            descriptors: BTreeMap::new(),
            tensors: BTreeMap::new(),
        }
    }

    #[test]
    fn disjoint_edits_merge_into_the_union() {
        let base = manifest();

        let mut ours = base.clone();
        ours.roots.insert("a.rs".to_owned(), root(b"a"));
        ours.insert_object(key(b"a"), entry(b"a", Tier::A)).unwrap();

        let mut theirs = base.clone();
        theirs.roots.insert("b.rs".to_owned(), root(b"b"));
        theirs
            .insert_object(key(b"b"), entry(b"b", Tier::A))
            .unwrap();

        let merged = Manifest::merge(Some(&base), &ours, &theirs).unwrap();
        assert_eq!(
            merged.manifest.roots.keys().collect::<Vec<_>>(),
            vec!["a.rs", "b.rs"]
        );
        assert!(merged.manifest.object(key(b"a")).is_some());
        assert!(merged.manifest.object(key(b"b")).is_some());
        assert!(merged.notes.is_empty());
    }

    #[test]
    fn both_sides_reindexing_one_resource_differently_conflicts() {
        let mut base = manifest();
        base.roots.insert("a.rs".to_owned(), root(b"original"));

        let mut ours = base.clone();
        ours.roots.insert("a.rs".to_owned(), root(b"ours"));
        let mut theirs = base.clone();
        theirs.roots.insert("a.rs".to_owned(), root(b"theirs"));

        let conflicts = Manifest::merge(Some(&base), &ours, &theirs).unwrap_err();
        assert_eq!(
            conflicts,
            vec![MergeConflict::Root {
                resource: "a.rs".to_owned()
            }]
        );
    }

    #[test]
    fn one_key_with_two_contents_is_never_unioned() {
        // Union is not a valid resolution under input-addressing (spec §4).
        let base = manifest();
        let k = key(b"contested");

        let mut ours = base.clone();
        ours.insert_object(k, entry(b"ours", Tier::C)).unwrap();
        let mut theirs = base.clone();
        theirs.insert_object(k, entry(b"theirs", Tier::C)).unwrap();

        let conflicts = Manifest::merge(Some(&base), &ours, &theirs).unwrap_err();
        assert!(
            matches!(conflicts.as_slice(), [MergeConflict::Object { key, .. }] if *key == k.to_string()),
            "{conflicts:?}"
        );
        assert!(conflicts[0].to_string().contains("union is not a valid"));
    }

    #[test]
    fn two_refabricated_trees_are_dropped_rather_than_conflicting() {
        // Neither tree describes the merged object set, so picking one would be
        // wrong rather than arbitrary. The cluster objects stay on disk, so
        // re-fabrication reuses every unchanged subtree.
        let mut base = manifest();
        base.clusters.insert("lexical".to_owned(), vec![key(b"c0")]);

        let mut ours = base.clone();
        ours.clusters.insert("lexical".to_owned(), vec![key(b"c1")]);
        ours.dimensions.get_mut("lexical").unwrap().levels = vec![0, 1];

        let mut theirs = base.clone();
        theirs
            .clusters
            .insert("lexical".to_owned(), vec![key(b"c2")]);

        let merged = Manifest::merge(Some(&base), &ours, &theirs).unwrap();
        assert!(!merged.manifest.clusters.contains_key("lexical"));
        assert_eq!(merged.manifest.dimensions["lexical"].levels, vec![0]);
        assert_eq!(merged.notes.len(), 1);
        assert!(merged.notes[0].contains("map index"), "{:?}", merged.notes);
    }

    #[test]
    fn provenance_takes_the_later_run() {
        let base = manifest();
        let mut ours = base.clone();
        ours.provenance = Provenance {
            map_version: "0.1.0".to_owned(),
            generated_at: 100,
        };
        let mut theirs = base.clone();
        theirs.provenance = Provenance {
            map_version: "0.2.0".to_owned(),
            generated_at: 200,
        };

        let merged = Manifest::merge(Some(&base), &ours, &theirs).unwrap();
        assert_eq!(merged.manifest.provenance, theirs.provenance);
        // And symmetrically, so the result does not depend on argument order.
        let merged = Manifest::merge(Some(&base), &theirs, &ours).unwrap();
        assert_eq!(merged.manifest.provenance, theirs.provenance);
    }

    #[test]
    fn a_resource_one_side_deleted_stays_deleted() {
        let mut base = manifest();
        base.roots.insert("gone.rs".to_owned(), root(b"gone"));
        base.roots.insert("kept.rs".to_owned(), root(b"kept"));

        let mut ours = base.clone();
        ours.roots.remove("gone.rs");
        let theirs = base.clone();

        let merged = Manifest::merge(Some(&base), &ours, &theirs).unwrap();
        assert_eq!(
            merged.manifest.roots.keys().collect::<Vec<_>>(),
            ["kept.rs"]
        );
    }

    #[test]
    fn a_missing_ancestor_merges_as_an_empty_one() {
        let mut ours = manifest();
        ours.roots.insert("a.rs".to_owned(), root(b"a"));
        let mut theirs = manifest();
        theirs.roots.insert("b.rs".to_owned(), root(b"b"));

        let merged = Manifest::merge(None, &ours, &theirs).unwrap();
        assert_eq!(
            merged.manifest.roots.keys().collect::<Vec<_>>(),
            vec!["a.rs", "b.rs"]
        );
    }

    #[test]
    fn every_conflict_is_reported_not_just_the_first() {
        let mut base = manifest();
        base.roots.insert("a.rs".to_owned(), root(b"original"));
        base.roots.insert("b.rs".to_owned(), root(b"original"));

        let mut ours = base.clone();
        let mut theirs = base.clone();
        for (side, seed) in [(&mut ours, b"ours"), (&mut theirs, b"thrs")] {
            side.roots.insert("a.rs".to_owned(), root(seed));
            side.roots.insert("b.rs".to_owned(), root(seed));
            side.config_fingerprint = Fingerprint::of(seed);
        }

        let conflicts = Manifest::merge(Some(&base), &ours, &theirs).unwrap_err();
        assert_eq!(conflicts.len(), 3, "{conflicts:?}");
        assert!(conflicts.contains(&MergeConflict::ConfigFingerprint));
    }

    #[test]
    fn a_foreign_format_is_not_merged_field_by_field() {
        let ours = manifest();
        let mut theirs = manifest();
        theirs.hash_algo = "sha256".to_owned();
        let conflicts = Manifest::merge(None, &ours, &theirs).unwrap_err();
        assert!(conflicts.contains(&MergeConflict::Format));
    }

    #[test]
    fn a_manifest_carrying_fields_this_build_dropped_still_opens() {
        // Older indexes recorded `state`, `score_stats`, `partition`, and
        // `attestation`. None of them was ever populated by a producer, so they
        // were removed rather than implemented — and an index written before
        // that must still load rather than fail on fields it legitimately holds.
        let m = manifest();
        let mut value: serde_json::Value = serde_json::from_slice(&m.to_bytes().unwrap()).unwrap();
        let object = value.as_object_mut().unwrap();
        object.insert("state".into(), serde_json::json!({ "status": "complete" }));
        object.insert("score_stats".into(), serde_json::json!({}));
        object.insert("partition".into(), serde_json::json!({ "scheme": "none" }));
        object.insert(
            "attestation".into(),
            serde_json::json!({ "scheme": "ed25519", "key_id": "abc", "signature": "sig" }),
        );

        let bytes = serde_json::to_vec(&value).unwrap();
        let back = Manifest::from_bytes(&bytes).unwrap();
        assert_eq!(
            back.determinism_digest().unwrap(),
            m.determinism_digest().unwrap()
        );
    }
}
