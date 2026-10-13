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
pub const MANIFEST_VERSION: u32 = 2;

/// The layout before: one object table for the whole index, sorted by key.
///
/// Still read, so an index that an earlier version made opens. Still written
/// for a manifest that was read in it, so a merge by this build does not hand
/// a team on the earlier version a file that their binary refuses. An index
/// build writes [`MANIFEST_VERSION`].
const TABLE_LAYOUT: u32 = 1;

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
    /// The version of the run that last changed the index. A later run that
    /// finds nothing to change leaves the two fields as they are.
    pub map_version: String,
    /// Unix seconds, of that same run. Only the table layout has it, and it
    /// reads as 0 from a manifest of this version.
    ///
    /// Not written any more because of what it did to a merge: two branches
    /// that each changed the index wrote two times, so the line was a conflict
    /// in every text merge, and a git server has no other kind. The commit
    /// that holds the manifest already says when.
    #[serde(default)]
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

    /// Every object key that a root or a cluster tree names.
    ///
    /// An index build records an object only when one of these names it, so
    /// for a manifest that a build wrote this is the whole object table. An
    /// entry outside this set is one that no query can reach.
    pub fn named_keys(&self) -> BTreeSet<String> {
        let roots = self.roots.values().flat_map(|root| {
            std::iter::once(&root.segments)
                .chain(root.descriptors.values())
                .chain(root.tensors.values())
        });
        roots
            .chain(self.clusters.values().flatten())
            .map(ObjectKey::to_string)
            .collect()
    }

    /// Whether the two manifests describe the same index: every field but
    /// [`Provenance`], which says who wrote it and when.
    pub fn same_content(&self, other: &Manifest) -> bool {
        self.format_version == other.format_version
            && self.hash_algo == other.hash_algo
            && self.config_fingerprint == other.config_fingerprint
            && self.dimensions == other.dimensions
            && self.objects == other.objects
            && self.roots == other.roots
            && self.clusters == other.clusters
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

    /// Serialize canonically, one resource per line.
    ///
    /// Still exactly one byte string per manifest — sorted map keys,
    /// declaration-ordered struct fields, compact values — so Tier A holds.
    ///
    /// The layout is chosen for what git does with it. A resource is one line:
    /// its root, and the entries of the objects that the root names. So a
    /// change to one resource is a change to one line, and a review diff names
    /// the resources that moved. An empty line stands between two resources,
    /// because git merges two changes that have an unchanged line between
    /// them and stops for two changes on adjacent lines.
    ///
    /// Measured on a repository of 1,259 resources, with the table layout of
    /// before: two branches with no file in common gave 8 conflicts in a text
    /// merge, because the table is sorted by key and each change lands at a
    /// random place in it. In this layout the same merge has none, and the
    /// result is the manifest that an index build of the merged files writes.
    ///
    /// A resource that both sides changed is still one line with two
    /// versions, which no text merge resolves. [`Manifest::merge`] does.
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

        /// A root with the entries of the objects it names.
        #[derive(Serialize)]
        struct RootLine<'a> {
            segments: &'a ObjectKey,
            descriptors: &'a BTreeMap<String, ObjectKey>,
            #[serde(skip_serializing_if = "BTreeMap::is_empty")]
            tensors: &'a BTreeMap<String, ObjectKey>,
            objects: BTreeMap<String, &'a ObjectEntry>,
        }
        #[derive(Serialize)]
        struct ProvenanceLine<'a> {
            map_version: &'a str,
            /// Always 0. A binary of the layout before reads the whole
            /// document before it looks at `format_version`, and it wants
            /// this field. Without it that binary reports a JSON error and
            /// says to delete the manifest; with it, it says that the index
            /// is of a newer version and names the upgrade. A constant is on
            /// each side of a merge, so it is never a conflict.
            generated_at: u64,
        }

        if self.format_version > TABLE_LAYOUT {
            let mut in_a_root: BTreeSet<String> = BTreeSet::new();
            let mut lines = Vec::with_capacity(self.roots.len());
            for (resource, root) in &self.roots {
                let mut objects = BTreeMap::new();
                let keys = std::iter::once(&root.segments)
                    .chain(root.descriptors.values())
                    .chain(root.tensors.values());
                for key in keys {
                    let hex = key.to_string();
                    if let Some(entry) = self.objects.get(&hex) {
                        objects.insert(hex.clone(), entry);
                    }
                    in_a_root.insert(hex);
                }
                let line = RootLine {
                    segments: &root.segments,
                    descriptors: &root.descriptors,
                    tensors: &root.tensors,
                    objects,
                };
                lines.push((resource, canonical_json(&line)?));
            }
            let rest: BTreeMap<String, &ObjectEntry> = self
                .objects
                .iter()
                .filter(|(key, _)| !in_a_root.contains(*key))
                .map(|(key, entry)| (key.clone(), entry))
                .collect();

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
            block(&mut out, "objects", &rest)?;
            out.extend_from_slice(b",\n");
            if !self.clusters.is_empty() {
                block(&mut out, "clusters", &self.clusters)?;
                out.extend_from_slice(b",\n");
            }
            let provenance = ProvenanceLine {
                map_version: &self.provenance.map_version,
                generated_at: 0,
            };
            scalar(&mut out, "provenance", &provenance)?;
            out.extend_from_slice(b",\n");
            // Last, and with an empty line between two resources. Git merges
            // two changes that have one line between them that did not
            // change, and stops for two changes on adjacent lines. Measured:
            // 2 conflicts with no empty line, 1 with it, for the same merge.
            out.extend_from_slice(b"  \"roots\": {\n");
            for (i, (resource, line)) in lines.iter().enumerate() {
                out.extend_from_slice(if i > 0 { b",\n\n" } else { b"\n" });
                out.extend_from_slice(b"    ");
                out.extend_from_slice(&canonical_json(resource)?);
                out.extend_from_slice(b": ");
                out.extend_from_slice(line);
            }
            out.extend_from_slice(b"\n\n  }\n}\n");
            return Ok(out);
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
        /// The two layouts in one shape: a root can have the entries of its
        /// objects, and the table has all entries or only those of no root.
        #[derive(Deserialize)]
        struct RootText {
            #[serde(flatten)]
            root: ResourceRoot,
            #[serde(default)]
            objects: BTreeMap<String, ObjectEntry>,
        }
        #[derive(Deserialize)]
        struct ManifestText {
            format_version: u32,
            hash_algo: String,
            config_fingerprint: Fingerprint,
            dimensions: BTreeMap<String, DimensionIdentity>,
            #[serde(default)]
            objects: BTreeMap<String, ObjectEntry>,
            #[serde(default)]
            roots: BTreeMap<String, RootText>,
            #[serde(default)]
            clusters: BTreeMap<String, Vec<ObjectKey>>,
            provenance: Provenance,
        }

        let text: ManifestText = match serde_json::from_slice(bytes) {
            Ok(text) => text,
            // Looked for only after the parse fails, so a manifest that reads
            // pays nothing for it. Markers are never valid JSON, so nothing
            // that parses can hold them.
            Err(_) if has_conflict_markers(bytes) => return Err(Error::ConflictMarkers),
            Err(e) => return Err(e.into()),
        };

        if text.format_version != MANIFEST_VERSION && text.format_version != TABLE_LAYOUT {
            return Err(Error::UnsupportedVersion {
                kind: "manifest",
                found: text.format_version,
                expected: MANIFEST_VERSION,
            });
        }

        let mut manifest = Manifest {
            format_version: text.format_version,
            hash_algo: text.hash_algo,
            config_fingerprint: text.config_fingerprint,
            dimensions: text.dimensions,
            objects: text.objects,
            roots: BTreeMap::new(),
            clusters: text.clusters,
            provenance: text.provenance,
        };
        for (resource, line) in text.roots {
            for (key, entry) in line.objects {
                // Two files with one content name one object, so one key can
                // be on two lines. Two contents for it is the collision that
                // the table refuses.
                match manifest.objects.get(&key) {
                    Some(existing) if existing.content_hash != entry.content_hash => {
                        return Err(Error::KeyCollision {
                            key,
                            a: existing.content_hash.to_string(),
                            b: entry.content_hash.to_string(),
                        });
                    }
                    Some(_) => {}
                    None => {
                        manifest.objects.insert(key, entry);
                    }
                }
            }
            manifest.roots.insert(resource, line.root);
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
    /// A textual merge cannot do this: it cannot tell a legitimate union of
    /// object tables from a same-key/different-bytes collision, and it cannot
    /// know that two re-fabricated cluster trees describe neither side's
    /// object set.
    ///
    /// The result holds what an index build of the merged resources records:
    /// the merged roots, and the entries those roots and the cluster trees
    /// name. So a build after a merge that had no conflict changes nothing.
    ///
    /// "Three-way" on a single value means: equal on both sides, take it;
    /// unchanged on our side, take theirs; unchanged on theirs, take ours;
    /// otherwise a conflict. A key missing on one side is a deletion when
    /// `base` had it and the other side left it alone, and an addition when
    /// `base` lacked it. `base` of `None` is an empty ancestor.
    ///
    /// A resource that each side indexed again is not a conflict. The merge
    /// only has the manifests, so it cannot build the entry of the merged
    /// resource. It keeps what `ours` holds and lists the resource in
    /// [`Merged::stale`]. That is the ordinary state of an index one edit
    /// behind its resource, and the next index build puts it right.
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
        if ours.hash_algo != theirs.hash_algo {
            conflicts.push(MergeConflict::Format);
        }
        // Both sides were read, so both are layouts that this build knows, and
        // they hold the same fields. A branch that an index build moved to the
        // new layout merges with one that it did not, and the result is in the
        // newer of the two.
        let format_version = ours.format_version.max(theirs.format_version);

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
        // The union is only where this starts: the entries that the merged
        // roots no longer name are taken out below.
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
        let mut stale = Vec::new();
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
                None => {
                    // Measured on a real merge: with no entry, a search for a
                    // name that this resource declares ranks another file
                    // first and says nothing. With the entry of one side the
                    // resource is still found. `ours` is the side whose
                    // objects are in the work tree in each kind of merge.
                    if let Some(root) = ours.roots.get(resource) {
                        roots.insert(resource.clone(), root.clone());
                    }
                    stale.push(resource.clone());
                }
            }
        }

        // Runs after `dimensions` because dropping a tree rewrites the built
        // levels that describe it.
        let mut clusters = BTreeMap::new();
        // The clusters of a dropped tree. Nothing names them any more, and the
        // next index build still reuses them, so their entries stay: a reused
        // object is checked against the hash recorded here (spec §8).
        let mut reusable: BTreeSet<String> = BTreeSet::new();
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
                    for side in [ours, theirs] {
                        let tree = side.clusters.get(name).into_iter().flatten();
                        reusable.extend(tree.map(ObjectKey::to_string));
                    }
                    notes.push(format!(
                        "clusters for dimension {name:?} were dropped: both sides re-fabricated \
                         and neither tree describes the merged object set. Run `map index` to \
                         rebuild it; unchanged subtrees are reused."
                    ));
                }
            }
        }

        // The side in the newer layout is the newer producer, and the result
        // is in its layout. Between two sides in the table layout, the later
        // run is the one that describes the objects. `ours` when nothing
        // tells them apart.
        let newer = |m: &Manifest| (m.format_version, m.provenance.generated_at);
        let provenance = if newer(theirs) > newer(ours) {
            theirs.provenance.clone()
        } else {
            ours.provenance.clone()
        };

        if !conflicts.is_empty() {
            return Err(conflicts);
        }

        let mut manifest = Manifest {
            format_version,
            hash_algo: ours.hash_algo.clone(),
            config_fingerprint,
            dimensions,
            objects,
            roots,
            clusters,
            provenance,
        };

        // A side that indexed a resource again put new entries in its table,
        // and the other side still holds the entries of the version before.
        // The union keeps those, although no merged root names them. An index
        // build records only what a root or a tree names, so without this the
        // next build takes them out, and a merge that changed nothing else
        // leaves the manifest to be written and committed one more time.
        let mut named = manifest.named_keys();
        named.append(&mut reusable);
        manifest.objects.retain(|key, _| named.contains(key));

        Ok(Merged {
            manifest,
            notes,
            stale,
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
    /// The resources that each side indexed again, in canonical order. The
    /// manifest has the entry of `ours` for each, or none when `ours` deleted
    /// the resource. An index build after the merge replaces them.
    pub stale: Vec<String>,
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
    /// Both sides changed a dimension's identity — its fingerprint, what it
    /// emits, or the levels it holds.
    Dimension { name: String },
    /// Both sides built from a different `config.toml`.
    ConfigFingerprint,
    /// Different hash algorithms; not a field-by-field question.
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
            MergeConflict::Dimension { name } => {
                write!(f, "dimension {name:?}: both sides changed its identity")
            }
            MergeConflict::ConfigFingerprint => f.write_str(
                "both sides built from a different config.toml; merge the config first, \
                 then re-index",
            ),
            MergeConflict::Format => {
                f.write_str("the two manifests disagree on the hash algorithm")
            }
        }
    }
}

/// Whether a git text merge left its conflict markers in these bytes.
///
/// [`Manifest::to_bytes`] starts every line with a brace or an indent, so a
/// line that starts with a marker is the trace of a merge and never data.
/// Git writes the same two markers in each conflict style.
pub fn has_conflict_markers(bytes: &[u8]) -> bool {
    bytes
        .split(|&byte| byte == b'\n')
        .any(|line| line.starts_with(b"<<<<<<<") || line.starts_with(b">>>>>>>"))
}

/// The names of the dimensions that these manifest bytes record, or `None`
/// when the bytes do not parse.
///
/// For a caller that needs this answer and nothing else. `map brief` runs at
/// the start of each agent session, and [`Manifest::from_bytes`] builds the
/// table of each object to give the same names: measured on a manifest of
/// 1,294 resources (1.05 MB), 4.0 ms against 0.5 ms here.
pub fn dimension_names(bytes: &[u8]) -> Option<BTreeSet<String>> {
    #[derive(Deserialize)]
    struct Names {
        dimensions: BTreeMap<String, serde::de::IgnoredAny>,
    }
    let names: Names = serde_json::from_slice(bytes).ok()?;
    Some(names.dimensions.into_keys().collect())
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

    /// What a manifest reads back as. The time of the run is not written.
    fn as_written(m: &Manifest) -> Manifest {
        let mut written = m.clone();
        if written.format_version > TABLE_LAYOUT {
            written.provenance.generated_at = 0;
        }
        written
    }

    #[test]
    fn roundtrips_through_json() {
        let m = manifest();
        assert_eq!(
            Manifest::from_bytes(&m.to_bytes().unwrap()).unwrap(),
            as_written(&m)
        );
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
        assert_eq!(
            Manifest::from_bytes(&m.to_bytes().unwrap()).unwrap(),
            as_written(&m)
        );
    }

    /// Resources whose objects have entries, as an index build records them,
    /// and one cluster object that no root names.
    fn indexed(resources: &[(&str, &[u8])]) -> Manifest {
        let mut m = manifest();
        for (resource, seed) in resources {
            let segments = key(&[seed, b"/segments".as_slice()].concat());
            let descriptor = key(&[seed, b"/descriptor".as_slice()].concat());
            for k in [segments, descriptor] {
                m.insert_object(k, entry(seed, Tier::A)).unwrap();
            }
            m.roots.insert(
                (*resource).to_owned(),
                ResourceRoot {
                    segments,
                    descriptors: BTreeMap::from([("structural".to_owned(), descriptor)]),
                    tensors: BTreeMap::new(),
                },
            );
        }
        m.insert_object(key(b"cluster"), entry(b"cluster", Tier::C))
            .unwrap();
        m.clusters
            .insert("lexical".to_owned(), vec![key(b"cluster")]);
        m
    }

    fn text_of(m: &Manifest) -> String {
        String::from_utf8(m.to_bytes().unwrap()).unwrap()
    }

    #[test]
    fn a_resource_is_one_line_with_the_entries_of_its_objects() {
        let m = indexed(&[("a.rs", b"a"), ("b.rs", b"b")]);
        let text = text_of(&m);
        let lines: Vec<&str> = text.lines().collect();

        for (resource, root) in &m.roots {
            let at = lines
                .iter()
                .position(|l| l.starts_with(&format!("    \"{resource}\": ")))
                .expect("the resource has a line");
            for k in [root.segments, root.descriptors["structural"]] {
                let k = k.to_string();
                // Once as the key that the root names, and once as the key of
                // the entry. On no other line: the table does not repeat it.
                assert_eq!(lines[at].matches(&k).count(), 2, "{resource}");
                assert_eq!(text.matches(&k).count(), 2, "{resource}");
            }
            // An empty line on each side is what lets git merge a change to
            // this resource with a change to the next one.
            assert_eq!(lines[at - 1], "", "before {resource}");
            assert_eq!(lines[at + 1], "", "after {resource}");
        }

        // The entry that no root names stays in the table.
        let cluster = key(b"cluster").to_string();
        let table = text.split("\"roots\"").next().unwrap();
        assert!(table.contains(&format!("    \"{cluster}\": {{\"content_hash\"")));

        // A time in the file is a line that each side of a merge changes. The
        // field stays, as a constant, for a reader of the layout before.
        assert_ne!(m.provenance.generated_at, 0);
        assert!(text.contains("\"provenance\": {\"map_version\":\"0.0.0\",\"generated_at\":0}"));
        assert!(!text.contains('\r'), "line endings must be LF");
        assert!(text.ends_with("}\n"), "must end with a trailing newline");
        assert_eq!(
            Manifest::from_bytes(text.as_bytes()).unwrap(),
            as_written(&m)
        );
    }

    #[test]
    fn a_change_to_one_resource_changes_one_line() {
        // What makes two branches merge as text: each touches only the lines
        // of the resources that it changed.
        let before = indexed(&[("a.rs", b"a"), ("b.rs", b"b"), ("c.rs", b"c")]);
        let after = indexed(&[("a.rs", b"a"), ("b.rs", b"b2"), ("c.rs", b"c")]);
        let (before, after) = (text_of(&before), text_of(&after));
        let (before, after): (Vec<&str>, Vec<&str>) =
            (before.lines().collect(), after.lines().collect());

        assert_eq!(before.len(), after.len());
        let changed: Vec<usize> = (0..before.len())
            .filter(|&i| before[i] != after[i])
            .collect();
        assert_eq!(changed.len(), 1, "{changed:?}");
        assert!(before[changed[0]].starts_with("    \"b.rs\": "));
    }

    #[test]
    fn the_table_layout_reads_and_is_written_back_byte_for_byte() {
        // An index of an earlier version. A merge by this build must not turn
        // it into a file that the binary of a team mate refuses.
        let mut m = indexed(&[("a.rs", b"a"), ("b.rs", b"b")]);
        m.format_version = TABLE_LAYOUT;
        let text = text_of(&m);

        assert!(text.contains("\"format_version\": 1,"));
        assert!(text.contains("\"generated_at\":1700000000"));
        // One table with each entry, and a root with no entries of its own.
        let table = text.split("\"roots\"").next().unwrap();
        for k in m.objects.keys() {
            assert_eq!(text.matches(k.as_str()).count(), 2, "{k}");
            assert!(table.contains(&format!("    \"{k}\": {{\"content_hash\"")));
        }

        let read = Manifest::from_bytes(text.as_bytes()).unwrap();
        assert_eq!(read, m);
        assert_eq!(read.to_bytes().unwrap(), text.as_bytes());
    }

    #[test]
    fn one_key_with_two_contents_on_two_lines_is_refused() {
        // Two files with one content name one object, so its entry is on the
        // lines of the two resources. A text merge can leave the two lines
        // with different bytes for it.
        let mut m = indexed(&[("a.rs", b"same"), ("b.rs", b"other")]);
        let shared = m.roots["a.rs"].clone();
        m.roots.insert("b.rs".to_owned(), shared);
        let named = m.named_keys();
        m.objects.retain(|k, _| named.contains(k));
        let text = text_of(&m);
        assert_eq!(
            Manifest::from_bytes(text.as_bytes()).unwrap().objects,
            m.objects
        );

        let hash = entry(b"same", Tier::A).content_hash.to_string();
        let forged = entry(b"forged", Tier::A).content_hash.to_string();
        let at = text.find("    \"b.rs\": ").unwrap();
        let text = format!("{}{}", &text[..at], text[at..].replacen(&hash, &forged, 1));
        assert!(matches!(
            Manifest::from_bytes(text.as_bytes()),
            Err(Error::KeyCollision { .. })
        ));
    }

    #[test]
    fn a_merge_of_the_two_layouts_is_in_the_new_layout() {
        // One branch ran an index build with this version, and one did not.
        let mut base = indexed(&[("shared.rs", b"shared")]);
        base.format_version = TABLE_LAYOUT;
        let mut ours = indexed(&[("shared.rs", b"shared"), ("ours.rs", b"ours")]);
        ours.format_version = TABLE_LAYOUT;
        let theirs = indexed(&[("shared.rs", b"shared"), ("theirs.rs", b"theirs")]);

        // The table layout has a time, and the new one reads as 0. The time
        // must not make the older producer the one that the result names.
        ours.provenance.map_version = "0.0.5".to_owned();
        let mut theirs = theirs;
        theirs.provenance.map_version = "0.0.6".to_owned();
        theirs.provenance.generated_at = 0;

        for (a, b) in [(&ours, &theirs), (&theirs, &ours)] {
            let merged = Manifest::merge(Some(&base), a, b).unwrap().manifest;
            assert_eq!(merged.format_version, MANIFEST_VERSION);
            assert_eq!(merged.provenance.map_version, "0.0.6");
            assert_eq!(
                merged.roots.keys().collect::<Vec<_>>(),
                vec!["ours.rs", "shared.rs", "theirs.rs"]
            );
        }
        // Two sides in the table layout stay in it.
        let merged = Manifest::merge(Some(&base), &ours, &base).unwrap().manifest;
        assert_eq!(merged.format_version, TABLE_LAYOUT);
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
        assert!(merged.stale.is_empty());
    }

    #[test]
    fn a_resource_that_each_side_indexed_again_keeps_our_entry_and_is_listed() {
        let mut base = manifest();
        for (resource, seed) in [("a.rs", b"a0".as_slice()), ("gone.rs", b"g0".as_slice())] {
            base.roots.insert(resource.to_owned(), root(seed));
            base.insert_object(key(seed), entry(seed, Tier::A)).unwrap();
        }

        let mut ours = base.clone();
        ours.roots.insert("a.rs".to_owned(), root(b"ours"));
        ours.insert_object(key(b"ours"), entry(b"ours", Tier::A))
            .unwrap();
        // Our side deleted this one, and their side changed it.
        ours.roots.remove("gone.rs");

        let mut theirs = base.clone();
        for (resource, seed) in [
            ("a.rs", b"theirs".as_slice()),
            ("gone.rs", b"g-theirs".as_slice()),
        ] {
            theirs.roots.insert(resource.to_owned(), root(seed));
            theirs
                .insert_object(key(seed), entry(seed, Tier::A))
                .unwrap();
        }

        let merged = Manifest::merge(Some(&base), &ours, &theirs).unwrap();
        assert_eq!(merged.stale, vec!["a.rs", "gone.rs"]);
        assert_eq!(merged.manifest.roots.get("a.rs"), Some(&root(b"ours")));
        assert!(
            !merged.manifest.roots.contains_key("gone.rs"),
            "our side has no entry for it, so the merge has none"
        );
        // The table holds what the merged roots name, and no entry of the
        // version that lost.
        assert_eq!(
            merged.manifest.objects.keys().cloned().collect::<Vec<_>>(),
            vec![key(b"ours").to_string()]
        );

        // A resource that only one side changed is not in the list.
        let mut theirs = base.clone();
        theirs.roots.insert("a.rs".to_owned(), root(b"theirs"));
        theirs
            .insert_object(key(b"theirs"), entry(b"theirs", Tier::A))
            .unwrap();
        let merged = Manifest::merge(Some(&base), &base, &theirs).unwrap();
        assert!(merged.stale.is_empty());
        assert_eq!(merged.manifest.roots.get("a.rs"), Some(&root(b"theirs")));
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
    fn a_side_that_built_without_a_dimension_takes_it_out_of_the_merge() {
        // One side built with a stage missing, and its resources have no data
        // for the dimension. A merged manifest that kept the record would say
        // that the dimension is built, and a search on it would then miss
        // those resources with no message.
        let base = manifest();
        let mut ours = base.clone();
        index(&mut ours, "src/a.rs", b"a");
        let mut theirs = base.clone();
        theirs.dimensions.remove("declaration");
        index(&mut theirs, "src/b.rs", b"b");

        for (left, right) in [(&ours, &theirs), (&theirs, &ours)] {
            let merged = Manifest::merge(Some(&base), left, right).unwrap();
            let names: Vec<&String> = merged.manifest.dimensions.keys().collect();
            assert_eq!(names, ["lexical"]);
            assert_eq!(merged.manifest.roots.len(), 2);
        }
    }

    /// Record `resource` the way an index build does: a root, and the object
    /// that the root names.
    fn index(manifest: &mut Manifest, resource: &str, seed: &[u8]) {
        manifest.roots.insert(resource.to_owned(), root(seed));
        manifest
            .insert_object(key(seed), entry(seed, Tier::A))
            .unwrap();
    }

    #[test]
    fn a_merge_holds_what_a_build_of_the_merged_resources_records() {
        // Each side indexed one resource again, and its build took the entry
        // of the version before out of its own table. The other side still
        // holds that entry. A union keeps the two, no root names them, and the
        // next build then changes the manifest of a merge that had no conflict.
        let mut base = manifest();
        index(&mut base, "a.rs", b"a1");
        index(&mut base, "b.rs", b"b1");

        let mut ours = manifest();
        index(&mut ours, "a.rs", b"a2");
        index(&mut ours, "b.rs", b"b1");

        let mut theirs = manifest();
        index(&mut theirs, "a.rs", b"a1");
        index(&mut theirs, "b.rs", b"b2");

        let mut built = manifest();
        index(&mut built, "a.rs", b"a2");
        index(&mut built, "b.rs", b"b2");

        let merged = Manifest::merge(Some(&base), &ours, &theirs)
            .unwrap()
            .manifest;
        assert!(
            merged.object(key(b"a1")).is_none(),
            "a.rs has a newer entry"
        );
        assert!(
            merged.object(key(b"b1")).is_none(),
            "b.rs has a newer entry"
        );
        assert!(merged.same_content(&built));
    }

    #[test]
    fn an_entry_that_a_root_of_the_other_side_names_stays() {
        // Two files with the same content use one object. One side changes
        // `a.rs`, and the other side adds a copy of the `a.rs` of before. A
        // three-way rule on the table would read that as "our side deleted
        // the entry" and leave the root of the copy pointing at nothing.
        let mut base = manifest();
        index(&mut base, "a.rs", b"shared");

        let mut ours = manifest();
        index(&mut ours, "a.rs", b"changed");

        let mut theirs = manifest();
        index(&mut theirs, "a.rs", b"shared");
        index(&mut theirs, "copy.rs", b"shared");

        let merged = Manifest::merge(Some(&base), &ours, &theirs)
            .unwrap()
            .manifest;
        assert_eq!(merged.roots["copy.rs"], root(b"shared"));
        assert!(merged.object(key(b"shared")).is_some());
        assert!(merged.object(key(b"changed")).is_some());
        assert_eq!(merged.objects.len(), 2);
    }

    #[test]
    fn the_clusters_of_a_dropped_tree_keep_their_entries() {
        // The next build reuses a cluster that is on disk, and it compares
        // the bytes with the hash that the manifest records. Without the
        // entry there is no hash, and the build takes the bytes as they are.
        let mut base = manifest();
        base.clusters.insert("lexical".to_owned(), vec![key(b"c0")]);

        let mut ours = base.clone();
        ours.clusters.insert("lexical".to_owned(), vec![key(b"c1")]);
        ours.insert_object(key(b"c1"), entry(b"c1", Tier::C))
            .unwrap();

        let mut theirs = base.clone();
        theirs
            .clusters
            .insert("lexical".to_owned(), vec![key(b"c2")]);
        theirs
            .insert_object(key(b"c2"), entry(b"c2", Tier::C))
            .unwrap();

        let merged = Manifest::merge(Some(&base), &ours, &theirs)
            .unwrap()
            .manifest;
        assert!(merged.clusters.is_empty());
        assert!(merged.object(key(b"c1")).is_some());
        assert!(merged.object(key(b"c2")).is_some());
    }

    #[test]
    fn the_named_keys_are_those_of_the_roots_and_the_cluster_trees() {
        let named = populated().named_keys();
        let expected: BTreeSet<String> = [b"segments".as_slice(), b"desc", b"c1", b"c2"]
            .into_iter()
            .map(|seed| key(seed).to_string())
            .collect();
        assert_eq!(named, expected);
    }

    #[test]
    fn the_same_content_is_every_field_but_provenance() {
        let m = populated();

        let mut later = m.clone();
        later.provenance = Provenance {
            map_version: "9.9.9".to_owned(),
            generated_at: m.provenance.generated_at + 1,
        };
        assert!(m.same_content(&later));

        let mut more = m.clone();
        more.insert_object(key(b"new"), entry(b"new", Tier::A))
            .unwrap();
        assert!(!m.same_content(&more));

        let mut moved = m.clone();
        moved.roots.insert("src/lib.rs".to_owned(), root(b"other"));
        assert!(!m.same_content(&moved));

        let mut taller = m.clone();
        taller.dimensions.get_mut("lexical").unwrap().levels = vec![0, 1];
        assert!(!m.same_content(&taller));
    }

    /// The text of `populated()` after a git text merge stopped on its first
    /// object entry.
    fn with_conflict_markers(open: &str, close: &str) -> Vec<u8> {
        let m = populated();
        let text = String::from_utf8(m.to_bytes().unwrap()).unwrap();
        let first = m.objects.keys().next().unwrap();
        let mut out = String::new();
        for line in text.lines() {
            if line.contains(first.as_str()) {
                out.push_str(&format!("{open}\n{line}\n=======\n{line}\n{close}\n"));
            } else {
                out.push_str(line);
                out.push('\n');
            }
        }
        out.into_bytes()
    }

    #[test]
    fn a_manifest_with_conflict_markers_says_so_and_gives_the_repair() {
        // What `serde_json` says about this file is "key must be a string at
        // line 7 column 1", which names neither the cause nor the way out.
        let bytes = with_conflict_markers("<<<<<<< HEAD", ">>>>>>> develop");
        let err = Manifest::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, Error::ConflictMarkers), "{err:?}");

        let message = err.to_string();
        for part in [
            "conflict markers",
            "git checkout --ours .map/manifest.json",
            "`map index`",
            "`git add .map`",
        ] {
            assert!(message.contains(part), "{part:?} is not in: {message}");
        }
        assert!(!message.contains('\n'), "callers quote it in a sentence");
    }

    #[test]
    fn a_manifest_that_is_only_damaged_keeps_its_json_error() {
        // The repair for a merge is the wrong advice for a file that was cut
        // short, and that one still needs the position in the file.
        let m = populated();
        let mut bytes = m.to_bytes().unwrap();
        bytes.truncate(bytes.len() / 2);
        assert!(!has_conflict_markers(&bytes));
        assert!(matches!(
            Manifest::from_bytes(&bytes).unwrap_err(),
            Error::Json(_)
        ));
    }

    #[test]
    fn the_names_only_read_gives_the_dimensions_of_the_full_parse() {
        // Two readers of one question. If they disagree, `map brief` offers a
        // dimension that `map find` refuses, or hides one that it has.
        let mut m = populated();
        m.dimensions.remove("lexical");
        for layout in [TABLE_LAYOUT, MANIFEST_VERSION] {
            m.format_version = layout;
            let bytes = m.to_bytes().unwrap();
            let full: BTreeSet<String> = Manifest::from_bytes(&bytes)
                .unwrap()
                .dimensions
                .into_keys()
                .collect();
            assert!(!full.is_empty() && !full.contains("lexical"), "{full:?}");
            assert_eq!(dimension_names(&bytes), Some(full), "layout {layout}");
        }

        // No answer is not the answer "no dimensions".
        let marked = with_conflict_markers("<<<<<<< HEAD", ">>>>>>> develop");
        assert_eq!(dimension_names(&marked), None);
    }

    #[test]
    fn conflict_markers_are_found_at_the_start_of_a_line_only() {
        assert!(!has_conflict_markers(&populated().to_bytes().unwrap()));
        // A checkout on Windows gives the file CRLF line ends.
        let crlf = String::from_utf8(with_conflict_markers("<<<<<<< HEAD", ">>>>>>> x"))
            .unwrap()
            .replace('\n', "\r\n");
        assert!(has_conflict_markers(crlf.as_bytes()));
        // A resource can have a name that holds a marker. It is inside a
        // JSON string then, after the indent of its line.
        let mut m = populated();
        m.roots.insert("<<<<<<< odd name".to_owned(), root(b"odd"));
        let bytes = m.to_bytes().unwrap();
        assert!(!has_conflict_markers(&bytes));
        assert_eq!(Manifest::from_bytes(&bytes).unwrap(), as_written(&m));
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
        let base = manifest();

        let mut ours = base.clone();
        let mut theirs = base.clone();
        for (side, seed) in [(&mut ours, b"ours"), (&mut theirs, b"thrs")] {
            for contested in [b"k1", b"k2"] {
                side.insert_object(key(contested), entry(seed, Tier::C))
                    .unwrap();
            }
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
