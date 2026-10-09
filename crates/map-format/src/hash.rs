//! Hashing primitives.
//!
//! # Two hashes that must never be conflated
//!
//! The format uses hashing for two unrelated jobs, and the whole incremental
//! story depends on keeping them distinct (spec §4):
//!
//! - **[`ObjectKey`]** — `hash(input + stage_config_fingerprint)`. This is
//!   *input*-addressed, not content-addressed. It answers "what work does this
//!   represent?" and stays stable even when the produced bytes are not
//!   reproducible (embeddings, LLM descriptors). That stability is exactly what
//!   makes incremental indexing possible.
//! - **[`ContentHash`]** — `hash(bytes)`. This answers "are these the bytes the
//!   producer wrote?" It exists because input-addressed objects cannot be
//!   self-verified, and it is the only integrity mechanism in the format.
//!
//! Two producers may legitimately write the same [`ObjectKey`] with different
//! [`ContentHash`]es. That is a collision to resolve, never a union to take.
//!
//! # Domain separation
//!
//! Every hash is computed over a domain tag so that a value hashed as one kind
//! can never collide with the same bytes hashed as another.

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::{Error, Result};

/// Name of the hash algorithm, recorded in the manifest.
///
/// Recorded rather than assumed, so an index names the algorithm it was built
/// with instead of relying on every reader having compiled the same constant.
pub const HASH_ALGO: &str = "blake3-256";

/// A 32-byte digest, displayed and serialized as lowercase hex.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Digest([u8; 32]);

impl Digest {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Digest(bytes)
    }

    pub fn to_hex(self) -> String {
        let mut s = String::with_capacity(64);
        for b in self.0 {
            s.push(char::from_digit((b >> 4) as u32, 16).expect("nibble"));
            s.push(char::from_digit((b & 0xf) as u32, 16).expect("nibble"));
        }
        s
    }

    /// Parse from lowercase or uppercase hex.
    pub fn from_hex(s: &str) -> Result<Self> {
        if s.len() != 64 {
            return Err(Error::MalformedDigest(s.to_owned()));
        }
        let mut out = [0u8; 32];
        let bytes = s.as_bytes();
        for (i, slot) in out.iter_mut().enumerate() {
            let hi = (bytes[i * 2] as char)
                .to_digit(16)
                .ok_or_else(|| Error::MalformedDigest(s.to_owned()))?;
            let lo = (bytes[i * 2 + 1] as char)
                .to_digit(16)
                .ok_or_else(|| Error::MalformedDigest(s.to_owned()))?;
            *slot = ((hi << 4) | lo) as u8;
        }
        Ok(Digest(out))
    }

    /// First two hex characters — the object store's fan-out directory.
    pub fn shard(&self) -> String {
        self.to_hex()[..2].to_owned()
    }
}

impl fmt::Display for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl fmt::Debug for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Short form keeps test output and logs readable.
        write!(f, "Digest({}…)", &self.to_hex()[..12])
    }
}

impl Serialize for Digest {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for Digest {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Digest::from_hex(&s).map_err(serde::de::Error::custom)
    }
}

/// Domain tags. Changing one of these changes every derived identity in the
/// index, so they are part of the on-disk format rather than an internal
/// detail.
mod domain {
    pub(super) const CONTENT: &[u8] = b"map/v1/content\0";
    pub(super) const OBJECT_KEY: &[u8] = b"map/v1/object-key\0";
    pub(super) const FINGERPRINT: &[u8] = b"map/v1/fingerprint\0";
    pub(super) const CLUSTER: &[u8] = b"map/v1/cluster\0";
}

/// Integrity hash over stored bytes. See module docs.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ContentHash(pub Digest);

impl ContentHash {
    pub fn of(bytes: &[u8]) -> Self {
        let mut h = blake3::Hasher::new();
        h.update(domain::CONTENT);
        h.update(bytes);
        ContentHash(Digest(*h.finalize().as_bytes()))
    }
}

impl fmt::Display for ContentHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

/// Keyed digest for **local, non-committed** markers — the cache-trust stamp
/// that proves a derived pack was built on this machine.
///
/// Unlike [`ContentHash`] and [`ObjectKey`], this is *not* part of the committed
/// format. The key is a per-machine secret, so the output is deliberately
/// unreproducible across machines — which is the point:
/// an attacker who ships a repository cannot compute a matching marker without
/// the key, so a force-committed cache pack cannot pass as locally built.
pub fn keyed_hex(key: &[u8; 32], data: &[u8]) -> String {
    blake3::keyed_hash(key, data).to_hex().to_string()
}

/// Fingerprint of a stage's configuration.
///
/// Folded into every [`ObjectKey`] that stage produces, so changing a
/// classifier prompt invalidates exactly the objects that prompt produced —
/// no more, and no less than must be recomputed anyway (spec §4).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Fingerprint(pub Digest);

impl Fingerprint {
    /// Fingerprint canonical configuration bytes.
    ///
    /// The caller is responsible for canonical encoding; see
    /// [`crate::codec::canonical_json`]. Feeding non-canonical bytes here is
    /// the easiest way to break Tier A determinism.
    pub fn of(canonical_config: &[u8]) -> Self {
        let mut h = blake3::Hasher::new();
        h.update(domain::FINGERPRINT);
        h.update(canonical_config);
        Fingerprint(Digest(*h.finalize().as_bytes()))
    }
}

impl fmt::Display for Fingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

/// Input-addressed identity of a stored object. See module docs.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ObjectKey(pub Digest);

impl ObjectKey {
    /// Derive a key from a stage's inputs and its configuration fingerprint.
    ///
    /// Each input is length-prefixed so that `["ab", "c"]` and `["a", "bc"]`
    /// cannot produce the same key.
    pub fn derive(inputs: &[&[u8]], fingerprint: Fingerprint) -> Self {
        let mut h = blake3::Hasher::new();
        h.update(domain::OBJECT_KEY);
        h.update(fingerprint.0.as_bytes());
        h.update(&(inputs.len() as u64).to_le_bytes());
        for input in inputs {
            h.update(&(input.len() as u64).to_le_bytes());
            h.update(input);
        }
        ObjectKey(Digest(*h.finalize().as_bytes()))
    }

    /// Identity of a level-0 record: the leaf the Merkle tree of
    /// [`ObjectKey::cluster`] is built over.
    ///
    /// `segments` is the key of the resource's segments object. That object is
    /// derived from normalized content plus the segmenter config, so it moves
    /// whenever the file's bytes move — and folding it in here is what makes a
    /// leaf id **content-sensitive** rather than merely span-sensitive.
    ///
    /// Without it, an edit that preserves a span's start and end offsets leaves
    /// every leaf id unchanged, so the Merkle clusters above it keep their
    /// identities too and the fabricator reuses a stale cluster label for
    /// content it no longer describes. Spec §3.1 promises the opposite: it is
    /// the *unchanged* subtree that keeps its identity.
    ///
    /// `dimension` is the dimension's artifact fingerprint, so the same span of
    /// the same file is a distinct record in each dimension.
    ///
    /// This is the single derivation for leaf identity. It replaces two
    /// hand-copied ones — in `map-index`'s `fabricate` and in `map-query` —
    /// which keyed on `(resource, start, end)` alone and had to agree byte for
    /// byte or the query side could not resolve a cluster's children back to
    /// the segments the indexer clustered.
    pub fn segment(
        resource: &str,
        start: u32,
        end: u32,
        segments: ObjectKey,
        dimension: Fingerprint,
    ) -> Self {
        ObjectKey::derive(
            &[
                resource.as_bytes(),
                &start.to_le_bytes(),
                &end.to_le_bytes(),
                segments.0.as_bytes(),
            ],
            dimension,
        )
    }

    /// Merkle identity of a cluster: `hash(sorted(child_ids) + fingerprint)`.
    ///
    /// Children are sorted here rather than trusted from the caller, so an
    /// unchanged subtree keeps its identity regardless of the order the
    /// fabricator happened to emit it in. This is what lets unchanged subtrees
    /// reuse their expensive labels and leave their stored bytes untouched
    /// (spec §3.1).
    ///
    /// Spec §3.1 requires the dimension **name** in this identity, not only the
    /// stage config: two dimensions with byte-identical stage config derive
    /// identical child ids and would otherwise collide on one cluster key. The
    /// name is not hashed here — the caller folds it into `fingerprint` before
    /// calling, so a fingerprint passed in that does not cover the name breaks
    /// the requirement silently.
    pub fn cluster(children: &[ObjectKey], fingerprint: Fingerprint) -> Self {
        let mut sorted: Vec<&ObjectKey> = children.iter().collect();
        sorted.sort_unstable();

        let mut h = blake3::Hasher::new();
        h.update(domain::CLUSTER);
        h.update(fingerprint.0.as_bytes());
        h.update(&(sorted.len() as u64).to_le_bytes());
        for child in sorted {
            h.update(child.0.as_bytes());
        }
        ObjectKey(Digest(*h.finalize().as_bytes()))
    }

    /// Fan-out directory for the object store.
    pub fn shard(&self) -> String {
        self.0.shard()
    }

    /// Filename within the shard directory.
    pub fn rest(&self) -> String {
        self.0.to_hex()[2..].to_owned()
    }
}

impl fmt::Display for ObjectKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fp(tag: &str) -> Fingerprint {
        Fingerprint::of(tag.as_bytes())
    }

    #[test]
    fn hex_roundtrips() {
        let d = ContentHash::of(b"hello").0;
        assert_eq!(Digest::from_hex(&d.to_hex()).unwrap(), d);
        assert_eq!(d.to_hex().len(), 64);
    }

    #[test]
    fn rejects_malformed_hex() {
        assert!(Digest::from_hex("").is_err());
        assert!(Digest::from_hex("zz").is_err());
        assert!(Digest::from_hex(&"g".repeat(64)).is_err());
    }

    #[test]
    fn domains_are_separated() {
        // The same bytes hashed for different purposes must not collide.
        let bytes = b"same";
        let content = ContentHash::of(bytes).0;
        let fingerprint = Fingerprint::of(bytes).0;
        assert_ne!(content, fingerprint);
    }

    #[test]
    fn object_key_is_input_addressed() {
        // Same input + same config => same key. This is the property the whole
        // incremental story rests on.
        let a = ObjectKey::derive(&[b"segment-bytes"], fp("cfg"));
        let b = ObjectKey::derive(&[b"segment-bytes"], fp("cfg"));
        assert_eq!(a, b);
    }

    #[test]
    fn config_change_invalidates_key() {
        let a = ObjectKey::derive(&[b"segment-bytes"], fp("prompt-v1"));
        let b = ObjectKey::derive(&[b"segment-bytes"], fp("prompt-v2"));
        assert_ne!(a, b, "editing a prompt must invalidate its objects");
    }

    #[test]
    fn inputs_are_length_prefixed() {
        // Without length prefixing these would collide.
        let a = ObjectKey::derive(&[b"ab", b"c"], fp("cfg"));
        let b = ObjectKey::derive(&[b"a", b"bc"], fp("cfg"));
        assert_ne!(a, b);
    }

    #[test]
    fn editing_a_resource_changes_every_leaf_id_in_it() {
        // The bug this protects against: an edit that preserves a span's
        // offsets — a same-length rename, a swapped literal — left the leaf id
        // identical, so the cluster above it kept its identity and its label.
        let before = ObjectKey::derive(&[b"content before"], fp("seg"));
        let after = ObjectKey::derive(&[b"content  after"], fp("seg"));

        let stale = ObjectKey::segment("src/lib.rs", 0, 40, before, fp("dim"));
        let fresh = ObjectKey::segment("src/lib.rs", 0, 40, after, fp("dim"));
        assert_ne!(stale, fresh);
    }

    #[test]
    fn a_leaf_id_is_distinct_per_dimension() {
        let segments = ObjectKey::derive(&[b"content"], fp("seg"));
        let lexical = ObjectKey::segment("src/lib.rs", 0, 40, segments, fp("lexical"));
        let semantic = ObjectKey::segment("src/lib.rs", 0, 40, segments, fp("semantic"));
        assert_ne!(lexical, semantic);
    }

    #[test]
    fn an_untouched_span_keeps_its_leaf_id() {
        let segments = ObjectKey::derive(&[b"content"], fp("seg"));
        assert_eq!(
            ObjectKey::segment("src/lib.rs", 0, 40, segments, fp("dim")),
            ObjectKey::segment("src/lib.rs", 0, 40, segments, fp("dim"))
        );
        // And the span itself still participates, so two spans of one file are
        // two records.
        assert_ne!(
            ObjectKey::segment("src/lib.rs", 0, 40, segments, fp("dim")),
            ObjectKey::segment("src/lib.rs", 32, 72, segments, fp("dim"))
        );
    }

    #[test]
    fn cluster_identity_ignores_child_order() {
        let c1 = ObjectKey::derive(&[b"1"], fp("cfg"));
        let c2 = ObjectKey::derive(&[b"2"], fp("cfg"));
        let c3 = ObjectKey::derive(&[b"3"], fp("cfg"));

        let forward = ObjectKey::cluster(&[c1, c2, c3], fp("fab"));
        let shuffled = ObjectKey::cluster(&[c3, c1, c2], fp("fab"));
        assert_eq!(
            forward, shuffled,
            "an unchanged subtree must keep its identity regardless of emit order"
        );
    }

    #[test]
    fn cluster_identity_tracks_membership() {
        let c1 = ObjectKey::derive(&[b"1"], fp("cfg"));
        let c2 = ObjectKey::derive(&[b"2"], fp("cfg"));

        let pair = ObjectKey::cluster(&[c1, c2], fp("fab"));
        let single = ObjectKey::cluster(&[c1], fp("fab"));
        assert_ne!(pair, single, "changing membership must change identity");
    }

    #[test]
    fn shard_splits_at_two_chars() {
        let k = ObjectKey::derive(&[b"x"], fp("cfg"));
        assert_eq!(k.shard().len(), 2);
        assert_eq!(k.rest().len(), 62);
        assert_eq!(format!("{}{}", k.shard(), k.rest()), k.to_string());
    }
}
