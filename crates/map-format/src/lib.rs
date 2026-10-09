//! The `.map` on-disk format.
//!
//! This crate is deliberately *dumb*. It stores two opaque payloads — freeform
//! text and freeform numeric — and never interprets either. Everything
//! semantic lives in the payloads and is versioned by the producing
//! dimension's config fingerprint.
//!
//! That is what keeps the frozen surface small. A new retrieval method is a
//! new stage implementation, never a format change: BM25, for instance,
//! requires nothing here, because term frequencies are simply what the lexical
//! dimension chooses to put in its descriptor text.
//!
//! # The pieces
//!
//! | Module | Role |
//! |---|---|
//! | [`record`] | The only searchable unit: `{descriptor?, tensor?, metadata}` |
//! | [`hash`] | Input-addressed object keys, content hashes, fingerprints |
//! | [`codec`] | Canonical encodings — byte-reproducibility lives here |
//! | [`config`] | `.map/config.toml`, including the dimension table |
//! | [`manifest`] | `.map/manifest.json`, integrity, provenance, attestation |
//! | [`store`] | Immutable loose object storage |
//!
//! # Two invariants worth internalizing
//!
//! **Object keys are input-addressed, not content-addressed.** They answer
//! "what work does this represent?", which stays stable even when the produced
//! bytes are not reproducible. That stability is what makes incremental
//! indexing possible — but it means bytes cannot be self-verified, so the
//! manifest carries a separate content hash per object, and two producers may
//! legitimately write the same key with different bytes.
//!
//! **Determinism is tiered.** Bit-identical output across machines is not
//! achievable in general and nothing here assumes it. Tier A (segmentation,
//! structural classification, BM25, all serialization) is bit-identical
//! everywhere including across platforms; Tier B (embeddings, clusters) is
//! producer-authoritative; Tier C (LLM descriptors, labels) is authored and
//! carries provenance instead.
//!
//! See `spec/format-v1.md` for the normative description.

pub mod codec;
pub mod config;
pub mod error;
pub mod hash;
pub mod manifest;
pub mod record;
pub mod store;

pub use config::{Config, DimensionConfig, Quant, StageRef, StoragePolicy};
pub use error::{Error, Result};
pub use hash::{keyed_hex, ContentHash, Digest, Fingerprint, ObjectKey, HASH_ALGO};
pub use manifest::{Manifest, ObjectEntry, Provenance, Tier};
pub use record::{DType, Record, RecordKind, RecordMeta, Tensor};
pub use store::ObjectStore;

/// Version of the format this build implements.
pub const FORMAT_VERSION: u32 = 1;
