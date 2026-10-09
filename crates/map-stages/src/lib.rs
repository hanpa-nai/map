//! Reference stage implementations.
//!
//! These are deliberately **correct but not competitive**. They exist so the
//! pipeline runs end-to-end with nothing configured.
//!
//! | Stage | Implementation | Notes |
//! |---|---|---|
//! | discover | [`discover::FsDiscoverer`] | gitignore-aware walk, sorted output |
//! | preprocess | [`preprocess::TextPreprocessor`] | UTF-8 decode, LF normalization |
//! | segment | [`segment::WindowSegmenter`] | line windows with overlap |
//! | classify | [`lexical::StructuralClassifier`] | term frequencies, no model |
//! | score | [`pack::PackBuilder`] / [`pack::Pack`] | Okapi BM25, memory-mapped |
//! | score | [`lexical::Bm25Builder`] | Okapi BM25, in-memory reference |
//! | score | [`densepack::DensePackBuilder`] | cosine over tensors, memory-mapped |
//! | score | [`cosine::CosineBuilder`] | cosine over tensors, in-memory reference |
//!
//! The structural classifier and the BM25 pack are all a default build
//! contains: offline, keyless, and Tier A deterministic. Embedding is not done
//! here at all — the pipeline's embedder is `map-embed`'s distilled one.
//!
//! # Why two scorers of each kind
//!
//! An interface with exactly one implementation is indistinguishable from that
//! implementation's internals, and this codebase has already been bitten by it:
//! the previous `Scorer` trait was shaped around an in-memory scorer, so the
//! memory-mapped one that actually runs could not implement it and silently
//! bypassed the interface entirely.

pub mod cache;
#[cfg(feature = "llm")]
pub mod chat;
pub mod content;
pub mod cosine;
pub mod declaration;
pub mod densepack;
pub mod discover;
pub mod fabricate;
pub mod lexical;
pub mod pack;
pub mod preprocess;
pub mod segment;

pub use content::ContentClassifier;
pub use cosine::CosineBuilder;
pub use declaration::DeclarationClassifier;
pub use densepack::{DensePack, DensePackBuilder};
pub use discover::FsDiscoverer;
pub use lexical::{Bm25Builder, StructuralClassifier};
pub use pack::{Pack, PackBuilder};
pub use preprocess::TextPreprocessor;
pub use segment::WindowSegmenter;
