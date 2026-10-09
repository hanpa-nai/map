//! The offline semantic embedder: static distilled token embeddings.
//!
//! # What this is
//!
//! A [Model2Vec](https://github.com/MinishLab/model2vec)-style static embedder.
//! The reference target is `minishlab/potion-retrieval-32M` (MIT, 512-dim, the
//! best-performing static retrieval model). Inference is **not a transformer
//! forward pass** — it is a token-embedding table lookup followed by mean
//! pooling and L2 normalization. That is the whole model:
//!
//! ```text
//! text ──tokenize──▶ [id, id, …] ──lookup──▶ rows ──mean──▶ v ──normalize──▶ vector
//! ```
//!
//! Because there is no attention and no matrix multiply, it needs neither
//! `candle` nor `onnxruntime`: the "model" is an embedded lookup table plus the
//! arithmetic in this file. That is what lets a real semantic dimension link
//! into one auditable binary, and ship as an optional plugin a lexical-only
//! build never compiles.
//!
//! # This module is the math core
//!
//! [`StaticMatrix`] is the lookup-and-pool arithmetic, and it depends on
//! nothing but the tensor type. It is deliberately separated from the weight
//! loader (the tokenizer, the safetensors reader, the model fetch) so the part
//! that decides retrieval quality is unit-testable offline, with a synthetic
//! matrix, before a single byte of real weights is downloaded.
//!
//! # Determinism
//!
//! Pooling sums rows in token order with plain `f32` addition, so the same
//! token ids always produce byte-identical output on the same target. Unlike
//! transformer inference — which varies across SIMD width and thread count and
//! is therefore Tier B — a static lookup can hold much closer to Tier A. The
//! one caveat is cross-architecture `f32` rounding; within a machine it is
//! exact.

use map_format::{DType, Tensor};

#[cfg(feature = "distilled")]
pub mod distilled;
#[cfg(feature = "distilled")]
pub use distilled::DistilledEmbedder;

#[cfg(feature = "auto-distilled")]
pub mod fetch;

/// A static token-embedding table: `vocab` rows of `dim` `f32` each, row-major.
///
/// This is the entire learned model. `potion-retrieval-32M` is one of these
/// with 512-wide rows; a synthetic 2-wide one drives the tests.
#[derive(Clone, Debug)]
pub struct StaticMatrix {
    data: Vec<f32>,
    vocab: usize,
    dim: usize,
    /// Whether to L2-normalize the pooled vector. `potion` sets this true.
    normalize: bool,
}

impl StaticMatrix {
    /// Build from a row-major `vocab × dim` buffer.
    ///
    /// Returns `None` if the buffer length does not match `vocab * dim`, so a
    /// truncated or mis-shaped weight file cannot silently produce garbage
    /// vectors.
    pub fn new(data: Vec<f32>, vocab: usize, dim: usize, normalize: bool) -> Option<Self> {
        if dim == 0 || vocab == 0 || data.len() != vocab.checked_mul(dim)? {
            return None;
        }
        Some(StaticMatrix {
            data,
            vocab,
            dim,
            normalize,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// The row for one token id, or `None` if the id is out of range.
    fn row(&self, id: u32) -> Option<&[f32]> {
        let start = (id as usize).checked_mul(self.dim)?;
        self.data.get(start..start + self.dim)
    }

    /// Pool a sequence of token ids into a single embedding.
    ///
    /// Mean of the in-vocabulary rows, then L2-normalized if configured. Ids
    /// outside the table are skipped rather than erroring: a tokenizer and a
    /// distilled matrix can legitimately disagree at the edges, and one stray
    /// id should not sink a whole segment's vector.
    ///
    /// An empty result (no ids, or none in range) is the zero vector — which is
    /// correct and, being non-normalizable, is left un-normalized. The scorer
    /// treats it as matching nothing.
    pub fn embed_ids(&self, ids: &[u32]) -> Vec<f32> {
        let mut sum = vec![0.0f32; self.dim];
        let mut count = 0u32;

        // Token order, plain addition: fixed and reproducible on this target.
        for &id in ids {
            if let Some(row) = self.row(id) {
                for (acc, x) in sum.iter_mut().zip(row) {
                    *acc += *x;
                }
                count += 1;
            }
        }
        if count == 0 {
            return sum; // all zeros
        }

        let inv = 1.0 / count as f32;
        for x in &mut sum {
            *x *= inv;
        }

        if self.normalize {
            let norm = sum.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut sum {
                    *x /= norm;
                }
            }
        }
        sum
    }

    /// Pool token ids straight into the stored tensor form.
    ///
    /// The record model stores tensors as little-endian `f32` bytes with a
    /// shape; this is that encoding, so an embedder built on this matrix needs
    /// no conversion glue.
    pub fn embed_ids_tensor(&self, ids: &[u32]) -> Tensor {
        let values = self.embed_ids(ids);
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for x in &values {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        Tensor {
            dtype: DType::F32,
            shape: vec![self.dim as u32],
            data: bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny 4-token, 2-dim table standing in for real weights.
    ///
    /// Rows are deliberately axis-aligned and unequal length so pooling and
    /// normalization have something to actually change.
    fn fixture(normalize: bool) -> StaticMatrix {
        StaticMatrix::new(
            vec![
                3.0, 4.0, // id 0: length 5
                1.0, 0.0, // id 1
                0.0, 2.0, // id 2
                -1.0, 0.0, // id 3
            ],
            4,
            2,
            normalize,
        )
        .unwrap()
    }

    fn decode(t: &Tensor) -> Vec<f32> {
        t.data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }

    #[test]
    fn mismatched_buffer_is_rejected() {
        // A weight file whose length disagrees with vocab*dim must not load.
        assert!(StaticMatrix::new(vec![1.0, 2.0, 3.0], 2, 2, true).is_none());
        assert!(StaticMatrix::new(vec![], 0, 2, true).is_none());
        assert!(StaticMatrix::new(vec![1.0, 2.0], 1, 0, true).is_none());
    }

    #[test]
    fn a_single_token_normalizes_to_unit_length() {
        // id 0 is (3,4), norm 5 → (0.6, 0.8).
        let v = fixture(true).embed_ids(&[0]);
        assert!((v[0] - 0.6).abs() < 1e-6, "{v:?}");
        assert!((v[1] - 0.8).abs() < 1e-6, "{v:?}");
        assert!((v.iter().map(|x| x * x).sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn pooling_is_the_mean_before_normalization() {
        // ids 1 and 2 → mean of (1,0) and (0,2) = (0.5, 1.0), unnormalized.
        let v = fixture(false).embed_ids(&[1, 2]);
        assert_eq!(v, vec![0.5, 1.0]);
    }

    #[test]
    fn normalization_is_applied_after_pooling() {
        // Same mean (0.5, 1.0), now L2-normalized: norm = sqrt(1.25).
        let v = fixture(true).embed_ids(&[1, 2]);
        let n = (0.25f32 + 1.0).sqrt();
        assert!((v[0] - 0.5 / n).abs() < 1e-6, "{v:?}");
        assert!((v[1] - 1.0 / n).abs() < 1e-6, "{v:?}");
    }

    #[test]
    fn out_of_range_ids_are_skipped_not_counted() {
        // id 99 does not exist; the result must equal embedding [1] alone.
        let only = fixture(false).embed_ids(&[1]);
        let with_junk = fixture(false).embed_ids(&[1, 99]);
        assert_eq!(only, with_junk);
    }

    #[test]
    fn no_usable_ids_yield_the_zero_vector() {
        assert_eq!(fixture(true).embed_ids(&[]), vec![0.0, 0.0]);
        assert_eq!(fixture(true).embed_ids(&[99, 100]), vec![0.0, 0.0]);
    }

    #[test]
    fn embedding_is_deterministic_for_the_same_ids() {
        // Tier-A-friendly: identical ids, byte-identical tensor.
        let m = fixture(true);
        assert_eq!(
            m.embed_ids_tensor(&[0, 1, 2]),
            m.embed_ids_tensor(&[0, 1, 2])
        );
    }

    #[test]
    fn tensor_encoding_matches_the_pooled_vector_and_shape() {
        let m = fixture(true);
        let t = m.embed_ids_tensor(&[1, 2]);
        assert_eq!(t.dtype, DType::F32);
        assert_eq!(t.shape, vec![2]);
        assert_eq!(t.data.len(), 8);
        t.validate().unwrap();
        assert_eq!(decode(&t), m.embed_ids(&[1, 2]));
    }
}
