//! The real-weights loader for the distilled embedder — the `distilled` plugin.
//!
//! Gated behind the `distilled` feature so a lexical-only build never
//! compiles the tokenizer or the safetensors reader. This turns the
//! [`StaticMatrix`] math core into a working
//! [`Embedder`] by attaching the model's tokenizer and weight table:
//!
//! ```text
//! model dir/                        DistilledEmbedder
//!   config.json      ─ dim, normalize ─┐
//!   model.safetensors ─ "embeddings" ──┼─▶ StaticMatrix
//!   tokenizer.json   ─ BGE WordPiece ──┴─▶ Tokenizer
//! ```
//!
//! The weights are `minishlab/potion-retrieval-32M` (MIT): tensor `embeddings`,
//! F32, `[63091, 512]`, loaded from [`model_dir`]. This module only ever reads
//! what is already on disk; downloading lives in the `fetch` module behind the
//! separate `auto-distilled` feature — not linked here, because under
//! `distilled` alone it does not exist — so a `distilled` build links no HTTP
//! stack. Populate the directory with `map model fetch`, or by hand.

use std::path::Path;

use map_core::{DimensionTensors, EmbedBatch, Embedder, Error, Result, Segment, Stage};
use map_format::Tensor;
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::StaticMatrix;

/// The subset of `config.json` that changes inference.
#[derive(Deserialize)]
struct ModelConfig {
    hidden_dim: usize,
    #[serde(default = "default_true")]
    normalize: bool,
}

fn default_true() -> bool {
    true
}

/// The model's directory name under `~/.map/models`, and the id folded into
/// the embedder's fingerprint.
pub const MODEL_ID: &str = "potion-retrieval-32M";

/// `~/.map/models/potion-retrieval-32M` — the one place the loader looks.
///
/// Defined here rather than at each call site so the loader, the indexer's
/// availability check, and `map model fetch` cannot drift onto different
/// directories.
pub fn model_dir() -> std::path::PathBuf {
    let home = std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(std::path::PathBuf::from)
        .unwrap_or_default();
    home.join(".map").join("models").join(MODEL_ID)
}

/// A loaded static distilled embedder: weight table plus its tokenizer.
pub struct DistilledEmbedder {
    matrix: StaticMatrix,
    tokenizer: Tokenizer,
    /// Model identity, folded into the config fingerprint so two indices built
    /// with different embedders never silently merge.
    id: String,
}

fn io_err(path: &Path, e: impl std::fmt::Display) -> Error {
    Error::io(path, std::io::Error::other(e.to_string()))
}

impl DistilledEmbedder {
    /// Load `config.json`, `model.safetensors`, and `tokenizer.json` from a
    /// directory.
    ///
    /// `id` names this model for the fingerprint — e.g. `potion-retrieval-32M`.
    pub fn from_dir(dir: &Path, id: impl Into<String>) -> Result<Self> {
        let config_path = dir.join("config.json");
        let config_text =
            std::fs::read_to_string(&config_path).map_err(|e| Error::io(&config_path, e))?;
        let config: ModelConfig =
            serde_json::from_str(&config_text).map_err(|e| io_err(&config_path, e))?;

        let matrix = load_matrix(
            &dir.join("model.safetensors"),
            config.hidden_dim,
            config.normalize,
        )?;

        let tok_path = dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| io_err(&tok_path, e))?;

        Ok(DistilledEmbedder {
            matrix,
            tokenizer,
            id: id.into(),
        })
    }

    pub fn dim(&self) -> usize {
        self.matrix.dim()
    }

    pub fn vocab(&self) -> usize {
        self.matrix.vocab()
    }

    /// Tokenize and pool one string into a tensor.
    ///
    /// Special tokens ([CLS]/[SEP]) are **not** added: the static model pools
    /// content-token vectors, and injecting sentinels would tug every segment's
    /// vector toward the same two rows. This matches Model2Vec's own encode.
    fn embed_text(&self, text: &str) -> Result<Tensor> {
        let encoding = self.tokenizer.encode(text, false).map_err(|e| {
            Error::io(
                Path::new("<tokenizer>"),
                std::io::Error::other(e.to_string()),
            )
        })?;
        Ok(self.matrix.embed_ids_tensor(encoding.get_ids()))
    }
}

/// Read the `embeddings` tensor from a safetensors file into a [`StaticMatrix`].
fn load_matrix(path: &Path, expected_dim: usize, normalize: bool) -> Result<StaticMatrix> {
    let bytes = std::fs::read(path).map_err(|e| Error::io(path, e))?;
    let tensors = safetensors::SafeTensors::deserialize(&bytes).map_err(|e| io_err(path, e))?;
    let view = tensors.tensor("embeddings").map_err(|e| io_err(path, e))?;

    if view.dtype() != safetensors::Dtype::F32 {
        return Err(io_err(path, "embeddings tensor is not F32"));
    }
    let shape = view.shape();
    if shape.len() != 2 || shape[1] != expected_dim {
        return Err(io_err(
            path,
            format!("embeddings shape {shape:?} does not match config hidden_dim {expected_dim}"),
        ));
    }
    let (vocab, dim) = (shape[0], shape[1]);

    let data: Vec<f32> = view
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();

    StaticMatrix::new(data, vocab, dim, normalize)
        .ok_or_else(|| io_err(path, "weight buffer length disagrees with declared shape"))
}

impl Stage for DistilledEmbedder {
    fn implementation(&self) -> &str {
        "distilled"
    }

    fn config(&self) -> String {
        // The model identity and dimension are what change stored vectors, so
        // they are the fingerprint. Two embedders differing in either must key
        // their objects apart.
        format!("distilled:{}:dim={}", self.id, self.matrix.dim())
    }
}

impl Embedder for DistilledEmbedder {
    /// Embed the classifier's descriptor where one exists, the raw segment text
    /// otherwise.
    ///
    /// A pure-embedding dimension (an embedder with no classifier) has no
    /// descriptor, so it embeds raw content exactly as before. A semantic
    /// dimension — classifier *and* embedder — embeds the prose the classifier
    /// wrote, which is also what lets a cluster's label become its tensor: the
    /// fabricate pass feeds the label in through `classified` and gets it
    /// embedded by this same path.
    fn embed(&self, batch: &EmbedBatch<'_>) -> Result<Vec<DimensionTensors>> {
        let mut out = Vec::with_capacity(batch.segments.len());
        for (i, segment) in batch.segments.iter().enumerate() {
            // Raw text is shared across dimensions that have no descriptor, so
            // it is embedded at most once per segment — and not at all when
            // every dimension in the batch brought one, which is the case for
            // any classifier+embedder dimension. Embedding it eagerly was a
            // whole extra tokenize-and-pool per segment, always discarded.
            let mut raw: Option<Tensor> = None;
            let mut per_dimension = DimensionTensors::new();
            for dimension in batch.dimensions {
                let descriptor = batch
                    .classified
                    .get(i)
                    .and_then(|records| records.get(dimension))
                    .and_then(|record| record.descriptor.as_deref())
                    .filter(|d| !d.is_empty());
                let tensor = match descriptor {
                    Some(text) => self.embed_text(text)?,
                    None => match &raw {
                        Some(cached) => cached.clone(),
                        None => {
                            let encoded = self.embed_text(slice(batch, segment))?;
                            raw = Some(encoded.clone());
                            encoded
                        }
                    },
                };
                per_dimension.insert(dimension.clone(), tensor);
            }
            out.push(per_dimension);
        }
        Ok(out)
    }

    fn encode_query(&self, _dimension: &str, text: &str) -> Result<Tensor> {
        self.embed_text(text)
    }
}

/// Borrow a segment's text out of the batch content.
fn slice<'a>(batch: &EmbedBatch<'a>, segment: &Segment) -> &'a str {
    &batch.content.text[segment.start as usize..segment.end as usize]
}

#[cfg(test)]
mod tests {
    use super::*;
    use map_core::Content;

    /// The real model, from the dev cache. Skips (does not fail) when absent so
    /// CI without the ~129MB weights still passes; a machine that has run the
    /// fetch exercises the real thing.
    fn model() -> Option<DistilledEmbedder> {
        let dir = dirs_home()?.join(".map/models/potion-retrieval-32M");
        if !dir.join("model.safetensors").is_file() {
            eprintln!("skipping: model not present at {}", dir.display());
            return None;
        }
        Some(DistilledEmbedder::from_dir(&dir, "potion-retrieval-32M").expect("load model"))
    }

    fn dirs_home() -> Option<std::path::PathBuf> {
        std::env::var_os("USERPROFILE")
            .or_else(|| std::env::var_os("HOME"))
            .map(std::path::PathBuf::from)
    }

    fn embed(m: &DistilledEmbedder, text: &str) -> Vec<f32> {
        let t = m.encode_query("d", text).unwrap();
        t.data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn real_model_has_the_expected_shape() {
        let Some(m) = model() else { return };
        assert_eq!(m.dim(), 512);
        assert_eq!(m.vocab(), 63091);
    }

    #[test]
    fn real_embeddings_are_unit_length() {
        let Some(m) = model() else { return };
        let v = embed(&m, "how does binary detection work");
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "norm was {norm}");
    }

    #[test]
    fn it_is_actually_semantic_not_lexical() {
        // The whole reason this exists: related words that SHARE NO CHARACTERS
        // must land closer than unrelated ones. BM25 and the hashing embedder
        // both fail this; a real distilled model must pass it.
        let Some(m) = model() else { return };
        let database = embed(&m, "database");
        let sql = embed(&m, "sql query table");
        let mountain = embed(&m, "mountain hiking trail");

        let related = cosine(&database, &sql);
        let unrelated = cosine(&database, &mountain);
        assert!(
            related > unrelated,
            "expected sql closer to database than mountain: {related} vs {unrelated}"
        );
    }

    #[test]
    fn embeds_the_descriptor_when_the_classifier_produced_one() {
        // A semantic segment must be embedded by its prose descriptor, not its
        // raw content — that is what makes the tensor match the query by meaning,
        // and it is the same path a cluster label rides through.
        let Some(m) = model() else { return };
        use map_core::DimensionRecords;
        use map_format::{Record, RecordKind, RecordMeta};

        let c = Content {
            key: "t.rs".into(),
            text: "fn qzx() { let a = 1; }".into(),
        };
        let segs = [Segment {
            start: 0,
            end: c.text.len() as u32,
        }];
        let mut records = DimensionRecords::new();
        records.insert(
            "descriptive".to_owned(),
            Record {
                descriptor: Some("renews an expired login session".into()),
                tensor: None,
                meta: RecordMeta {
                    kind: RecordKind::Segment,
                    dimension: "descriptive".into(),
                    level: 0,
                    children: vec![],
                },
            },
        );
        let classified = [records];
        let batch = EmbedBatch {
            content: &c,
            segments: &segs,
            dimensions: &["descriptive".to_owned()],
            classified: &classified,
        };
        let embedded = m.embed(&batch).unwrap();
        let with_desc = &embedded[0]["descriptive"];
        assert_eq!(
            with_desc,
            &m.encode_query("descriptive", "renews an expired login session")
                .unwrap(),
            "must embed the descriptor text"
        );
        assert_ne!(
            with_desc,
            &m.encode_query("descriptive", &c.text).unwrap(),
            "must not embed the raw content when a descriptor exists"
        );
    }

    #[test]
    fn embedding_is_deterministic() {
        let Some(m) = model() else { return };
        let c = Content {
            key: "t.rs".into(),
            text: "fn refresh_token() {}".into(),
        };
        let segs = [Segment {
            start: 0,
            end: c.text.len() as u32,
        }];
        let batch = EmbedBatch {
            content: &c,
            segments: &segs,
            dimensions: &["descriptive".to_owned()],
            classified: &[],
        };
        assert_eq!(m.embed(&batch).unwrap(), m.embed(&batch).unwrap());
    }
}
