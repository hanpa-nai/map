//! The identity classifier: content shaped for the embedder, stored nowhere.
//!
//! A `semantic` dimension matches raw content against raw content, so its
//! "descriptor" is the content itself. Naming that stage rather than leaving
//! the embedder to fall through to segment text makes the pipeline uniform —
//! every dimension is classifier → embedder → scorer — and gives the level
//! ladder somewhere to hang.
//!
//! It is paired with `persist_output = false`. Storing the output would write
//! the whole corpus into `desc/` a second time (3.17 MB against ripgrep's
//! 12.74 MB index) to avoid a transformation that is a string join.

use map_core::{Classifier, ClassifyBatch, DimensionRecords, Result, Stage};
use map_format::{Record, RecordKind, RecordMeta};

/// Joins an item's source texts and hands them on unchanged.
#[derive(Clone, Debug, Default)]
pub struct ContentClassifier;

impl Stage for ContentClassifier {
    fn implementation(&self) -> &str {
        "content"
    }

    fn config(&self) -> String {
        "content:v1".to_owned()
    }
}

impl Classifier for ContentClassifier {
    fn classify(&self, batch: &ClassifyBatch<'_>) -> Result<Vec<DimensionRecords>> {
        let mut out = Vec::with_capacity(batch.items.len());
        for sources in batch.items {
            // The same join the LLM classifier applies: one segment's text at
            // level 0, a group's descriptors above it.
            let descriptor = sources.join("\n\n");

            let mut per_dimension = DimensionRecords::new();
            for dimension in batch.dimensions {
                // An empty segment yields nothing rather than an empty
                // descriptor, which `Record::validate` rejects as a record
                // carrying neither payload.
                if descriptor.is_empty() {
                    continue;
                }
                per_dimension.insert(
                    dimension.clone(),
                    Record {
                        descriptor: Some(descriptor.clone()),
                        tensor: None,
                        meta: RecordMeta {
                            kind: if batch.level == 0 {
                                RecordKind::Segment
                            } else {
                                RecordKind::Cluster
                            },
                            dimension: dimension.clone(),
                            level: batch.level,
                            children: Vec::new(),
                            // Left None deliberately, as in every classifier:
                        },
                    },
                );
            }
            out.push(per_dimension);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classify(items: &[Vec<&str>], level: u16) -> Vec<DimensionRecords> {
        let dimensions = vec!["semantic".to_owned()];
        ContentClassifier
            .classify(&ClassifyBatch {
                items,
                dimensions: &dimensions,
                level,
            })
            .unwrap()
    }

    #[test]
    fn a_segment_is_passed_through_unchanged() {
        let out = classify(&[vec!["fn main() {}"]], 0);
        assert_eq!(
            out[0]["semantic"].descriptor.as_deref(),
            Some("fn main() {}")
        );
    }

    #[test]
    fn an_items_sources_are_joined_the_way_the_llm_classifier_joins_them() {
        // Divergence here would embed a cluster differently depending on which
        // classifier built it, in a space they are supposed to share.
        let out = classify(&[vec!["walks a directory", "matches ignore rules"]], 1);
        assert_eq!(
            out[0]["semantic"].descriptor.as_deref(),
            Some("walks a directory\n\nmatches ignore rules")
        );
    }

    #[test]
    fn a_record_above_level_zero_is_a_cluster() {
        let out = classify(&[vec!["a theme"]], 2);
        assert_eq!(out[0]["semantic"].meta.kind, RecordKind::Cluster);
        assert_eq!(out[0]["semantic"].meta.level, 2);
    }

    #[test]
    fn an_empty_item_yields_no_record_rather_than_an_empty_descriptor() {
        let out = classify(&[vec![]], 0);
        assert!(out[0].is_empty());
    }
}
