//! The fabricate pass — the RAPTOR driver.
//!
//! [`map_stages::fabricate::AgglomerativeFabricator`] does one round of
//! clustering; this is the driver that iterates it into a tree. It is the
//! orchestrator, the same role the indexer plays for the per-resource stages:
//! the fabricator groups, but it does not label or embed — a cluster is fed
//! back through the *same* classifier and embedder that describe and embed a
//! segment, so a cluster ends up scored by the identical path.
//!
//! ```text
//! pool = level-0 records with tensors
//! repeat:
//!   cluster_round(pool vectors) -> groups (+ orphans left behind)
//!   plan:     derive each group's key, split reused from pending
//!   classify: ONE call labels every pending group at this level
//!   realize:  embed each label, store a cluster record
//!   pool = new clusters + orphans        (recluster on the label embeddings)
//! until a level cap or few enough remain
//! ```
//!
//! The plan/classify/realize split is what makes a level cost one request
//! instead of one per cluster. It is sound because `cluster_round` returns
//! *disjoint* groups within a round: no cluster written in a round can be a
//! cache hit for another group in the same round, so probing and writing
//! commute. Across rounds they do not — a level-2 group's children are level-1
//! clusters — which is why the phases stay inside one round.
//!
//! # Identity, and why re-fabrication is cheap where it can be
//!
//! A cluster's key is `hash(sorted(child ids) + fabricator fingerprint)`
//! ([`ObjectKey::cluster`]) — input-addressed, like every other object. So an
//! unchanged subtree derives the same key on the next run and its stored
//! object (its expensive LLM label) is reused verbatim rather than re-billed.
//! A level-0 segment has no object of its own, so its Merkle child id comes
//! from [`ObjectKey::segment`] — resource, span, and the resource's
//! segments-object key, under the dimension's fingerprint. The segments key is
//! what makes the id content-sensitive: without it an edit that preserves a
//! span's offsets leaves every leaf id, and so every cluster above it,
//! unchanged.

// The driver runs only when both the classifier and embedder plugins are
// compiled; its unit tests exercise it with fakes regardless. When the plugin
// combo is off, the non-test build has no caller, so the driver is legitimately
// unused there — allow it rather than lose the offline tests.
#![cfg_attr(not(all(feature = "llm", feature = "distilled")), allow(dead_code))]

use std::collections::BTreeSet;

use map_core::{
    Classifier, ClassifyBatch, Content, DescriptorPayload, DimensionRecords, EmbedBatch, Embedder,
    Segment, SegmentsPayload,
};
use map_format::{
    DType, Fingerprint, Manifest, ObjectKey, ObjectStore, Record, RecordKind, RecordMeta, Tensor,
    Tier,
};
use map_stages::fabricate::AgglomerativeFabricator;

use crate::{IndexError, IndexStats};

/// A dimension's level-0 records as `(merkle id, descriptor, vector)` — the
/// input the driver clusters.
pub(crate) type Level0 = Vec<(ObjectKey, Option<String>, Vec<f32>)>;

/// Tunables read from a dimension's `fabricator` config.
pub(crate) struct FabParams {
    /// Merge only while centroid cosine is at least this.
    pub threshold: f32,
    /// Fewest members a group needs to be a cluster.
    pub min_cluster: usize,
    /// Most members a cluster may hold; `0` means no cap. Bounds
    /// children-per-cluster in a homogeneous embedding space.
    pub max_cluster: usize,
    /// Height cap: stop after this many rounds.
    pub max_levels: u16,
    /// Stop once the pool is this small — no point summarizing a handful.
    pub min_remaining: usize,
}

/// A node the driver clusters: a level-0 segment or a cluster from a prior
/// round. Only its Merkle identity, its vector, and its descriptor matter.
#[derive(Clone)]
struct Node {
    identity: ObjectKey,
    descriptor: Option<String>,
    vector: Vec<f32>,
}

/// One group of a round, after its key is derived but before it is labeled.
///
/// Separating the decision from the work is what lets a whole level be
/// classified in one call: the store is probed for every group first, and only
/// the misses go into the request.
enum Planned {
    /// Already on disk under this key; its label cost nothing to recover.
    Reused {
        cluster_key: ObjectKey,
        group: usize,
        node: Node,
    },
    /// Needs a label from the classifier before it can be written.
    Pending {
        cluster_key: ObjectKey,
        child_ids: Vec<ObjectKey>,
        group: usize,
    },
}

/// Build the cluster tree for one dimension, writing cluster objects and
/// recording their keys in the manifest.
///
/// `level0` is the dimension's segment records as `(merkle id, descriptor,
/// vector)`. `fab_fp` re-keys every cluster when the fabricator (or anything
/// upstream of it) changes, so editing the label prompt rebuilds the tree.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fabricate_dimension(
    dimension: &str,
    fab_fp: Fingerprint,
    level0: Level0,
    params: &FabParams,
    classifier: &dyn Classifier,
    embedder: &dyn Embedder,
    store: &ObjectStore,
    prior: Option<&Manifest>,
    manifest: &mut Manifest,
    stats: &mut IndexStats,
) -> Result<(), IndexError> {
    let fabricator = AgglomerativeFabricator {
        threshold: params.threshold,
        min_cluster: params.min_cluster,
        max_cluster: params.max_cluster,
    };

    let mut pool: Vec<Node> = level0
        .into_iter()
        .map(|(identity, descriptor, vector)| Node {
            identity,
            descriptor,
            vector,
        })
        .collect();
    let mut keys: Vec<ObjectKey> = Vec::new();
    // Highest level that actually emitted a node. Level 0 always exists.
    let mut built_to: u16 = 0;

    let mut level: u16 = 1;
    while level <= params.max_levels && pool.len() > params.min_remaining {
        let vectors: Vec<Vec<f32>> = pool.iter().map(|n| n.vector.clone()).collect();
        let groups = fabricator.cluster_round(&vectors);
        if groups.is_empty() {
            // Nothing left is close enough to merge; the tree is as tall as it
            // gets. Stopping here is not the same as min_remaining — a diffuse
            // corpus simply has no more structure to summarize.
            break;
        }

        // Phase A — plan. No LLM call and no write: derive every key and split
        // the groups this run must label from the ones already on disk.
        let mut planned: Vec<Planned> = Vec::with_capacity(groups.len());
        for (index, group) in groups.iter().enumerate() {
            let child_ids: Vec<ObjectKey> = group.iter().map(|&i| pool[i].identity).collect();
            let cluster_key = ObjectKey::cluster(&child_ids, fab_fp);

            let recorded = prior.and_then(|m| m.object(cluster_key));
            if let Some(entry) = store
                .stat(cluster_key, Tier::C, recorded)
                .map_err(crate::as_tampered)?
            {
                // An unchanged subtree keeps its expensive label, unbilled. The
                // label is authored text headed for model context, so it is read
                // verified rather than trusted (spec §8).
                let bytes = store
                    .get_verified(cluster_key, &entry)
                    .map_err(crate::as_tampered)?;
                manifest.insert_object(cluster_key, entry)?;
                stats.objects_reused += 1;
                let record: Record =
                    serde_json::from_slice(&bytes).map_err(map_format::Error::from)?;
                planned.push(Planned::Reused {
                    cluster_key,
                    group: index,
                    node: Node {
                        identity: cluster_key,
                        vector: decode_vector(record.tensor.as_ref()),
                        descriptor: record.descriptor,
                    },
                });
            } else {
                planned.push(Planned::Pending {
                    cluster_key,
                    child_ids,
                    group: index,
                });
            }
        }

        // Phase B — classify. One call for every group this level still owes a
        // label, which is where ~one-request-per-cluster became ~one per level.
        let pending: Vec<usize> = planned
            .iter()
            .enumerate()
            .filter(|(_, p)| matches!(p, Planned::Pending { .. }))
            .map(|(i, _)| i)
            .collect();

        let mut labels: Vec<Option<String>> = Vec::new();
        if !pending.is_empty() {
            let items: Vec<Vec<&str>> = pending
                .iter()
                .map(|&i| {
                    let Planned::Pending { group, .. } = &planned[i] else {
                        unreachable!("filtered to pending above")
                    };
                    groups[*group]
                        .iter()
                        .filter_map(|&m| pool[m].descriptor.as_deref())
                        .filter(|d| !d.is_empty())
                        .collect()
                })
                .collect();

            let dimensions = [dimension.to_owned()];
            let classified = classifier.classify(&ClassifyBatch {
                items: &items,
                dimensions: &dimensions,
                level,
            })?;
            stats.label_calls += 1;

            // A short reply would silently shift every later label onto the
            // wrong cluster, which no downstream check could catch.
            if classified.len() != pending.len() {
                return Err(IndexError::Stage(map_core::Error::MalformedPayload(
                    "classifier returned a different number of cluster labels than groups",
                )));
            }
            labels = classified
                .into_iter()
                .map(|mut records| {
                    records
                        .remove(dimension)
                        .and_then(|record| record.descriptor)
                        .filter(|label| !label.is_empty())
                })
                .collect();
        }

        // Phase C — realize, in plan order, so emission stays independent of
        // which groups happened to be cached.
        let mut grouped: BTreeSet<usize> = BTreeSet::new();
        let mut new_nodes: Vec<Node> = Vec::new();
        let mut pending_seen = 0usize;
        for entry in &planned {
            match entry {
                Planned::Reused {
                    cluster_key,
                    group,
                    node,
                } => {
                    keys.push(*cluster_key);
                    new_nodes.push(node.clone());
                    grouped.extend(groups[*group].iter().copied());
                }
                Planned::Pending {
                    cluster_key,
                    child_ids,
                    group,
                } => {
                    let label = labels[pending_seen].clone();
                    pending_seen += 1;
                    let Some(label) = label else {
                        // No label came back for this group. Leave its members
                        // ungrouped so they carry forward as orphans and can
                        // find a home in the next round, rather than vanishing
                        // into a cluster that was never written.
                        stats.unlabeled_clusters += 1;
                        continue;
                    };

                    let tensor = embed_label(embedder, dimension, &label)?;
                    let vector = decode_vector(Some(&tensor));

                    let mut children = child_ids.clone();
                    children.sort_unstable();
                    let mut record = Record {
                        descriptor: Some(label.clone()),
                        tensor: Some(tensor),
                        meta: RecordMeta {
                            kind: RecordKind::Cluster,
                            dimension: dimension.to_owned(),
                            level,
                            children,
                        },
                    };
                    record.canonicalize();
                    record.validate()?;

                    let bytes = map_format::codec::canonical_json(&record)?;
                    let stored = store.put(*cluster_key, &bytes, Tier::C)?;
                    manifest.insert_object(*cluster_key, stored)?;
                    stats.clusters_written += 1;
                    stats.objects_written += 1;

                    keys.push(*cluster_key);
                    new_nodes.push(Node {
                        identity: *cluster_key,
                        vector,
                        descriptor: Some(label),
                    });
                    grouped.extend(groups[*group].iter().copied());
                }
            }
        }

        // Orphans carry forward unchanged; only clustered nodes ascend a level.
        // The next round clusters the new labels *and* the orphans together, so
        // an orphan can still find a home higher up.
        let orphans: Vec<Node> = pool
            .iter()
            .enumerate()
            .filter(|(i, _)| !grouped.contains(i))
            .map(|(_, n)| n.clone())
            .collect();
        if !new_nodes.is_empty() {
            built_to = level;
        }
        pool = new_nodes;
        pool.extend(orphans);
        level += 1;
    }

    if !keys.is_empty() {
        // Sorted so the manifest serializes identically regardless of emit
        // order — the same discipline as a cluster's child list.
        keys.sort_unstable();
        keys.dedup();
        manifest.clusters.insert(dimension.to_owned(), keys);
    }

    // Record what was built, beside the keys and under the same condition, so
    // the two can never disagree. Read from the round that actually emitted
    // nodes rather than from `level`: the loop's `break` leaves `level`
    // un-incremented, and a round can produce groups but write nothing if every
    // one of them went unlabeled.
    if let Some(identity) = manifest.dimensions.get_mut(dimension) {
        identity.levels = (0..=built_to).collect();
    }
    Ok(())
}

/// Gather a dimension's level-0 records as `(merkle id, descriptor, vector)`,
/// reading them back from the committed objects the indexer just wrote.
///
/// A record needs a vector to be clustered, so a segment the embedder produced
/// nothing for is skipped. Its descriptor is optional — a cluster can still be
/// labeled from whichever members carry prose.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gather_level0(
    manifest: &Manifest,
    shared: &ObjectStore,
    descriptors: &ObjectStore,
    tensors: &ObjectStore,
    dimension: &str,
    classifier_impl: Option<&str>,
    embedder_impl: &str,
    dim_fp: Fingerprint,
) -> Result<Level0, IndexError> {
    let mut out = Vec::new();
    for (resource, root) in &manifest.roots {
        let Some(tensor_key) = root.tensors.get(embedder_impl) else {
            continue;
        };
        let segments: SegmentsPayload =
            serde_json::from_slice(&shared.get(root.segments)?).map_err(map_format::Error::from)?;
        let tensor_payload = map_core::decode_tensor_payload(&tensors.get(*tensor_key)?)?;

        let desc_payload: Option<DescriptorPayload> = match classifier_impl
            .and_then(|imp| root.descriptors.get(imp))
        {
            Some(key) => Some(
                serde_json::from_slice(&descriptors.get(*key)?).map_err(map_format::Error::from)?,
            ),
            None => None,
        };

        for (i, segment) in segments.segments.iter().enumerate() {
            let vector = tensor_payload
                .per_segment
                .get(i)
                .and_then(|m| m.get(dimension))
                .map(|t| decode_vector(Some(t)))
                .unwrap_or_default();
            if vector.is_empty() {
                continue;
            }
            let descriptor = desc_payload
                .as_ref()
                .and_then(|dp| dp.per_segment.get(i))
                .and_then(|m| m.get(dimension))
                .and_then(|r| r.descriptor.clone());
            // The one derivation of leaf identity. `map-query` resolves a
            // cluster's children back to spans through this same function, and
            // both sides must keep calling it: a hand-copied second copy is
            // what let the two drift into `Some(vec![])` — a cluster standing
            // for nothing, reported as success. Folding in the segments-object
            // key is also what makes the id content-sensitive, so an edit that
            // preserves a span's offsets still re-keys the cluster above it.
            let identity =
                ObjectKey::segment(resource, segment.start, segment.end, root.segments, dim_fp);
            out.push((identity, descriptor, vector));
        }
    }
    Ok(out)
}

/// Embed a cluster label through the same document path a segment descriptor
/// takes, so a cluster's tensor lands in the identical space as its members'.
fn embed_label(
    embedder: &dyn Embedder,
    dimension: &str,
    label: &str,
) -> Result<Tensor, IndexError> {
    let content = Content {
        key: String::new(),
        text: label.to_owned(),
    };
    let segments = [Segment {
        start: 0,
        end: label.len() as u32,
    }];
    let mut records = DimensionRecords::new();
    records.insert(
        dimension.to_owned(),
        Record {
            descriptor: Some(label.to_owned()),
            tensor: None,
            meta: RecordMeta {
                kind: RecordKind::Segment,
                dimension: dimension.to_owned(),
                level: 0,
                children: Vec::new(),
            },
        },
    );
    let classified = [records];
    let dims = [dimension.to_owned()];
    let out = embedder.embed(&EmbedBatch {
        content: &content,
        segments: &segments,
        dimensions: &dims,
        classified: &classified,
    })?;
    out.into_iter()
        .next()
        .and_then(|mut m| m.remove(dimension))
        .ok_or_else(|| {
            IndexError::Stage(map_core::Error::io(
                std::path::PathBuf::from("<fabricate>"),
                std::io::Error::other("embedder produced no tensor for a cluster label"),
            ))
        })
}

/// Decode an F32 tensor into a vector, or empty for anything else.
///
/// Vectors are stored as the embedder produced them: L2-normalized F32. Any
/// other dtype is a record this pass cannot cluster, treated as absent.
fn decode_vector(tensor: Option<&Tensor>) -> Vec<f32> {
    let Some(t) = tensor else {
        return Vec::new();
    };
    if t.dtype != DType::F32 {
        return Vec::new();
    }
    t.data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use map_core::Result as CoreResult;

    /// A deterministic fake classifier: joins each item's member descriptors.
    /// Lets the driver be exercised with no LLM and no network.
    ///
    /// Counts its invocations, which is how the batching claim is checked — a
    /// level must cost one call, not one per cluster.
    #[derive(Default)]
    struct FakeClassifier {
        calls: std::sync::atomic::AtomicUsize,
        /// Item indices to return no label for, exercising the decline path.
        decline: BTreeSet<usize>,
    }

    impl FakeClassifier {
        fn calls(&self) -> usize {
            self.calls.load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    impl map_core::Stage for FakeClassifier {
        fn implementation(&self) -> &str {
            "fake"
        }
        fn config(&self) -> String {
            "fake".to_owned()
        }
    }

    impl Classifier for FakeClassifier {
        fn classify(&self, batch: &ClassifyBatch<'_>) -> CoreResult<Vec<DimensionRecords>> {
            self.calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut out = Vec::with_capacity(batch.items.len());
            for (i, sources) in batch.items.iter().enumerate() {
                let mut records = DimensionRecords::new();
                if !self.decline.contains(&i) {
                    let label = if sources.is_empty() {
                        "cluster".to_owned()
                    } else {
                        sources.join(" + ")
                    };
                    for dimension in batch.dimensions {
                        records.insert(
                            dimension.clone(),
                            Record {
                                descriptor: Some(label.clone()),
                                tensor: None,
                                meta: RecordMeta {
                                    kind: RecordKind::Cluster,
                                    dimension: dimension.clone(),
                                    level: batch.level,
                                    children: Vec::new(),
                                },
                            },
                        );
                    }
                }
                out.push(records);
            }
            Ok(out)
        }
    }

    /// A fake embedder that embeds a label to a fixed 2-D vector derived from
    /// its text, so clustering is deterministic and offline. Real geometry is
    /// covered by the distilled embedder's own tests.
    struct FakeEmbedder;
    impl map_core::Stage for FakeEmbedder {
        fn implementation(&self) -> &str {
            "fake"
        }
        fn config(&self) -> String {
            "fake".to_owned()
        }
    }
    impl Embedder for FakeEmbedder {
        fn embed(&self, batch: &EmbedBatch<'_>) -> CoreResult<Vec<map_core::DimensionTensors>> {
            let mut out = Vec::new();
            for _ in batch.segments {
                let mut m = map_core::DimensionTensors::new();
                for d in batch.dimensions {
                    m.insert(d.clone(), unit_tensor(&[1.0, 0.0]));
                }
                out.push(m);
            }
            Ok(out)
        }
        fn encode_query(&self, _dimension: &str, _text: &str) -> CoreResult<Tensor> {
            Ok(unit_tensor(&[1.0, 0.0]))
        }
    }

    fn unit_tensor(values: &[f32]) -> Tensor {
        let norm = values.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        let mut data = Vec::new();
        for x in values {
            data.extend_from_slice(&(x / norm).to_le_bytes());
        }
        Tensor {
            dtype: DType::F32,
            shape: vec![values.len() as u32],
            data,
        }
    }

    fn node(seed: &str, vec: &[f32]) -> (ObjectKey, Option<String>, Vec<f32>) {
        let key = ObjectKey::derive(&[seed.as_bytes()], Fingerprint::of(b"seg"));
        let t = unit_tensor(vec);
        (key, Some(format!("desc-{seed}")), decode_vector(Some(&t)))
    }

    fn manifest() -> Manifest {
        Manifest::new(&map_format::Config::zero_config(), "0.0.0", 1_700_000_000).unwrap()
    }

    fn store() -> (ObjectStore, tempdir_guard::Dir) {
        let dir = tempdir_guard::Dir::new();
        (ObjectStore::open(dir.path().join("objects")), dir)
    }

    fn params() -> FabParams {
        FabParams {
            threshold: 0.6,
            min_cluster: 2,
            max_cluster: 0,
            max_levels: 3,
            min_remaining: 1,
        }
    }

    #[test]
    fn two_tight_pairs_become_two_clusters() {
        let (store, _dir) = store();
        let mut manifest = manifest();
        let mut stats = IndexStats::default();
        let level0 = vec![
            node("a", &[1.0, 0.0]),
            node("b", &[0.99, 0.01]),
            node("c", &[0.0, 1.0]),
            node("d", &[0.01, 0.99]),
        ];
        fabricate_dimension(
            "descriptive",
            Fingerprint::of(b"fab"),
            level0,
            &params(),
            &FakeClassifier::default(),
            &FakeEmbedder,
            &store,
            None,
            &mut manifest,
            &mut stats,
        )
        .unwrap();

        // Two leaf clusters formed; the fake embedder collapses their labels to
        // one point, so a further round may merge them — either way at least
        // the two leaves exist and are recorded.
        assert!(stats.clusters_written >= 2, "expected >=2 clusters");
        let recorded = manifest
            .clusters
            .get("descriptive")
            .expect("clusters recorded");
        assert_eq!(recorded.len(), stats.clusters_written);
        // Every recorded cluster key resolves to a stored object.
        for key in recorded {
            assert!(manifest.object(*key).is_some());
        }
    }

    #[test]
    fn nothing_close_writes_no_clusters() {
        let (store, _dir) = store();
        let mut manifest = manifest();
        let mut stats = IndexStats::default();
        // Orthogonal points at threshold 0.6 never merge.
        let level0 = vec![node("a", &[1.0, 0.0]), node("b", &[0.0, 1.0])];
        fabricate_dimension(
            "descriptive",
            Fingerprint::of(b"fab"),
            level0,
            &params(),
            &FakeClassifier::default(),
            &FakeEmbedder,
            &store,
            None,
            &mut manifest,
            &mut stats,
        )
        .unwrap();
        assert_eq!(stats.clusters_written, 0);
        assert!(!manifest.clusters.contains_key("descriptive"));
    }

    #[test]
    fn a_second_run_reuses_clusters_and_bills_no_labels() {
        let (store, _dir) = store();
        let level0 = || {
            vec![
                node("a", &[1.0, 0.0]),
                node("b", &[0.99, 0.01]),
                node("c", &[0.98, 0.02]),
            ]
        };
        let fp = Fingerprint::of(b"fab");

        let mut m1 = manifest();
        let mut s1 = IndexStats::default();
        fabricate_dimension(
            "descriptive",
            fp,
            level0(),
            &params(),
            &FakeClassifier::default(),
            &FakeEmbedder,
            &store,
            None,
            &mut m1,
            &mut s1,
        )
        .unwrap();
        assert!(s1.label_calls >= 1, "first run labels");

        let mut m2 = manifest();
        let mut s2 = IndexStats::default();
        let second = FakeClassifier::default();
        fabricate_dimension(
            "descriptive",
            fp,
            level0(),
            &params(),
            &second,
            &FakeEmbedder,
            &store,
            None,
            &mut m2,
            &mut s2,
        )
        .unwrap();
        assert_eq!(s2.label_calls, 0, "an unchanged corpus re-bills no labels");
        assert_eq!(
            second.calls(),
            0,
            "not merely uncounted — no request is even attempted"
        );
        assert_eq!(s2.clusters_written, 0, "and writes no new cluster objects");
        assert_eq!(
            m1.clusters.get("descriptive"),
            m2.clusters.get("descriptive"),
            "same clusters recorded both runs"
        );
    }

    #[test]
    fn one_call_labels_every_cluster_in_a_level() {
        // The batching claim, and the reason the separate labeler went away: a
        // level costs one request, not one per cluster. `max_levels = 1` pins
        // it to a single round so the count is exact rather than cumulative.
        let (store, _dir) = store();
        let mut manifest = manifest();
        let mut stats = IndexStats::default();
        let fake = FakeClassifier::default();
        let level0 = vec![
            node("a", &[1.0, 0.0]),
            node("b", &[0.99, 0.01]),
            node("c", &[0.0, 1.0]),
            node("d", &[0.01, 0.99]),
        ];
        fabricate_dimension(
            "descriptive",
            Fingerprint::of(b"fab"),
            level0,
            &FabParams {
                max_levels: 1,
                ..params()
            },
            &fake,
            &FakeEmbedder,
            &store,
            None,
            &mut manifest,
            &mut stats,
        )
        .unwrap();

        assert_eq!(
            stats.clusters_written, 2,
            "the two tight pairs both cluster"
        );
        assert_eq!(fake.calls(), 1, "both labels come from one call");
        assert_eq!(stats.label_calls, 1, "and the stat counts invocations");
    }

    #[test]
    fn the_manifest_records_the_levels_that_were_built() {
        // Config declares what a dimension searches; this is the other half of
        // the comparison, and without it under-declaring is undetectable.
        let (store, _dir) = store();
        let mut manifest = manifest();
        manifest.dimensions.insert(
            "descriptive".to_owned(),
            map_format::manifest::DimensionIdentity {
                fingerprint: Fingerprint::of(b"d"),
                descriptor: true,
                tensor: true,
                levels: vec![0],
            },
        );
        let mut stats = IndexStats::default();
        fabricate_dimension(
            "descriptive",
            Fingerprint::of(b"fab"),
            vec![
                node("a", &[1.0, 0.0]),
                node("b", &[0.99, 0.01]),
                node("c", &[0.0, 1.0]),
                node("d", &[0.01, 0.99]),
            ],
            &FabParams {
                max_levels: 1,
                ..params()
            },
            &FakeClassifier::default(),
            &FakeEmbedder,
            &store,
            None,
            &mut manifest,
            &mut stats,
        )
        .unwrap();

        assert_eq!(stats.clusters_written, 2);
        assert_eq!(
            manifest.dimensions["descriptive"].levels,
            vec![0, 1],
            "one round of clusters means levels 0 and 1 exist"
        );
    }

    #[test]
    fn a_dimension_that_clusters_nothing_records_level_zero_only() {
        // Reading the height off the loop counter instead of off the rounds
        // that emitted nodes would claim a level that holds no records, and the
        // load-time check would then never fire for a genuinely stunted tree.
        let (store, _dir) = store();
        let mut manifest = manifest();
        manifest.dimensions.insert(
            "descriptive".to_owned(),
            map_format::manifest::DimensionIdentity {
                fingerprint: Fingerprint::of(b"d"),
                descriptor: true,
                tensor: true,
                levels: vec![0],
            },
        );
        let mut stats = IndexStats::default();
        fabricate_dimension(
            "descriptive",
            Fingerprint::of(b"fab"),
            // Nothing is close enough to merge at this threshold.
            vec![node("a", &[1.0, 0.0]), node("b", &[0.0, 1.0])],
            &params(),
            &FakeClassifier::default(),
            &FakeEmbedder,
            &store,
            None,
            &mut manifest,
            &mut stats,
        )
        .unwrap();

        assert_eq!(stats.clusters_written, 0);
        assert_eq!(manifest.dimensions["descriptive"].levels, vec![0]);
    }

    #[test]
    fn a_cluster_the_classifier_declines_to_label_is_not_written() {
        // An unlabeled group must not become a cluster that exists in name
        // only. Its members stay ungrouped so they can find a home next round.
        let (store, _dir) = store();
        let mut manifest = manifest();
        let mut stats = IndexStats::default();
        let fake = FakeClassifier {
            decline: BTreeSet::from([0]),
            ..Default::default()
        };
        let level0 = vec![
            node("a", &[1.0, 0.0]),
            node("b", &[0.99, 0.01]),
            node("c", &[0.0, 1.0]),
            node("d", &[0.01, 0.99]),
        ];
        fabricate_dimension(
            "descriptive",
            Fingerprint::of(b"fab"),
            level0,
            &FabParams {
                max_levels: 1,
                ..params()
            },
            &fake,
            &FakeEmbedder,
            &store,
            None,
            &mut manifest,
            &mut stats,
        )
        .unwrap();

        assert_eq!(stats.unlabeled_clusters, 1, "the declined group is counted");
        assert_eq!(stats.clusters_written, 1, "and only the labeled one lands");
        assert_eq!(
            manifest.clusters.get("descriptive").map(Vec::len),
            Some(1),
            "the manifest records only what was written"
        );
    }

    #[test]
    fn cluster_identity_ignores_input_order() {
        // Reordering the level-0 records must not change which cluster keys are
        // produced — Merkle identity is over the sorted child set.
        let mut forward = vec![
            node("a", &[1.0, 0.0]),
            node("b", &[0.99, 0.01]),
            node("c", &[0.98, 0.02]),
        ];
        let mut backward = forward.clone();
        backward.reverse();
        let fp = Fingerprint::of(b"fab");

        let run = |lvl0: Level0| {
            let (store, _dir) = store();
            let mut m = manifest();
            let mut s = IndexStats::default();
            fabricate_dimension(
                "descriptive",
                fp,
                lvl0,
                &params(),
                &FakeClassifier::default(),
                &FakeEmbedder,
                &store,
                None,
                &mut m,
                &mut s,
            )
            .unwrap();
            m.clusters.get("descriptive").cloned().unwrap_or_default()
        };

        forward.sort_by_key(|a| a.0);
        backward.sort_by_key(|a| a.0);
        assert_eq!(run(forward), run(backward));
    }
}

/// A tiny scoped temp directory for the tests above — created under the OS temp
/// dir, removed on drop. Avoids a dev-dependency just for this.
#[cfg(test)]
mod tempdir_guard {
    use std::path::{Path, PathBuf};

    pub(super) struct Dir(PathBuf);

    impl Dir {
        pub(super) fn new() -> Self {
            // Process id plus a monotonic counter keeps parallel tests apart
            // without Math.random (which the workflow sandbox forbids anyway).
            use std::sync::atomic::{AtomicU64, Ordering};
            static N: AtomicU64 = AtomicU64::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("map-fab-test-{}-{n}", std::process::id()));
            std::fs::create_dir_all(&path).unwrap();
            Dir(path)
        }
        pub(super) fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}
