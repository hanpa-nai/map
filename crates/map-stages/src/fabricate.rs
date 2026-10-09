//! One round of agglomerative clustering; the driver iterates it into a tree.
//!
//! The fabricator does **not** build the whole tree in one shot. It clusters the
//! current top-level nodes into groups, leaving any node that fits nothing as an
//! *orphan*. The driver then labels and embeds each new cluster — through the
//! same classifier and embedder that process segments — adds those cluster
//! nodes back into the pool alongside the orphans, and calls another round,
//! now clustering on the **label** embeddings. It repeats until a level cap or
//! few enough nodes remain.
//!
//! # Why re-embedding the label costs nothing extra
//!
//! Every cluster is labeled and embedded once regardless — that is what gives it
//! a descriptor and a tensor to be retrieved by. Clustering the next level on
//! those label embeddings *reuses the same vectors*; it does not compute a
//! second, centroid-based embedding. So the faithful "cluster on the summary
//! embedding" behaviour is the same LLM/embed cost as clustering on centroids,
//! and it keeps the grouping and the scoring in one space.

/// Bottom-up agglomerative clustering, one round.
#[derive(Clone, Copy, Debug)]
pub struct AgglomerativeFabricator {
    /// Merge the two closest groups only while their centroid cosine is at
    /// least this. A node that never reaches it with anything stays an orphan,
    /// which the driver carries forward unchanged.
    pub threshold: f32,
    /// A group needs at least this many members to become a cluster; a group
    /// that never reaches it is left as orphans, not emitted.
    pub min_cluster: usize,
    /// Refuse a merge that would grow a group beyond this many members; `0`
    /// means no cap. A threshold alone cannot bound size in a homogeneous
    /// embedding space — LLM-written descriptions cluster so tightly that a
    /// reasonable threshold still merges hundreds into one blob — so the cap is
    /// what actually holds children-per-cluster in a target band. The pair is
    /// simply skipped, so its members stay free to form other clusters.
    pub max_cluster: usize,
}

impl Default for AgglomerativeFabricator {
    fn default() -> Self {
        // Tuned on ripgrep's descriptive embeddings (2,481 segments): 0.70 forms
        // clusters by similarity rather than by hitting the cap, and the 5..15
        // band keeps children-per-cluster reasonable. All embedding-dependent —
        // re-tune against a new corpus.
        AgglomerativeFabricator {
            threshold: 0.70,
            min_cluster: 5,
            max_cluster: 15,
        }
    }
}

impl AgglomerativeFabricator {
    /// Cluster `embeddings` (each L2-normalized) into groups for one level.
    ///
    /// Merges the two groups with the highest centroid cosine repeatedly, while
    /// that cosine is at least `threshold`. Returns the groups with at least
    /// `min_cluster` members, as lists of indices into `embeddings`. **Indices
    /// in no returned group are orphans** — the driver keeps them for the next
    /// round. Ties break toward the lowest index pair, so the result is
    /// deterministic.
    pub fn cluster_round(&self, embeddings: &[Vec<f32>]) -> Vec<Vec<usize>> {
        let n = embeddings.len();
        if n < self.min_cluster.max(2) {
            return Vec::new();
        }

        // Cache the full centroid-similarity matrix. The straightforward form
        // recomputes every pair's dot product on every merge — O(n³·d), which at
        // corpus scale (n≈2500, d=512) is tens of minutes to hours. Caching pulls
        // the d-dimensional dot out of the repeated "closest pair" scan, leaving
        // O(n³) scalar comparisons (~seconds), and a merge only has to refresh
        // the one row that changed. A merged group is marked dead rather than
        // removed, so indices stay stable and the output is a deterministic
        // function of the input bytes (Tier A) — verified against the naive form
        // in the tests.
        let cap = if self.max_cluster == 0 {
            usize::MAX
        } else {
            self.max_cluster
        };

        let mut groups: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        let mut centroids: Vec<Vec<f32>> = embeddings.to_vec();
        let mut alive = vec![true; n];
        let mut alive_count = n;

        let mut sim = vec![f32::NEG_INFINITY; n * n];
        for i in 0..n {
            for j in (i + 1)..n {
                let s = dot(&centroids[i], &centroids[j]);
                sim[i * n + j] = s;
                sim[j * n + i] = s;
            }
        }

        while alive_count >= 2 {
            // Closest live pair whose merge stays within the size cap. The
            // ascending scan with a strict `>` breaks ties toward the lowest
            // (i, j), so the merge order is deterministic.
            let (mut best_i, mut best_j, mut best_sim) =
                (usize::MAX, usize::MAX, f32::NEG_INFINITY);
            for i in 0..n {
                if !alive[i] {
                    continue;
                }
                let row = i * n;
                for j in (i + 1)..n {
                    if alive[j]
                        && sim[row + j] > best_sim
                        && groups[i].len() + groups[j].len() <= cap
                    {
                        best_sim = sim[row + j];
                        best_i = i;
                        best_j = j;
                    }
                }
            }
            // Below the threshold, nothing left is close enough — the rest are
            // orphans.
            if best_i == usize::MAX || best_sim < self.threshold {
                break;
            }

            // Merge the higher index into the lower; the survivor keeps the
            // group's smallest original member as its id.
            let moved = std::mem::take(&mut groups[best_j]);
            groups[best_i].extend(moved);
            centroids[best_i] =
                normalized_mean(groups[best_i].iter().map(|&k| embeddings[k].as_slice()));
            alive[best_j] = false;
            alive_count -= 1;

            // Only the merged group's similarities changed; refresh its row.
            for k in 0..n {
                if !alive[k] || k == best_i {
                    continue;
                }
                let s = dot(&centroids[best_i], &centroids[k]);
                sim[best_i * n + k] = s;
                sim[k * n + best_i] = s;
            }
        }

        groups
            .into_iter()
            .enumerate()
            .filter(|(i, g)| alive[*i] && g.len() >= self.min_cluster)
            .map(|(_, g)| g)
            .collect()
    }
}

/// L2-normalized mean of a set of equal-length vectors.
fn normalized_mean<'a>(vectors: impl Iterator<Item = &'a [f32]>) -> Vec<f32> {
    let mut sum: Vec<f32> = Vec::new();
    let mut count = 0u32;
    for v in vectors {
        if sum.is_empty() {
            sum = vec![0.0; v.len()];
        }
        for (acc, x) in sum.iter_mut().zip(v) {
            *acc += *x;
        }
        count += 1;
    }
    if count == 0 {
        return sum;
    }
    let inv = 1.0 / count as f32;
    for x in &mut sum {
        *x *= inv;
    }
    let norm = sum.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut sum {
            *x /= norm;
        }
    }
    sum
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn unit(e: &[f32]) -> Vec<f32> {
        let norm = e.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        e.iter().map(|x| x / norm).collect()
    }

    /// Two tight pairs on orthogonal axes, plus one point close to neither.
    fn two_pairs_and_an_orphan() -> Vec<Vec<f32>> {
        vec![
            unit(&[1.0, 0.0, 0.0]),
            unit(&[0.98, 0.02, 0.0]),
            unit(&[0.0, 1.0, 0.0]),
            unit(&[0.02, 0.98, 0.0]),
            unit(&[0.0, 0.0, 1.0]), // orphan
        ]
    }

    fn grouped(groups: &[Vec<usize>]) -> BTreeSet<usize> {
        groups.iter().flatten().copied().collect()
    }

    #[test]
    fn merges_the_tight_pairs_and_leaves_the_orphan() {
        let fab = AgglomerativeFabricator {
            threshold: 0.6,
            min_cluster: 2,
            max_cluster: 0,
        };
        let groups = fab.cluster_round(&two_pairs_and_an_orphan());

        assert_eq!(groups.len(), 2, "two clusters, not the orphan");
        let sorted: Vec<Vec<usize>> = groups
            .iter()
            .map(|g| {
                let mut g = g.clone();
                g.sort_unstable();
                g
            })
            .collect();
        assert!(sorted.contains(&vec![0, 1]), "x-axis pair: {sorted:?}");
        assert!(sorted.contains(&vec![2, 3]), "y-axis pair: {sorted:?}");
        // Index 4 is an orphan — in no group, carried forward by the driver.
        assert!(
            !grouped(&groups).contains(&4),
            "the distant point stays an orphan"
        );
    }

    #[test]
    fn nothing_close_clusters_nothing() {
        // Three mutually orthogonal points: no pair reaches the threshold, so
        // every node stays an orphan and no cluster is emitted.
        let fab = AgglomerativeFabricator {
            threshold: 0.5,
            min_cluster: 2,
            max_cluster: 0,
        };
        let orthogonal = vec![
            unit(&[1.0, 0.0, 0.0]),
            unit(&[0.0, 1.0, 0.0]),
            unit(&[0.0, 0.0, 1.0]),
        ];
        assert!(fab.cluster_round(&orthogonal).is_empty());
    }

    #[test]
    fn three_close_points_form_one_cluster() {
        let fab = AgglomerativeFabricator {
            threshold: 0.6,
            min_cluster: 2,
            max_cluster: 0,
        };
        let close = vec![unit(&[1.0, 0.0]), unit(&[0.99, 0.01]), unit(&[0.98, 0.02])];
        let groups = fab.cluster_round(&close);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 3);
    }

    #[test]
    fn the_size_cap_splits_a_would_be_blob() {
        // Ten near-identical points would all merge into one group; a cap of 3
        // forces them into groups of at most 3 instead — the whole point of the
        // cap in a homogeneous space.
        let fab = AgglomerativeFabricator {
            threshold: 0.6,
            min_cluster: 2,
            max_cluster: 3,
        };
        let blob: Vec<Vec<f32>> = (0..10).map(|i| unit(&[1.0, i as f32 * 0.001])).collect();
        let groups = fab.cluster_round(&blob);
        assert!(
            groups.iter().all(|g| g.len() <= 3),
            "cap exceeded: {groups:?}"
        );
        assert!(
            groups.iter().all(|g| g.len() >= 2),
            "min not met: {groups:?}"
        );
    }

    #[test]
    fn clustering_is_deterministic() {
        let fab = AgglomerativeFabricator {
            threshold: 0.6,
            min_cluster: 2,
            max_cluster: 0,
        };
        let e = two_pairs_and_an_orphan();
        assert_eq!(fab.cluster_round(&e), fab.cluster_round(&e));
    }

    /// The straightforward O(n³·d) version, kept only as a reference oracle: the
    /// cached fast path must agree with it on tie-free inputs.
    fn naive(fab: &AgglomerativeFabricator, embeddings: &[Vec<f32>]) -> Vec<Vec<usize>> {
        if embeddings.len() < fab.min_cluster.max(2) {
            return Vec::new();
        }
        let cap = if fab.max_cluster == 0 {
            usize::MAX
        } else {
            fab.max_cluster
        };
        let mut groups: Vec<Vec<usize>> = (0..embeddings.len()).map(|i| vec![i]).collect();
        let mut centroids: Vec<Vec<f32>> = embeddings.to_vec();
        while groups.len() >= 2 {
            let (mut bi, mut bj, mut bs) = (usize::MAX, usize::MAX, f32::NEG_INFINITY);
            for i in 0..groups.len() {
                for j in (i + 1)..groups.len() {
                    let s = dot(&centroids[i], &centroids[j]);
                    if s > bs && groups[i].len() + groups[j].len() <= cap {
                        bs = s;
                        bi = i;
                        bj = j;
                    }
                }
            }
            if bi == usize::MAX || bs < fab.threshold {
                break;
            }
            let moved = std::mem::take(&mut groups[bj]);
            groups[bi].extend(moved);
            centroids[bi] = normalized_mean(groups[bi].iter().map(|&k| embeddings[k].as_slice()));
            groups.remove(bj);
            centroids.remove(bj);
        }
        groups
            .into_iter()
            .filter(|g| g.len() >= fab.min_cluster)
            .collect()
    }

    /// Deterministic pseudo-random unit vectors — an xorshift, so no `rand` dep
    /// and no `Math.random`.
    fn random_units(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut state = seed.wrapping_add(0x9e37_79b9_7f4a_7c15) | 1;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| {
                let v: Vec<f32> = (0..d).map(|_| (next() * 2.0 - 1.0) as f32).collect();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
                v.iter().map(|x| x / norm).collect()
            })
            .collect()
    }

    fn canonical(mut groups: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
        for g in &mut groups {
            g.sort_unstable();
        }
        groups.sort();
        groups
    }

    #[test]
    fn fast_path_matches_the_naive_reference_on_random_data() {
        // Float inputs have no exact ties, so the merge sequence is unique and
        // the cached path must reproduce the naive grouping exactly. This is the
        // guard that the O(n³·d) → O(n³) rewrite changed only the cost.
        // Both an uncapped and a capped configuration, so the cap's effect on
        // the fast path is checked against the reference too.
        for max_cluster in [0usize, 6] {
            let fab = AgglomerativeFabricator {
                threshold: 0.3,
                min_cluster: 2,
                max_cluster,
            };
            for seed in 0..6u64 {
                let emb = random_units(120, 8, seed);
                assert_eq!(
                    canonical(fab.cluster_round(&emb)),
                    canonical(naive(&fab, &emb)),
                    "cached and naive diverged on seed {seed}, cap {max_cluster}"
                );
            }
        }
    }

    #[test]
    fn too_few_nodes_cluster_nothing() {
        let fab = AgglomerativeFabricator::default();
        assert!(fab.cluster_round(&[unit(&[1.0, 0.0])]).is_empty());
        assert!(fab.cluster_round(&[]).is_empty());
    }

    #[test]
    fn min_cluster_drops_lone_merges() {
        // With min_cluster = 3, a pair is not enough to be a cluster.
        let fab = AgglomerativeFabricator {
            threshold: 0.6,
            min_cluster: 3,
            max_cluster: 0,
        };
        let groups = fab.cluster_round(&two_pairs_and_an_orphan());
        assert!(
            groups.is_empty(),
            "pairs are below min_cluster=3, so all orphan"
        );
    }
}
