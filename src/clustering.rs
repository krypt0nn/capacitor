// SPDX-License-Identifier: GPL-3.0-or-later
//
// capacitor
// Copyright (C) 2025 - 2026  Nikita Podvirnyi <krypt0nn@vk.ru>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

use std::collections::{HashMap, HashSet};
use std::cmp::Ordering;

use rand_chacha::rand_core::Rng;
use rayon::prelude::*;

/// Distinct tokens are addressed by their `u16` value directly - there can be
/// at most 65536 of them, so all dense buffers are preallocated to that size.
const MAX_TOKENS: usize = u16::MAX as usize + 1;

#[derive(Debug, Clone)]
pub struct Cluster {
    ranks: HashMap<u16, f32>
}

impl Cluster {
    #[inline]
    pub fn new(ranks: impl IntoIterator<Item = (u16, f32)>) -> Self {
        Self {
            ranks: HashMap::from_iter(ranks)
        }
    }

    #[inline]
    pub fn into_inner(self) -> HashMap<u16, f32> {
        self.ranks
    }

    /// Calculate similarity between the current cluster to the given document.
    pub fn similarity(&self, document: impl IntoIterator<Item = u16>) -> f32 {
        let mut similarity = 0.0;

        for token in document {
            if let Some(rank) = self.ranks.get(&token) {
                similarity += *rank;
            }
        }

        similarity
    }
}

impl From<HashMap<u16, f32>> for Cluster {
    #[inline(always)]
    fn from(ranks: HashMap<u16, f32>) -> Self {
        Self { ranks }
    }
}

/// Distinct tokens of a document with their occurrence counts.
type Profile = Box<[(u16, u32)]>;

/// Clusterize documents into semantic groups using weighted farthest-point
/// centroid sampling over sparse token rank vectors.
///
/// Returns the clusters alongside per-document index of the most similar
/// cluster (the assignment is computed during clusterization itself).
#[allow(clippy::type_complexity)]
pub fn clusterize<R: Rng>(
    mut clusters_num: usize,
    mut centroids_num: usize,
    documents: impl AsRef<[Box<[u16]>]>,
    mut rng: R
) -> anyhow::Result<(Box<[Cluster]>, Box<[usize]>)> {
    let documents = documents.as_ref();

    if clusters_num == 0 {
        clusters_num = documents.len().isqrt().max(1);
    }

    if centroids_num == 0 {
        centroids_num = clusters_num.isqrt().max(1);
    }

    // Pool of documents eligible for centroid sampling, referenced by their
    // index in the input documents slice.
    //
    // Keep only long documents for better expert representation. Fall back to
    // the whole corpus when there are too few long documents.

    let mut lengths = documents.iter()
        .map(|document| document.len())
        .collect::<Vec<usize>>();

    let median_len = if lengths.is_empty() {
        0
    } else {
        let n = lengths.len();

        let (_, median, _) = lengths.select_nth_unstable(n / 2);

        *median
    };

    let mut remaining = documents.iter()
        .enumerate()
        .filter(|(_, document)| document.len() > median_len / 3)
        .map(|(i, _)| i as u32)
        .collect::<Vec<u32>>();

    // Fall back to using all documents if there's too few long ones.

    if clusters_num * centroids_num > remaining.len() {
        remaining = (0..documents.len() as u32).collect();
    }

    if clusters_num * centroids_num > remaining.len() {
        anyhow::bail!("clusters_num * centroids_num must be lower or equal to the documents amount");
    }

    // Count distinct token occurrences of every document. One scratch counter
    // buffer indexed by token value is reused across documents.

    let mut counters = vec![0_u32; MAX_TOKENS];
    let mut touched = Vec::<u16>::new();

    let mut profiles = Vec::<Profile>::with_capacity(documents.len());
    let mut total_tokens = 0;

    for document in documents {
        touched.clear();

        for &token in document.iter() {
            let count = &mut counters[token as usize];

            if *count == 0 {
                touched.push(token);
            }

            *count += 1;
        }

        total_tokens += document.len();

        let mut profile = Vec::with_capacity(touched.len());

        for &token in &touched {
            profile.push((token, counters[token as usize]));
        }

        profiles.push(profile.into_boxed_slice());

        for &token in &touched {
            counters[token as usize] = 0;
        }
    }

    // Appearance-in-documents frequency of every token.

    let mut doc_appearances = vec![0_u32; MAX_TOKENS];

    for profile in &profiles {
        for &(token, _) in profile.iter() {
            doc_appearances[token as usize] += 1;
        }
    }

    // Negated log2 appearance frequency of every token.

    let neg_log_tf = doc_appearances.iter()
        .map(|&count| -((count.max(1) as f32 / total_tokens.max(1) as f32).log2()))
        .collect::<Vec<f32>>();

    drop(doc_appearances);

    // Dense working buffers reused by every clustering round: accumulated
    // frequencies, resulting ranks and non-zero rank members.

    let mut acc = vec![0.0_f32; MAX_TOKENS];
    let mut ranks = vec![0.0_f32; MAX_TOKENS];
    let mut members = Vec::<u16>::with_capacity(MAX_TOKENS);

    // Cumulative products of similarities to the clusters formed so far -
    // one entry per document. Used for weighted farthest-document centroid
    // sampling.

    let mut cumulative_sims = vec![1.0_f32; documents.len()];

    // Per-document best cluster tracked while clusters are being formed -
    // replaces a separate full scoring pass after clusterization.

    let mut best_similarity = vec![f32::NEG_INFINITY; documents.len()];
    let mut best_cluster = vec![0_usize; documents.len()];

    // Initial cluster population.

    let mut clusters_ids = Vec::with_capacity(clusters_num);
    let mut cluster = Vec::with_capacity(centroids_num);

    for _ in 0..centroids_num {
        let index = rng.next_u64() as usize % remaining.len();

        cluster.push(remaining[index]);

        remaining.swap_remove(index);
    }

    clusters_ids.push(cluster);

    // Sparse rank snapshots of every formed cluster - dense buffers are zeroed
    // after each round, so snapshots are the only preserved representation.

    let mut clusters_ranks = Vec::<Box<[(u16, f32)]>>::with_capacity(clusters_num);

    for i in 0..clusters_num {
        // Sum per-document token frequencies of the cluster subset.

        for &id in &clusters_ids[i] {
            let cluster_doc_len = lengths[id as usize] as f32;

            for &(token, count) in profiles[id as usize].iter() {
                acc[token as usize] += count as f32 / cluster_doc_len;
            }
        }

        // Transform frequencies into log-ranks.

        let cluster_len = clusters_ids[i].len() as f32;

        for (token, freq) in acc.iter_mut().enumerate() {
            if *freq != 0.0 {
                members.push(token as u16);

                ranks[token] = ((*freq / cluster_len).log2()) + neg_log_tf[token];
            }
        }

        clusters_ranks.push(members.iter()
            .map(|&token| (token, ranks[token as usize]))
            .collect());

        // Score every document against the current cluster ranks.

        let scores = profiles.par_iter()
            .map(|profile| {
                profile.iter()
                    .map(|&(token, count)| ranks[token as usize] * count as f32)
                    .sum::<f32>()
            })
            .collect::<Box<[f32]>>();

        // Track best cluster assignments of all the documents.

        for (document, &similarity) in scores.iter().enumerate() {
            if similarity > best_similarity[document] {
                best_similarity[document] = similarity;
                best_cluster[document] = i;
            }
        }

        // Account the new cluster in cumulative similarities of the still
        // unsampled documents.

        if i + 1 < clusters_num {
            for &id in &remaining {
                cumulative_sims[id as usize] *= scores[id as usize];
            }
        }

        // Reset dense buffers before forming the next cluster.

        for &token in &members {
            acc[token as usize] = 0.0;
            ranks[token as usize] = 0.0;
        }

        members.clear();

        if i + 1 == clusters_num {
            break;
        }

        // Weight of every remaining document is an inverse geometric mean of
        // its cumulative similarity products. Rare (dissimilar) documents get
        // higher weights and thus are more likely to be chosen as centroids.

        let weights = remaining.par_iter()
            .map(|&id| 1.0 / cumulative_sims[id as usize].powf(1.0 / (i + 1) as f32))
            .collect::<Vec<f32>>();

        let mut total_weight = weights.par_iter()
            .sum::<f32>();

        // Sample `centroids_num` documents proportionally to their weights.

        let mut sampled = HashSet::<u32>::with_capacity(centroids_num);

        for _ in 0..centroids_num {
            if total_weight <= 0.0 {
                break;
            }

            let cutoff = rng.next_u32() as f32 / u32::MAX as f32 * total_weight;

            let mut curr_weight = 0.0;
            let mut chosen = None;

            for (weight_index, &id) in remaining.iter().enumerate() {
                if sampled.contains(&id) {
                    continue;
                }

                curr_weight += weights[weight_index];

                if cutoff <= curr_weight {
                    chosen = Some(weight_index);

                    break;
                }
            }

            let Some(weight_index) = chosen else {
                break;
            };

            let id = remaining[weight_index];

            total_weight -= weights[weight_index];

            sampled.insert(id);
        }

        // If it happened that we lack some documents in the cluster - fill it
        // with the least similar ones (documents with the highest weight).
        // This normally will never happen but presented just in case.

        while sampled.len() < centroids_num && sampled.len() < remaining.len() {
            let Some((_, &id)) = remaining.iter()
                .enumerate()
                .filter(|(_, id)| !sampled.contains(*id))
                .max_by(|a, b| {
                    weights[a.0]
                        .partial_cmp(&weights[b.0])
                        .unwrap_or(Ordering::Equal)
                })
            else {
                break;
            };

            sampled.insert(id);
        }

        // Remove all the taken documents from the pool.

        remaining.retain(|id| !sampled.contains(id));

        clusters_ids.push(sampled.into_iter().collect());
    }

    // Prepare clusters output.

    let clusters = clusters_ranks.into_iter()
        .map(|snapshot| Cluster {
            ranks: HashMap::from_iter(snapshot)
        })
        .collect::<Vec<Cluster>>();

    Ok((
        clusters.into_boxed_slice(),
        best_cluster.into_boxed_slice()
    ))
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use rand_chacha::rand_core::SeedableRng;

    use super::clusterize;

    fn make_document(token: u16, len: usize) -> Box<[u16]> {
        vec![token; len].into_boxed_slice()
    }

    #[test]
    fn clusters_are_semantically_separated() {
        let documents = vec![
            make_document(1, 40),
            make_document(1, 50),
            make_document(2, 45),
            make_document(2, 55),
            make_document(3, 42),
            make_document(3, 48),
        ];

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(42);

        let (clusters, assignments) =
            clusterize(3, 2, &documents, &mut rng).expect("should clusterize");

        assert_eq!(clusters.len(), 3);
        assert_eq!(assignments.len(), documents.len());

        // Documents sharing their whole vocabulary must land together.
        for i in 0..documents.len() {
            for j in 0..documents.len() {
                if documents[i][0] == documents[j][0] {
                    assert_eq!(
                        assignments[i],
                        assignments[j],
                        "docs with token {} must share a cluster",
                        documents[i][0]
                    );
                }
            }
        }

        for &cluster_index in assignments.iter() {
            assert!(cluster_index < clusters.len());
        }

        // Three separated vocabularies must not collapse into a single
        // assignment value.
        assert!(assignments.iter().collect::<HashSet<_>>().len() >= 2);
    }

    #[test]
    fn empty_input_bails() {
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(42);

        let result = clusterize(4, 2, Vec::new(), &mut rng);

        assert!(result.is_err());
    }
}
