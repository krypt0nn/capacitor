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

#[derive(Debug, Clone)]
pub struct Cluster<T> {
    ranks: HashMap<T, f32>
}

impl<T: PartialEq + Eq + std::hash::Hash> Cluster<T> {
    #[inline]
    pub fn new(ranks: impl IntoIterator<Item = (T, f32)>) -> Self {
        Self {
            ranks: HashMap::from_iter(ranks)
        }
    }

    #[inline]
    pub fn into_inner(self) -> HashMap<T, f32> {
        self.ranks
    }

    /// Calculate similarity between the current cluster to the given document.
    pub fn similarity(&self, document: impl IntoIterator<Item = T>) -> f32 {
        let mut similarity = 0.0;

        for token in document {
            if let Some(rank) = self.ranks.get(&token) {
                similarity += *rank;
            }
        }

        similarity
    }
}

impl<T> From<HashMap<T, f32>> for Cluster<T> {
    #[inline(always)]
    fn from(ranks: HashMap<T, f32>) -> Self {
        Self { ranks }
    }
}

#[allow(clippy::needless_range_loop)]
pub fn clusterize<T: Clone + PartialEq + Eq + std::hash::Hash + Send + Sync>(
    mut clusters_num: usize,
    mut centroids_num: usize,
    documents: impl AsRef<[Box<[T]>]>,
    rng: &mut impl Rng
) -> anyhow::Result<Box<[Cluster<T>]>> {
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

    // Calculate appearance frequencies for each token within all the documents.

    let mut documents_frequencies = HashMap::<&[T], HashMap<&T, f32>>::with_capacity(documents.len());
    let mut total_frequencies = HashMap::<&T, f32>::new();
    let mut total_appearances = HashMap::<&T, usize>::new();
    let mut total_tokens = 0;

    for document in documents {
        let mut document_appearances = HashMap::<&T, usize>::new();

        for token in document {
            *document_appearances.entry(token)
                .or_default() += 1;
        }

        let mut frequencies = HashMap::<&T, f32>::new();
        let document_len = document.len();

        for (token, count) in document_appearances {
            frequencies.insert(token, count as f32 / document_len as f32);

            *total_appearances.entry(token)
                .or_default() += 1;
        }

        total_tokens += document_len;

        documents_frequencies.insert(document.as_ref(), frequencies);
    }

    for (token, count) in total_appearances.drain() {
        total_frequencies.insert(token, count as f32 / total_tokens as f32);
    }

    // Calculate similarity between the given document and the given ranks.

    fn calc_similarity<T: PartialEq + Eq + std::hash::Hash>(
        ranks: &HashMap<&T, f32>,
        document: &[T]
    ) -> f32 {
        let mut similarity = 0.0;

        for token in document {
            if let Some(rank) = ranks.get(&token) {
                similarity += *rank;
            }
        }

        similarity
    }

    // Cumulative products of similarities to the clusters formed so far -
    // one entry per document. Used for weighted farthest-document centroid
    // sampling.

    let mut cumulative_sims = vec![1.0_f32; documents.len()];

    // Multiply similarity of every remaining document to the given ranks
    // map into the cumulative products.

    fn multiply_similarities<T: PartialEq + Eq + std::hash::Hash + Send + Sync>(
        cumulative_sims: &mut [f32],
        remaining: &[u32],
        ranks: &HashMap<&T, f32>,
        documents: &[Box<[T]>]
    ) {
        let sims = remaining.par_iter()
            .map(|&id| calc_similarity(ranks, &documents[id as usize]))
            .collect::<Vec<f32>>();

        for (&id, sim) in remaining.iter().zip(sims) {
            cumulative_sims[id as usize] *= sim;
        }
    }

    // Initial cluster population.

    let mut clusters = Vec::with_capacity(clusters_num);
    let mut cluster = Vec::with_capacity(centroids_num);

    for _ in 0..centroids_num {
        let index = rng.next_u64() as usize % remaining.len();

        cluster.push(documents[remaining[index] as usize].as_ref());

        remaining.swap_remove(index);
    }

    clusters.push(cluster);

    // Calculate tokens ranks in the initial cluster.

    fn calc_tokens_ranks<'tokens, T: PartialEq + Eq + std::hash::Hash>(
        documents: &[&'tokens [T]],
        total_frequencies: &HashMap<&T, f32>,
        documents_frequencies: &HashMap<&[T], HashMap<&T, f32>>
    ) -> HashMap<&'tokens T, f32> {
        let mut subset_frequencies = HashMap::<&T, f32>::new();
        let mut ranks = HashMap::new();

        let documents_len = documents.len();

        // Calculate frequencies for tokens within the provided documents.
        // These are (generally) different from the total documents frequencies.

        for document in documents {
            for token in document.iter() {
                *subset_frequencies.entry(token).or_default() += documents_frequencies.get(document)
                    .and_then(|freq| freq.get(&token))
                    .copied()
                    .unwrap_or_default() / documents_len as f32;
            }
        }

        // Calculate ranks for all the tokens in the provided documents.

        for (token, df) in subset_frequencies.drain() {
            let tf = total_frequencies.get(&token)
                .copied()
                .unwrap_or(f32::EPSILON);

            ranks.insert(token, df.log2() - tf.log2());
        }

        ranks
    }

    let mut ranks = Vec::with_capacity(clusters_num);

    ranks.push(calc_tokens_ranks(
        &clusters[0],
        &total_frequencies,
        &documents_frequencies
    ));

    // Account the initial cluster in cumulative similarities.

    if clusters_num > 1 {
        multiply_similarities(
            &mut cumulative_sims,
            &remaining,
            &ranks[0],
            documents
        );
    }

    // Populate other clusters.

    for i in 1..clusters_num {
        // Weight of every remaining document is an inverse geometric mean of
        // its cumulative similarity products. Rare (dissimilar) documents get
        // higher weights and thus are more likely to be chosen as centroids.

        let weights = remaining.par_iter()
            .map(|&id| 1.0 / cumulative_sims[id as usize].powf(1.0 / i as f32))
            .collect::<Vec<f32>>();

        let mut total_weight = weights.par_iter()
            .sum::<f32>();

        // Sample `centroids_num` documents proportionally to their weights.

        let mut cluster = HashSet::<u32>::with_capacity(centroids_num);

        for _ in 0..centroids_num {
            if total_weight <= 0.0 {
                break;
            }

            let cutoff = rng.next_u32() as f32 / u32::MAX as f32 * total_weight;

            let mut curr_weight = 0.0;
            let mut chosen = None;

            for (weight_index, &id) in remaining.iter().enumerate() {
                if cluster.contains(&id) {
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

            cluster.insert(id);
        }

        // If it happened that we lack some documents in the cluster - fill it
        // with the least similar ones (documents with the highest weight).
        // This normally will never happen but presented just in case.

        while cluster.len() < centroids_num && cluster.len() < remaining.len() {
            let Some((_, &id)) = remaining.iter()
                .enumerate()
                .filter(|(_, id)| !cluster.contains(*id))
                .max_by(|a, b| {
                    weights[a.0]
                        .partial_cmp(&weights[b.0])
                        .unwrap_or(Ordering::Equal)
                })
            else {
                break;
            };

            cluster.insert(id);
        }

        // Remove all the taken documents from the pool.

        remaining.retain(|id| !cluster.contains(id));

        clusters.push(cluster.into_iter()
            .map(|index| documents[index as usize].as_ref())
            .collect::<Vec<&[T]>>());

        // Calculate tokens ranks in the newly formed cluster.

        ranks.push(calc_tokens_ranks(
            &clusters[i],
            &total_frequencies,
            &documents_frequencies
        ));

        // Account the new cluster in cumulative similarities for all the
        // future rounds.

        if i + 1 < clusters_num {
            multiply_similarities(
                &mut cumulative_sims,
                &remaining,
                &ranks[i],
                documents
            );
        }
    }

    // Prepare clusters output.

    let mut clusters = Vec::with_capacity(clusters_num);

    for cluster in ranks {
        clusters.push(Cluster {
            ranks: cluster.into_iter()
                .map(|(k, v)| (k.clone(), v))
                .collect::<HashMap<T, f32>>()
        });
    }

    Ok(clusters.into_boxed_slice())
}
