use std::collections::{HashMap, HashSet};
use std::cmp::Ordering;

use rand_chacha::rand_core::RngCore;

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
pub fn clusterize<T: Clone + PartialEq + Eq + std::hash::Hash>(
    mut clusters_num: usize,
    mut centroids_num: usize,
    documents: impl AsRef<[Box<[T]>]>,
    rand: &mut impl RngCore
) -> anyhow::Result<Box<[Cluster<T>]>> {
    let documents = documents.as_ref();

    if clusters_num == 0 {
        clusters_num = documents.len().isqrt().max(1);
    }

    if centroids_num == 0 {
        centroids_num = clusters_num.isqrt().max(1);
    }

    if clusters_num * centroids_num > documents.len() {
        anyhow::bail!("clusters_num * centroids_num must be lower or equal to the documents amount");
    }

    // Prepare vector of references to the whole documents set.
    // This is needed to be able to remove taken documents from the set.
    //
    // Additionally, calculate appearance frequencies for each token within
    // all the documents.

    let mut documents_set = Vec::with_capacity(documents.len());
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
        documents_set.push(document.as_ref());
    }

    for (token, count) in total_appearances.drain() {
        total_frequencies.insert(token, count as f32 / total_tokens as f32);
    }

    // Initial cluster population.

    let mut clusters = Vec::with_capacity(clusters_num);
    let mut cluster = Vec::with_capacity(centroids_num);

    for _ in 0..centroids_num {
        let document = documents_set.swap_remove(rand.next_u64() as usize % documents_set.len());

        cluster.push(document);
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

    // Calculate similarities from the initial cluster to all the other documents.

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

    let mut clusters_similarities = Vec::with_capacity(clusters_num);
    let mut similarities = HashMap::<&[T], f32>::with_capacity(documents.len());

    for document in &clusters[0] {
        similarities.insert(document, calc_similarity(&ranks[0], document));
    }

    clusters_similarities.push(similarities);

    // Populate other clusters.

    for i in 1..clusters_num {
        // Calculate cummulative similarity from all the existing clusters to all
        // the documents as geometric mean of all the known similarities.

        let mut total_similarity = 0.0;
        let mut cummulative_similarities = Vec::with_capacity(documents.len());

        for document in &documents_set {
            let mut cummulative_similarity = 1.0;

            for j in 0..i {
                cummulative_similarity *= clusters_similarities[j].get(document)
                    .copied()
                    .unwrap_or(1.0);
            }

            cummulative_similarity = 1.0 / cummulative_similarity.powf(1.0 / i as f32);

            total_similarity += cummulative_similarity;

            cummulative_similarities.push((document, cummulative_similarity));
        }

        // Sort similarities in descending order. We will generate
        // `centroids_num` random floats from 0.0 to `total_distance` and
        // iterating over the sorted `cummulative_distances` vector choose
        // farthest documents.

        cummulative_similarities.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
        });

        let mut cluster = HashSet::<&[T]>::with_capacity(centroids_num);

        for _ in 0..centroids_num {
            let mut curr_distance = 0.0;
            let cutoff = rand.next_u32() as f32 / u32::MAX as f32 * total_similarity;

            for (document, similarity) in &cummulative_similarities {
                if cluster.contains(*document) {
                    continue;
                }

                curr_distance += similarity;

                if cutoff <= curr_distance {
                    total_similarity -= similarity;

                    cluster.insert(document);

                    break;
                }
            }
        }

        // If it happened that we lack some documents in the cluster - manually
        // fill them from the farthest list. This normally will never happen
        // but presented just in case.

        let mut k = 0;
        let n = cummulative_similarities.len();

        // i < n just in case but it shouldn't be needed here.
        while cluster.len() < centroids_num && k < n {
            cluster.insert(cummulative_similarities[k].0);

            k += 1;
        }

        // Remove all the taken documents from the documents set.
        for document in &cluster {
            for i in 0..documents_set.len() {
                if &documents_set[i] == document {
                    documents_set.swap_remove(i);

                    break;
                }
            }
        }

        clusters.push(cluster.into_iter().collect());

        // Calculate tokens ranks in the newly formed cluster.

        ranks.push(calc_tokens_ranks(
            &clusters[i],
            &total_frequencies,
            &documents_frequencies
        ));

        // Calculate similarities for newly formed cluster.

        let mut similarities = HashMap::<&[T], f32>::with_capacity(documents.len());

        for document in &clusters[i] {
            similarities.insert(document, calc_similarity(&ranks[i], document));
        }

        clusters_similarities.push(similarities);
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
