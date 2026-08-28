// SPDX-License-Identifier: GPL-3.0-or-later
//
// capacitor
// Copyright (C) 2026  Nikita Podvirnyi <krypt0nn@vk.ru>
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

use compact_str::{CompactString, ToCompactString};
use rand_chacha::rand_core::Rng;
use rayon::prelude::*;

use crate::tokens::{pre_tokenize, TokensMap};
use crate::transitions::TransitionsMap;
use crate::clustering::clusterize;
use crate::recipe::Recipe;

use super::{Model, Expert};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Progress {
    /// Read dataset documents.
    ReadFiles {
        /// Processed files.
        current: usize,

        /// Total files.
        total: usize
    },

    /// Pre-tokenize dataset documents.
    PreTokenize {
        /// Processed bytes.
        current: u64,

        /// Total bytes.
        total: u64
    },

    /// Fit BPE tokenizer on pre-tokenized dataset documents.
    FitTokenizer {
        /// Learned tokens.
        current: u16,

        /// Total tokens to learn.
        total: u16
    },

    /// Build binary searchable tokens map for trained BPE tokenizer.
    BuildTokensMap,

    /// Build shared transitions table.
    BuildSharedTransitions,

    /// Clusterize datasets into experts count.
    ClusterizeDatasets,

    /// Build model experts.
    BuildExperts {
        /// Built experts.
        current: usize,

        /// Total experts.
        total: usize
    },

    /// Finish model building.
    Done
}

pub fn build(
    mut recipe: Recipe,
    rng: &mut impl Rng,
    progress: impl Fn(Progress) + Send + Sync
) -> anyhow::Result<Model> {
    // Read documents from the dataset files.

    let mut documents = Vec::with_capacity(recipe.files.len());

    progress(Progress::ReadFiles {
        current: 0,
        total: recipe.files.len()
    });

    for (i, file) in recipe.files.iter().enumerate() {
        // Read documents from the dataset file.
        let dataset = std::fs::read_to_string(&file.path)?;

        let mut dataset_documents = dataset.split(&file.delimiter)
            .map(|document| {
                format!(
                    "{}{document}{}",
                    Model::START_TOKEN,
                    Model::STOP_TOKEN
                )
            })
            .collect::<Vec<String>>();

        // Shuffle documents.
        if file.shuffle {
            let n = dataset_documents.len();

            for _ in 0..n {
                let document = dataset_documents.swap_remove(
                    (rng.next_u64() % n as u64) as usize
                );

                dataset_documents.push(document);
            }
        }

        documents.extend(dataset_documents);

        progress(Progress::ReadFiles {
            current: i + 1,
            total: recipe.files.len()
        });
    }

    // Bail out before any tokenization work when the requested experts layout
    // can't possibly fit into the corpus. The clusterize() check against its
    // filtered document pool happens later and is stricter.
    if recipe.experts.num_total > 0 {
        let centroids_num = if recipe.experts.num_centroids == 0 {
            recipe.experts.num_total.isqrt().max(1)
        } else {
            recipe.experts.num_centroids
        };

        if recipe.experts.num_total * centroids_num > documents.len() {
            anyhow::bail!("clusters_num * centroids_num must be lower or equal to the documents amount");
        }
    }

    // Prefill tokenizer with some standard tokens.

    fn internal_word(
        word: &str,
        alphabet: &mut HashMap<CompactString, u16>,
        vocab: &mut Vec<CompactString>
    ) -> u16 {
        if let Some(&id) = alphabet.get(word) {
            return id;
        }

        let id = vocab.len() as u16;

        alphabet.insert(word.to_compact_string(), id);
        vocab.push(word.to_compact_string());

        id
    }

    fn document_pairs(
        document: &[u16]
    ) -> HashMap<(u16, u16), u64> {
        let mut counts = HashMap::with_capacity(document.len());

        for pair in document.windows(2) {
            let value: &mut u64 = counts.entry((pair[0], pair[1]))
                .or_default();

            *value = value.saturating_add(1);
        }

        counts
    }

    fn replace_pairs(
        document: &mut Vec<u16>,
        id_1: u16,
        id_2: u16,
        new_id: u16
    ) {
        let mut i = 0;
        let mut j = 0;

        while j < document.len() {
            if j + 1 < document.len()
                && document[j] == id_1
                && document[j + 1] == id_2
            {
                document[i] = new_id;

                j += 2;
            } else {
                document[i] = document[j];

                j += 1;
            }

            i += 1;
        }

        document.truncate(i);
    }

    let mut alphabet = HashMap::new();
    let mut vocab = Vec::new();

    // Special tags must never be merged with other tokens.
    let mut special_tags = HashSet::new();

    special_tags.insert(internal_word(Model::START_TOKEN, &mut alphabet, &mut vocab));
    special_tags.insert(internal_word(Model::STOP_TOKEN, &mut alphabet, &mut vocab));

    // Pre-fill latin letters, digits and special characters.
    for c in 32..127_u8 {
        internal_word(&(c as char).to_string(), &mut alphabet, &mut vocab);
    }

    // Pre-tokenize input documents.
    let total = documents.iter()
        .map(|document| document.len() as u64)
        .sum::<u64>();

    let mut current = 0;

    let mut pre_tokenized_documents = Vec::with_capacity(documents.len());

    progress(Progress::PreTokenize { current, total });

    for document in &documents {
        current += document.len() as u64;

        let pre_tokenized_document = pre_tokenize(
            document,
            recipe.tokenizer.make_lowercase,
            recipe.tokenizer.force_alphanumeric
        );

        pre_tokenized_documents.push(
            pre_tokenized_document.into_iter()
                .map(|token| {
                    let id = internal_word(
                        &token,
                        &mut alphabet,
                        &mut vocab
                    );

                    if token.starts_with("<|") && token.ends_with("|>") {
                        special_tags.insert(id);
                    }

                    id
                })
                .collect::<Vec<u16>>()
        );

        progress(Progress::PreTokenize { current, total });
    }

    // The very last document may be overwritten in the progress view by the
    // next stage before it gets rendered - close the bar explicitly.
    progress(Progress::PreTokenize { current: total, total });

    // Train tokens model.
    //
    // Take small part of the pre-tokenized documents to speed-up BPE
    // model building. Training merges are applied to a copy of the
    // sample so that the full corpus keeps its pre-tokenization intact
    // for the final tokenization and model stages below.

    if vocab.len() > u16::MAX as usize {
        anyhow::bail!("dataset contains more than 65535 unique BPE tokens and cannot be used");
    }

    let mut taken_samples = 0;

    let mut training_documents = pre_tokenized_documents.iter()
        .filter(|document| {
            if taken_samples > recipe.tokenizer.num_samples {
                return false;
            }

            taken_samples += document.len();

            true
        })
        .cloned()
        .collect::<Vec<Vec<u16>>>();

    progress(Progress::FitTokenizer {
        current: vocab.len() as u16,
        total: recipe.tokenizer.num_tokens
    });

    // Count symbol pairs and index documents containing each pair.
    let mut pair_frequencies = HashMap::<(u16, u16), u64>::new();
    let mut pair_documents = HashMap::<(u16, u16), Vec<u32>>::new();
    let mut seen_documents = HashSet::new();

    for (document_index, document) in training_documents.iter().enumerate() {
        for (pair, count) in document_pairs(document.as_ref()) {
            // Never let special tags merge with other tokens.
            if special_tags.contains(&pair.0) || special_tags.contains(&pair.1) {
                continue;
            }

            let value = pair_frequencies.entry(pair)
                .or_default();

            *value = value.saturating_add(count);

            pair_documents.entry(pair)
                .or_default()
                .push(document_index as u32);
        }
    }

    while (vocab.len() as u16) < recipe.tokenizer.num_tokens {
        // Take the most frequent pair. Merges which would produce a token
        // ending with whitespace are rejected: whitespace must lead the
        // following token instead (` there`), so bare word tokens keep the
        // word boundary statistics that inference prompts end on.

        let Some((&best_pair, _)) = pair_frequencies.par_iter()
            .filter(|(pair, _)| {
                !vocab[pair.1 as usize].ends_with(char::is_whitespace)
            })
            .max_by_key(|(_, frequency)| **frequency)
        else {
            // Stop learning if no more token pairs available.
            break;
        };

        let mut new_token = vocab[best_pair.0 as usize].clone();

        new_token.push_str(&vocab[best_pair.1 as usize]);

        // Merging may produce an already known word.
        if alphabet.contains_key(new_token.as_str()) {
            pair_frequencies.remove(&best_pair);

            continue;
        }

        let new_id = vocab.len() as u16;

        alphabet.insert(new_token.clone(), new_id);
        vocab.push(new_token);

        // Rewrite documents which contain the merged pair, updating pair
        // statistics incrementally.
        let affected_document = pair_documents.remove(&best_pair)
            .unwrap_or_default();

        seen_documents.clear();

        for document_index in affected_document {
            if !seen_documents.insert(document_index) {
                continue;
            }

            let document = &mut training_documents[document_index as usize];

            // Remove old pairs of the document from global statistics.

            for (pair, count) in document_pairs(document.as_ref()) {
                if let Some(frequency) = pair_frequencies.get_mut(&pair) {
                    *frequency -= count.min(*frequency);

                    if *frequency == 0 {
                        pair_frequencies.remove(&pair);
                        pair_documents.remove(&pair);
                    }
                }
            }

            // Merge the pairs in place.

            replace_pairs(document, best_pair.0, best_pair.1, new_id);

            if document.is_empty() {
                continue;
            }

            // Add new pairs of the document back to global statistics.

            for (pair, count) in document_pairs(document) {
                // Never let special tags merge with other tokens.
                if special_tags.contains(&pair.0) || special_tags.contains(&pair.1) {
                    continue;
                }

                let value = pair_frequencies.entry(pair)
                    .or_default();

                *value = value.saturating_add(count);

                pair_documents.entry(pair)
                    .or_default()
                    .push(document_index);
            }
        }

        progress(Progress::FitTokenizer {
            current: vocab.len() as u16,
            total: recipe.tokenizer.num_tokens
        });
    }

    // The corpus may run out of pairs before the requested token amount is
    // learned - close the bar explicitly.
    progress(Progress::FitTokenizer {
        current: recipe.tokenizer.num_tokens,
        total: recipe.tokenizer.num_tokens
    });

    // Build tokens map.

    progress(Progress::BuildTokensMap);

    let tokens_map = TokensMap::from_words(vocab.iter())?;

    let words_table = tokens_map.as_words_table();

    // Alias for every vocab word: token id of its bare last word. From-grams
    // are keyed on these so a word's statistics do not depend on how it
    // happened to be tokenized (` chat`, `hi chat` -> `chat`).
    let word_aliases = super::word_aliases(&tokens_map);

    // Tokenize documents.
    //
    // Find already learned multi-character tokens in every document of
    // the full corpus - including ones which were not used for tokenizer
    // training.

    let max_token_len = tokens_map.max_token_len();

    let documents = pre_tokenized_documents.into_par_iter()
        .map(|document| {
            let n = document.len();

            let mut tokenized_document = Vec::with_capacity(n);

            // Reused buffer for candidate token strings.
            let mut token = String::new();

            let mut i = 0;

            while i < n {
                let mut j = (i + max_token_len).min(n);

                while j > i {
                    token.clear();

                    for id in &document[i..j] {
                        token.push_str(&vocab[*id as usize]);
                    }

                    if let Some(&t) = words_table.get(token.as_bytes()) {
                        tokenized_document.push(t);

                        break;
                    }

                    j -= 1;
                }

                if i == j {
                    // Should be impossible to hit because we've made
                    // one-character tokens on pre-tokenization stage.
                    i += 1;
                } else {
                    i = j;
                }
            }

            tokenized_document.into_boxed_slice()
        })
        .collect::<Box<[Box<[u16]>]>>();

    // Count transitions for every document.
    //
    // From-grams are keyed on the alias-normalized copy of each document, so
    // every from token is represented by the bare last word it contains.
    // To-grams keep their original tokens.

    progress(Progress::BuildSharedTransitions);

    let normalized_documents = documents.par_iter()
        .map(|document| {
            document.iter()
                .map(|&token| word_aliases.get(&token).copied().unwrap_or(token))
                .collect::<Box<[u16]>>()
        })
        .collect::<Box<[Box<[u16]>]>>();

    let min_len = recipe.ngrams.num_from + recipe.ngrams.num_to;

    let transitions = documents.par_iter()
        .zip(normalized_documents.par_iter())
        .filter_map(|(document, normalized)| {
            if document.len() < min_len {
                return None;
            }

            let mut document_transitions = HashMap::<(&[u16], &[u16]), usize>::new();

            let doc_len = document.len() - min_len;
            let mut i = 0;

            while i < doc_len {
                let transition = (
                    &normalized[i..i + recipe.ngrams.num_from],
                    &document[i + recipe.ngrams.num_from..i + min_len]
                );

                *document_transitions.entry(transition)
                    .or_default() += 1;

                i += 1;
            }

            Some((document.as_ref(), document_transitions))
        })
        .collect::<HashMap<&[u16], HashMap<(&[u16], &[u16]), usize>>>();

    // Create transitions map for the whole dataset.

    let mut cummulative_transitions = HashMap::<&(&[u16], &[u16]), usize>::new();

    for document_transitions in transitions.values() {
        for (transition, count) in document_transitions.iter() {
            *cummulative_transitions.entry(transition)
                .or_default() += *count;
        }
    }

    let total_transitions = cummulative_transitions.values()
        .copied()
        .sum::<usize>();

    let cummulative_transitions = cummulative_transitions.into_par_iter()
        .map(|(transition, count)| {
            let frequency = count as f32 / total_transitions as f32;

            (transition.0, transition.1, (frequency * u32::MAX as f32) as u32)
        })
        .collect::<Vec<_>>();

    let transitions_map = TransitionsMap::from_transitions(
        cummulative_transitions
    )?;

    // Build model experts if needed.
    let mut experts = Vec::with_capacity(recipe.experts.num_total);

    if recipe.experts.num_total > 0 {
        // Clusterize documents.

        progress(Progress::ClusterizeDatasets);

        let (clusters, document_assignment) = clusterize(
            recipe.experts.num_total,
            recipe.experts.num_centroids,
            &documents,
            rng
        )?;

        // Documents were already assigned to their most similar cluster
        // during clusterization.

        let mut documents_clusters = vec![Vec::new(); clusters.len()];

        for (document, document_cluster) in documents.iter()
            .zip(document_assignment.into_vec())
        {
            documents_clusters[document_cluster].push(document);
        }

        // Create experts from clusters in parallel, preserving their order.

        let n = clusters.len();

        progress(Progress::BuildExperts {
            current: 0,
            total: n
        });

        let built_experts = std::sync::atomic::AtomicUsize::new(0);

        experts = clusters.into_par_iter()
            .enumerate()
            .map(|(i, cluster)| {
                let mut cluster_transitions = HashMap::<&(&[u16], &[u16]), usize>::new();

                for document in &documents_clusters[i] {
                    if let Some(document_transitions) = transitions.get(document.as_ref()) {
                        for (transition, count) in document_transitions.iter() {
                            *cluster_transitions.entry(transition)
                                .or_default() += *count;
                        }
                    }
                }

                let total_transitions = cluster_transitions.values()
                    .copied()
                    .sum::<usize>();

                // A cluster can contain only documents which are too short
                // to produce any transition.
                if total_transitions == 0 {
                    return Ok(None);
                }

                let cluster_transitions = cluster_transitions.into_par_iter()
                    .map(|(transition, count)| {
                        let frequency = count as f32 / total_transitions as f32;

                        (transition.0, transition.1, (frequency * u32::MAX as f32) as u32)
                    })
                    .collect::<Vec<_>>();

                let transitions = TransitionsMap::from_transitions(
                    cluster_transitions
                )?;

                let current = built_experts.fetch_add(
                    1,
                    std::sync::atomic::Ordering::Relaxed
                );

                progress(Progress::BuildExperts {
                    current: current + 1,
                    total: n
                });

                Ok(Some(Expert {
                    cluster,
                    transitions
                }))
            })
            .collect::<anyhow::Result<Vec<Option<Expert>>>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<Expert>>();

        // Empty clusters are skipped without reporting progress - close the
        // bar explicitly.
        progress(Progress::BuildExperts { current: n, total: n });
    }

    // Prefill default metadata keys.

    // Model tokenizer.
    recipe.keys.entry(String::from("model.tokenizer.make_lowercase"))
        .or_insert({
            if recipe.tokenizer.make_lowercase {
                String::from("true")
            } else {
                String::from("false")
            }
        });

    recipe.keys.entry(String::from("model.tokenizer.force_alphanumeric"))
        .or_insert({
            if recipe.tokenizer.force_alphanumeric {
                String::from("true")
            } else {
                String::from("false")
            }
        });

    recipe.keys.entry(String::from("model.tokenizer.num_tokens"))
        .or_insert(recipe.tokenizer.num_tokens.to_string());

    // Model tokens (ngrams).
    recipe.keys.entry(String::from("model.tokens.from_depth"))
        .or_insert(recipe.ngrams.num_from.to_string());

    recipe.keys.entry(String::from("model.tokens.to_depth"))
        .or_insert(recipe.ngrams.num_to.to_string());

    recipe.keys.entry(String::from("model.tokens.start_token"))
        .or_insert(Model::START_TOKEN.to_string());

    recipe.keys.entry(String::from("model.tokens.stop_token"))
        .or_insert(Model::STOP_TOKEN.to_string());

    // Model experts.
    recipe.keys.entry(String::from("model.experts.active"))
        .or_insert(recipe.experts.num_active.to_string());

    recipe.keys.entry(String::from("model.experts.total"))
        .or_insert(experts.len().to_string());

    recipe.keys.entry(String::from("model.experts.centroids"))
        .or_insert(recipe.experts.num_centroids.to_string());

    // Model inference params.
    recipe.keys.entry(String::from("model.inference.template"))
        .or_insert(recipe.template);

    for (i, token) in recipe.stop_tokens.into_iter().enumerate() {
        recipe.keys.entry(format!("model.inference.stop_tokens[{i}]"))
            .or_insert(token);
    }

    recipe.keys.entry(String::from("model.inference.top_k"))
        .or_insert(Model::DEFAULT_TOP_K.to_string());

    recipe.keys.entry(String::from("model.inference.temperature"))
        .or_insert(Model::DEFAULT_TEMPERATURE.to_string());

    recipe.keys.entry(String::from("model.inference.max_tokens"))
        .or_insert(Model::DEFAULT_MAX_TOKENS.to_string());

    recipe.keys.entry(String::from("model.inference.active_experts"))
        .or_insert(recipe.experts.num_active.to_string());

    recipe.keys.entry(String::from("model.inference.experts_context"))
        .or_insert(Model::DEFAULT_EXPERTS_CONTEXT.to_string());

    // Build the model.

    progress(Progress::Done);

    Ok(Model {
        keys: recipe.keys,
        tokens: tokens_map,
        transitions: transitions_map,
        experts: experts.into_boxed_slice(),
        word_aliases
    })
}
