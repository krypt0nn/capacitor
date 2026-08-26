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

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::iter::FusedIterator;

use rand_chacha::rand_core::Rng;
use rayon::prelude::*;

use crate::tokens::TokensMap;
use crate::transitions::{Transition, TransitionsMap};
use crate::clustering::{Cluster, clusterize};
use crate::recipe::Recipe;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuildProgress {
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
        current: usize,

        /// Total tokens to learn.
        total: usize
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

#[derive(Debug, Clone)]
pub struct Expert {
    cluster: Cluster<u16>,
    transitions: TransitionsMap
}

impl Expert {
    #[inline(always)]
    pub const fn cluster(&self) -> &Cluster<u16> {
        &self.cluster
    }

    #[inline(always)]
    pub const fn transitions(&self) -> &TransitionsMap {
        &self.transitions
    }

    #[inline]
    pub fn similarity(&self, document: impl IntoIterator<Item = u16>) -> f32 {
        self.cluster.similarity(document)
    }

    #[inline]
    pub fn find_transitions(
        &self,
        from: impl AsRef<[u16]>
    ) -> HashSet<Transition> {
        self.transitions.find_transitions(from)
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    keys: HashMap<String, String>,
    tokens: TokensMap,
    transitions: TransitionsMap,
    experts: Box<[Expert]>
}

impl Model {
    pub const START_TOKEN: &str = "<|start|>";
    pub const STOP_TOKEN: &str = "<|stop|>";

    pub fn open(model: impl AsRef<[u8]>) -> anyhow::Result<Self> {
        let model = model.as_ref();
        let n = model.len();

        if n < 13 {
            anyhow::bail!("invalid model format: too short");
        }

        if &model[..9] != b"capacitor" {
            anyhow::bail!("invalid model format: not a capacitor model");
        }

        if &model[9..11] != b"v2" {
            anyhow::bail!("unsupported model format: v2 expected, got {}", String::from_utf8_lossy(&model[9..11]));
        }

        let keys_num = u16::from_le_bytes([model[11], model[12]]) as usize;

        // Read key-values table (metadata).

        let mut keys = HashMap::<String, String>::with_capacity(keys_num);

        let mut offset = 13;
        let mut i = 0;

        while i < keys_num {
            let key_len = model[offset] as usize;

            let value_len = u16::from_le_bytes([
                model[offset + 1],
                model[offset + 2]
            ]) as usize;

            offset += 3;

            let key = String::from_utf8_lossy(&model[offset..offset + key_len]);
            let value = String::from_utf8_lossy(&model[offset + key_len..offset + key_len + value_len]);

            offset += key_len + value_len;
            i += 1;

            keys.insert(key.to_string(), value.to_string());
        }

        // Read tokens map.

        let tokens_map_len = u64::from_le_bytes([
            model[offset    ], model[offset + 1], model[offset + 2], model[offset + 3],
            model[offset + 4], model[offset + 5], model[offset + 6], model[offset + 7]
        ]) as usize;

        offset += 8;

        let tokens_map = TokensMap::open(&model[offset..offset + tokens_map_len]);

        offset += tokens_map_len;

        // Read base model transitions map.

        let transitions_map_len = u64::from_le_bytes([
            model[offset    ], model[offset + 1], model[offset + 2], model[offset + 3],
            model[offset + 4], model[offset + 5], model[offset + 6], model[offset + 7]
        ]) as usize;

        offset += 8;

        let transitions_map = TransitionsMap::open(&model[offset..offset + transitions_map_len])?;

        offset += transitions_map_len;

        // Read experts.

        let total_experts = u32::from_le_bytes([
            model[offset    ], model[offset + 1],
            model[offset + 2], model[offset + 3]
        ]) as usize;

        offset += 4;
        i = 0;

        let mut experts = Vec::with_capacity(total_experts);

        while i < total_experts {
            let cluster_len = u32::from_le_bytes([
                model[offset    ], model[offset + 1],
                model[offset + 2], model[offset + 3]
            ]) as usize;

            let transitions_map_len = u64::from_le_bytes([
                model[offset + 4], model[offset + 5], model[offset +  6], model[offset +  7],
                model[offset + 8], model[offset + 9], model[offset + 10], model[offset + 11]
            ]) as usize;

            offset += 12;

            // Read expert cluster centroids.

            let mut cluster = HashMap::<u16, f32>::with_capacity(cluster_len);
            let mut frequency = [0; 4];
            let mut j = 0;

            while j < cluster_len {
                let token = u16::from_le_bytes([
                    model[offset], model[offset + 1]
                ]);

                frequency.copy_from_slice(&model[offset + 2..offset + 6]);

                cluster.insert(token, f32::from_le_bytes(frequency));

                offset += 6;
                j += 1;
            }

            // Read expert transitions matrix and store it.

            let transitions = TransitionsMap::open(
                &model[offset..offset + transitions_map_len]
            )?;

            offset += transitions_map_len;

            experts.push(Expert {
                cluster: Cluster::from(cluster),
                transitions
            });

            i += 1;
        }

        // Return the parsed model.

        Ok(Self {
            keys,
            tokens: tokens_map,
            transitions: transitions_map,
            experts: experts.into_boxed_slice()
        })
    }

    pub fn into_bytes(self) -> Box<[u8]> {
        // I technically can calculate exact container size needed to store
        // this model but who cares?
        let mut model = Vec::new();

        model.extend(b"capacitorv2");

        // Encode metadata keys.

        model.extend((self.keys.len() as u16).to_le_bytes());

        for (key, value) in self.keys {
            model.push(key.len() as u8);
            model.extend((value.len() as u16).to_le_bytes());
            model.extend(key.as_bytes());
            model.extend(value.as_bytes());
        }

        // Encode tokens map.

        let tokens = self.tokens.into_inner();

        model.extend((tokens.len() as u64).to_le_bytes());
        model.extend(tokens);

        // Encode base model transitions map.

        let transitions = self.transitions.into_inner();

        model.extend((transitions.len() as u64).to_le_bytes());
        model.extend(transitions);

        // Encode experts.

        model.extend(&(self.experts.len() as u32).to_le_bytes());

        for expert in self.experts {
            let cluster = expert.cluster.into_inner();
            let transitions = expert.transitions.into_inner();

            model.extend((cluster.len() as u32).to_le_bytes());
            model.extend((transitions.len() as u64).to_le_bytes());

            for (token, rank) in cluster {
                model.extend(token.to_le_bytes());
                model.extend(rank.to_le_bytes());
            }

            model.extend(transitions);
        }

        model.into_boxed_slice()
    }

    pub fn build(
        mut recipe: Recipe,
        rng: &mut impl Rng,
        progress: impl Fn(BuildProgress) + Send + Sync
    ) -> anyhow::Result<Self> {
        // Read documents from the dataset files.

        let mut documents = Vec::with_capacity(recipe.files.len());

        progress(BuildProgress::ReadFiles {
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
                        Self::START_TOKEN,
                        Self::STOP_TOKEN
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

            progress(BuildProgress::ReadFiles {
                current: i + 1,
                total: recipe.files.len()
            });
        }

        // Fit tokenizer on documents.

        // Prefill tokenizer with some standard tokens.

        fn internal_word(
            word: &str,
            alphabet: &mut HashMap<String, u32>,
            vocab: &mut Vec<String>
        ) -> u32 {
            if let Some(&id) = alphabet.get(word) {
                return id;
            }

            let id = vocab.len() as u32;

            alphabet.insert(word.to_string(), id);
            vocab.push(word.to_string());

            id
        }

        fn document_pairs(
            document: &[u32]
        ) -> HashMap<(u32, u32), u64> {
            let mut counts = HashMap::with_capacity(document.len());

            for pair in document.windows(2) {
                let value: &mut u64 = counts.entry((pair[0], pair[1]))
                    .or_default();

                *value = value.saturating_add(1);
            }

            counts
        }

        fn replace_pairs(
            document: &mut Vec<u32>,
            id_1: u32,
            id_2: u32,
            new_id: u32
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

        let mut alphabet = HashMap::<String, u32>::new();
        let mut vocab = Vec::<String>::new();

        // Special tags must never be merged with other tokens.
        let mut special_tags = HashSet::<u32>::new();

        special_tags.insert(internal_word(Self::START_TOKEN, &mut alphabet, &mut vocab));
        special_tags.insert(internal_word(Self::STOP_TOKEN, &mut alphabet, &mut vocab));

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

        progress(BuildProgress::PreTokenize { current, total });

        for document in &documents {
            current += document.len() as u64;

            let document = document.chars().collect::<Box<[char]>>();

            let n = document.len();
            let mut i = 0;

            let mut pre_tokenized_document = Vec::with_capacity(document.len());

            while i < n {
                // Preserve special tags as separate tokens.
                if document[i] == '<' && i + 1 < n && document[i + 1] == '|' {
                    let mut j = i + 2;
                    let mut found = false;

                    while j < n && (j - i) < 256 {
                        if document[j] == '>' && document[j - 1] == '|' {
                            found = true;

                            break;
                        }

                        j += 1;
                    }

                    // If we found <|token|>, then store it. Otherwise process
                    // < symbol as separate character.
                    if found {
                        let token = document[i..=j].iter().collect::<String>();

                        let id = internal_word(
                            &token,
                            &mut alphabet,
                            &mut vocab
                        );

                        special_tags.insert(id);
                        pre_tokenized_document.push(id);

                        i = j + 1;

                        continue;
                    }
                }

                let token = if recipe.tokenizer.make_lowercase {
                    document[i].to_lowercase().collect::<String>()
                } else {
                    document[i].to_string()
                };

                pre_tokenized_document.push(internal_word(
                    &token,
                    &mut alphabet,
                    &mut vocab
                ));

                i += 1;
            }

            pre_tokenized_documents.push(pre_tokenized_document);

            progress(BuildProgress::PreTokenize { current, total });
        }

        // Train tokens model.
        //
        // Take small part of the pre-tokenized documents to speed-up BPE
        // model building. Training merges are applied to a copy of the
        // sample so that the full corpus keeps its pre-tokenization intact
        // for the final tokenization and model stages below.

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
            .collect::<Vec<Vec<u32>>>();

        progress(BuildProgress::FitTokenizer {
            current: vocab.len(),
            total: recipe.tokenizer.num_tokens
        });

        // Count symbol pairs and index documents containing each pair.
        let mut pair_frequencies = HashMap::<(u32, u32), u64>::new();
        let mut pair_documents = HashMap::<(u32, u32), Vec<u32>>::new();
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

        while vocab.len() < recipe.tokenizer.num_tokens {
            // Take the most frequent pair.

            let Some((&best_pair, _)) = pair_frequencies.par_iter()
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

            let new_id = vocab.len() as u32;

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

            progress(BuildProgress::FitTokenizer {
                current: vocab.len(),
                total: recipe.tokenizer.num_tokens
            });
        }

        // Build tokens map.

        progress(BuildProgress::BuildTokensMap);

        let tokens_map = TokensMap::from_words(vocab.iter())?;

        let words_table = tokens_map.as_words_table();

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

        progress(BuildProgress::BuildSharedTransitions);

        let min_len = recipe.from_depth + recipe.to_depth;

        let transitions = documents.par_iter()
            .filter_map(|document| {
                if document.len() < min_len {
                    return None;
                }

                let mut document_transitions = HashMap::<(&[u16], &[u16]), usize>::new();

                let doc_len = document.len() - min_len;
                let mut i = 0;

                while i < doc_len {
                    let transition = (
                        &document[i..i + recipe.from_depth],
                        &document[i + recipe.from_depth..i + min_len]
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
        let mut experts = Vec::with_capacity(recipe.total_experts);

        if recipe.total_experts > 0 {
            // Clusterize documents.

            progress(BuildProgress::ClusterizeDatasets);

            let (clusters, document_assignment) = clusterize(
                recipe.total_experts,
                recipe.centroids,
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

            progress(BuildProgress::BuildExperts {
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

                    progress(BuildProgress::BuildExperts {
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
        }

        // Prefill default metadata keys.

        recipe.keys.entry(String::from("model.tokenizer.make_lowercase"))
            .or_insert({
                if recipe.tokenizer.make_lowercase {
                    String::from("true")
                } else {
                    String::from("false")
                }
            });

        recipe.keys.entry(String::from("model.tokenizer.num_tokens"))
            .or_insert(recipe.tokenizer.num_tokens.to_string());

        recipe.keys.entry(String::from("model.tokens.from_depth"))
            .or_insert(recipe.from_depth.to_string());

        recipe.keys.entry(String::from("model.tokens.to_depth"))
            .or_insert(recipe.to_depth.to_string());

        recipe.keys.entry(String::from("model.tokens.start_token"))
            .or_insert(Self::START_TOKEN.to_string());

        recipe.keys.entry(String::from("model.tokens.stop_token"))
            .or_insert(Self::STOP_TOKEN.to_string());

        recipe.keys.entry(String::from("model.experts.total"))
            .or_insert(experts.len().to_string());

        recipe.keys.entry(String::from("model.experts.active"))
            .or_insert(recipe.active_experts.to_string());

        recipe.keys.entry(String::from("model.experts.centroids"))
            .or_insert(recipe.centroids.to_string());

        recipe.keys.entry(String::from("model.inference.template"))
            .or_insert(recipe.template);

        for (i, token) in recipe.stop_tokens.into_iter().enumerate() {
            recipe.keys.entry(format!("model.inference.stop_tokens[{i}]"))
                .or_insert(token);
        }

        recipe.keys.entry(String::from("model.inference.top_k"))
            .or_insert(String::from("10"));

        recipe.keys.entry(String::from("model.inference.max_tokens"))
            .or_insert(String::from("1024"));

        // Build the model.

        progress(BuildProgress::Done);

        Ok(Self {
            keys: recipe.keys,
            tokens: tokens_map,
            transitions: transitions_map,
            experts: experts.into_boxed_slice()
        })
    }

    #[inline(always)]
    pub const fn keys_ref(&self) -> &HashMap<String, String> {
        &self.keys
    }

    #[inline(always)]
    pub const fn keys_mut(&mut self) -> &mut HashMap<String, String> {
        &mut self.keys
    }

    #[inline(always)]
    pub const fn tokens_ref(&self) -> &TokensMap {
        &self.tokens
    }

    #[inline(always)]
    pub const fn transitions_ref(&self) -> &TransitionsMap {
        &self.transitions
    }

    #[inline(always)]
    pub const fn experts_ref(&self) -> &[Expert] {
        &self.experts
    }

    /// Encode given text using model's tokenizer. Unknown tokens will be
    /// skipped entirely.
    pub fn tokenize(
        &self,
        text: impl AsRef<str>
    ) -> Box<[u16]> {
        // Look if we need to convert input text to lowercase.
        let make_lowercase = self.keys.get("model.tokenizer.make_lowercase")
            .map(|value| value.as_str())
            .unwrap_or("false") == "true";

        // Pre-tokenize given text.
        let text = text.as_ref();

        let text = text.chars()
            .map(|c| {
                if make_lowercase {
                    c.to_lowercase().to_string()
                } else {
                    c.to_string()
                }
            })
            .collect::<Box<[String]>>();

        // Tokenize the text.
        let n = text.len();
        let max_token_len = self.tokens.max_token_len();

        let mut tokenized_text = Vec::with_capacity(n);
        let mut i = 0;

        while i < n {
            let mut j = (i + max_token_len).min(n);

            while j > i {
                let token = text[i..j].concat();

                if let Some(token) = self.tokens.find_token(&token) {
                    tokenized_text.push(token);

                    break;
                }

                j -= 1;
            }

            if i == j {
                i += 1;
            } else {
                i = j;
            }
        }

        tokenized_text.into_boxed_slice()
    }

    /// Get iterator that will generate new tokens to the given prefix, using
    /// provided random numbers generator for seeding.
    pub fn generate<'model, R: Rng>(
        &'model self,
        content: impl AsRef<str>,
        rng: &'model mut R
    ) -> anyhow::Result<TokensGenerator<'model, R>> {
        // Parse model's template, stop tokens and prefill values.

        let template = self.keys.get("model.inference.template")
            .cloned()
            .unwrap_or_else(|| String::from("{{content}}"));

        let mut stop_tokens = Vec::new();
        let mut i = 0;

        while let Some(stop_token) = self.keys.get(&format!("model.inference.stop_tokens[{i}]")) {
            stop_tokens.push(stop_token.clone());

            i += 1;
        }

        let start_token = self.keys.get("model.tokens.start_token")
            .cloned()
            .unwrap_or_else(|| Self::START_TOKEN.to_string());

        let stop_token = self.keys.get("model.tokens.stop_token")
            .cloned()
            .unwrap_or_else(|| Self::STOP_TOKEN.to_string());

        // Format the query according to the template and encode it to tokens.

        let generation_prefix = template
            .replace("{{content}}", content.as_ref())
            .replace("{{start_token}}", &start_token)
            .replace("{{stop_token}}", &stop_token);

        let mut generation_prefix = self.tokenize(generation_prefix);

        if generation_prefix.is_empty() {
            let start_token = self.tokens.find_token(&start_token)
                .ok_or_else(|| anyhow::anyhow!("failed to find start token"))?;

            generation_prefix = Box::new([start_token]);
        }

        // Extend stop tokens with model's start and stop tokens.

        stop_tokens.push(start_token);
        stop_tokens.push(stop_token);

        // Parse inference parameters.

        let active_experts = self.keys.get("model.experts.active")
            .map(|value| value.parse::<usize>())
            .unwrap_or(Ok(0))?;

        let top_k = self.keys.get("model.inference.top_k")
            .map(|value| value.parse::<usize>())
            .unwrap_or(Ok(10))?;

        let max_tokens = self.keys.get("model.inference.max_tokens")
            .map(|value| value.parse::<usize>())
            .unwrap_or(Ok(1024))?;

        Ok(TokensGenerator {
            model: self,
            sequence_ptr: generation_prefix.len() - 1,
            sequence: generation_prefix.to_vec(),
            rng,
            stats: TokensGeneratorStats {
                experts_use: HashMap::from_iter({
                    self.experts.iter()
                        .enumerate()
                        .map(|(i, _)| (i, 0))
                })
            },
            stop_tokens,
            active_experts,
            top_k,
            max_tokens
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokensGeneratorStats {
    experts_use: HashMap<usize, usize>
}

impl TokensGeneratorStats {
    /// Get total amount of experts.
    pub fn total_experts(&self) -> usize {
        self.experts_use.keys().count()
    }

    /// Get use frequency for given expert index.
    pub fn expert_frequency(&self, expert: usize) -> Option<f32> {
        let total_calls = self.experts_use.values()
            .copied()
            .sum::<usize>();

        self.experts_use.get(&expert)
            .map(|calls| *calls as f32 / total_calls as f32)
    }
}

#[derive(Debug)]
pub struct TokensGenerator<'model, R: Rng> {
    model: &'model Model,
    sequence: Vec<u16>,
    sequence_ptr: usize,
    rng: &'model mut R,

    stats: TokensGeneratorStats,

    /// Tokens after which the inference must be stopped.
    stop_tokens: Vec<String>,

    /// Amount of experts to use for each token generation.
    active_experts: usize,

    /// Amount of best match token to randomly choose from.
    top_k: usize,

    /// Maximal amount of tokens to generate.
    max_tokens: usize
}

impl<R: Rng> TokensGenerator<'_, R> {
    #[inline(always)]
    pub const fn stats(&self) -> &TokensGeneratorStats {
        &self.stats
    }
}

impl<R: Rng> Iterator for TokensGenerator<'_, R> {
    type Item = Box<[u8]>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.top_k == 0 || self.sequence.len() >= self.max_tokens {
            return None;
        }

        if let Some(token) = self.sequence.get(self.sequence_ptr + 1) {
            self.sequence_ptr += 1;

            let token = self.model.tokens.find_word(*token)?;

            for stop_token in &self.stop_tokens {
                if stop_token.as_bytes() == token.as_ref() {
                    return None;
                }
            }

            return Some(token);
        }

        // Find best experts for the current tokens stream.

        let total_experts = self.model.experts.len();

        let mut experts = Vec::with_capacity(total_experts);

        for (i, expert) in self.model.experts.iter().enumerate() {
            let similarity = expert.similarity(self.sequence.iter().copied());

            experts.push((i, expert, similarity));
        }

        experts.sort_by(|a, b| {
            b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal)
        });

        experts.truncate(self.active_experts);

        for (i, _, _) in &experts {
            *self.stats.experts_use.entry(*i)
                .or_default() += 1;
        }

        // Find transitions from the base model and loaded experts.

        let mut transitions = self.model.transitions.find_transitions(&self.sequence)
            .into_iter()
            .map(|transition| (
                transition.from,
                transition.to,
                transition.weight as u64,
                1.0
            ))
            .collect::<Vec<_>>();

        let total_similarity = experts.iter()
            .map(|expert| expert.2)
            .sum::<f32>();

        for expert in experts {
            let expert_transitions = expert.1.find_transitions(&self.sequence)
                .into_iter()
                .map(|transition| (
                    transition.from,
                    transition.to,
                    transition.weight as u64,
                    expert.2 / total_similarity * (total_experts as f32 / self.active_experts as f32).sqrt()
                ))
                .collect::<Vec<_>>();

            transitions.extend(expert_transitions);
        }

        // Resolve tokens if it's trivial.

        if transitions.is_empty() {
            return None;
        }

        if let Some((_, to, _, _)) = transitions.first() {
            self.sequence.extend_from_slice(to);
            self.sequence_ptr += 1;

            let token = self.model.tokens.find_word(to[0])?;

            for stop_token in &self.stop_tokens {
                if stop_token.as_bytes() == token.as_ref() {
                    return None;
                }
            }

            return Some(token);
        }

        // Calculate normalized weights for each transition.

        let total_weight = transitions.iter()
            .map(|transition| transition.2 + 1)
            .sum::<u64>();

        let raw_transitions = transitions.into_iter()
            .map(|(from, to, weight, multiplier)| {
                let weight = (weight + 1) as f64 / total_weight as f64 * multiplier as f64;

                (from, to, weight.max(f64::EPSILON))
            })
            .collect::<Vec<_>>();

        let mut transitions = HashMap::<(Box<[u16]>, Box<[u16]>), f64>::with_capacity(
            raw_transitions.len()
        );

        for (from, to, weight) in raw_transitions {
            *transitions.entry((from, to)).or_default() += weight;
        }

        let mut transitions = transitions.into_iter()
            .map(|(k, v)| (k.0, k.1, v))
            .collect::<Vec<_>>();

        transitions.sort_by(|a, b| {
            b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal)
        });

        transitions.truncate(self.top_k);

        // Predict the next token.

        let total_weight = transitions.iter()
            .map(|transition| transition.2)
            .sum::<f64>();

        let target_weight = self.rng.next_u32() as f64 / u32::MAX as f64 * total_weight;

        let mut curr_weight = 0.0;

        for (_, to, weight) in &transitions {
            curr_weight += *weight;

            if curr_weight >= target_weight {
                self.sequence.extend_from_slice(to);
                self.sequence_ptr += 1;

                let token = self.model.tokens.find_word(to[0])?;

                for stop_token in &self.stop_tokens {
                    if stop_token.as_bytes() == token.as_ref() {
                        return None;
                    }
                }

                return Some(token);
            }
        }

        self.sequence.extend_from_slice(&transitions[0].1);
        self.sequence_ptr += 1;

        let token = self.model.tokens.find_word(transitions[0].1[0])?;

        for stop_token in &self.stop_tokens {
            if stop_token.as_bytes() == token.as_ref() {
                return None;
            }
        }

        Some(token)
    }
}

impl<R: Rng> FusedIterator for TokensGenerator<'_, R> {}
