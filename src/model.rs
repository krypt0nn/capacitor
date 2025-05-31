use std::collections::{HashMap, HashSet};

use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha12Rng;

use crate::tokens::{Token, TokensMap};
use crate::tokenizer::{Tokenizer, WordTokenizer};
use crate::transitions::{Transition, TransitionsMap};
use crate::clustering::{Cluster, clusterize};
use crate::recipe::{Recipe, Tokenizer as RecipeTokenizer};

#[derive(Debug, Clone)]
pub struct Expert<const SIZE: usize, T: Token<SIZE>> {
    cluster: Cluster<T>,
    transitions: TransitionsMap<SIZE, T>
}

impl<const SIZE: usize, T: Token<SIZE>> Expert<SIZE, T> {
    #[inline]
    pub fn distance(&self, document: impl IntoIterator<Item = T>) -> f32 {
        self.cluster.distance(document)
    }

    #[inline]
    pub fn transitions(&self, from: impl AsRef<[T]>) -> HashSet<Transition<SIZE, T>> {
        self.transitions.find_transitions(from)
    }
}

#[derive(Debug, Clone)]
pub struct Model<const SIZE: usize, T: Token<SIZE>> {
    keys: HashMap<String, String>,
    tokens: TokensMap<SIZE, T>,
    transitions: TransitionsMap<SIZE, T>,
    active_experts: usize,
    experts: Box<[Expert<SIZE, T>]>
}

impl<const SIZE: usize, T: Token<SIZE>> Model<SIZE, T> {
    pub const START_TOKEN: &'static str = "<|start|>";
    pub const STOP_TOKEN: &'static str = "<|stop|>";

    pub fn open(model: impl AsRef<[u8]>) -> anyhow::Result<Self> {
        let model = model.as_ref();
        let n = model.len();

        if n < 13 {
            anyhow::bail!("invalid model format: too short");
        }

        if &model[..9] != b"capacitor" {
            anyhow::bail!("invalid model format: not a capacitor model");
        }

        if &model[9..11] != b"v1" {
            anyhow::bail!("unsupported model format: v1 expected, got {}", String::from_utf8_lossy(&model[9..11]));
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

            offset += 2;

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

        let tokens_map = TokensMap::<SIZE, T>::open(&model[offset..offset + tokens_map_len]);

        offset += tokens_map_len;

        // Read base model transitions map.

        let transitions_map_len = u64::from_le_bytes([
            model[offset    ], model[offset + 1], model[offset + 2], model[offset + 3],
            model[offset + 4], model[offset + 5], model[offset + 6], model[offset + 7]
        ]) as usize;

        offset += 8;

        let transitions_map = TransitionsMap::<SIZE, T>::open(&model[offset..offset + transitions_map_len])?;

        offset += transitions_map_len;

        // Read experts.

        let total_experts = u32::from_le_bytes([
            model[offset    ], model[offset + 1],
            model[offset + 2], model[offset + 3]
        ]) as usize;

        let active_experts = u32::from_le_bytes([
            model[offset + 4], model[offset + 5],
            model[offset + 6], model[offset + 7]
        ]) as usize;

        offset += 8;
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

            let mut cluster = HashMap::<T, f32>::with_capacity(cluster_len);
            let mut token = [0; SIZE];
            let mut frequency = [0; 4];
            let mut j = 0;

            while j < cluster_len {
                token.copy_from_slice(&model[offset..offset + SIZE]);
                frequency.copy_from_slice(&model[offset + SIZE..offset + SIZE + 4]);

                cluster.insert(T::decode(token), f32::from_le_bytes(frequency));

                offset += SIZE + 4;
                j += 1;
            }

            // Read expert transitions matrix and store it.

            let transitions = TransitionsMap::<SIZE, T>::open(&model[offset..offset + transitions_map_len])?;

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
            active_experts,
            experts: experts.into_boxed_slice()
        })
    }

    pub fn into_bytes(self) -> Box<[u8]> {
        // I technically can calculate exact container size needed to store
        // this model but who cares?
        let mut model = Vec::new();

        model.extend_from_slice(b"capacitorv1");

        // Encode metadata keys.

        model.extend_from_slice(&(self.keys.len() as u16).to_le_bytes());

        for (key, value) in self.keys {
            model.push(key.len() as u8);
            model.extend_from_slice(&(value.len() as u16).to_le_bytes());
            model.extend_from_slice(key.as_bytes());
            model.extend_from_slice(value.as_bytes());
        }

        // Encode tokens map.

        let tokens = self.tokens.into_inner();

        model.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
        model.extend(tokens);

        // Encode base model transitions map.

        let transitions = self.transitions.into_inner();

        model.extend_from_slice(&(transitions.len() as u64).to_le_bytes());
        model.extend(transitions);

        // Encode experts.

        let total_experts = self.experts.len() as u64;
        let active_experts = self.active_experts as u64;

        model.extend_from_slice(&total_experts.to_le_bytes());
        model.extend_from_slice(&active_experts.to_le_bytes());

        for expert in self.experts {
            let cluster = expert.cluster.into_inner();
            let transitions = expert.transitions.into_inner();

            model.extend_from_slice(&(cluster.len() as u32).to_le_bytes());
            model.extend_from_slice(&(transitions.len() as u64).to_le_bytes());

            for (token, rank) in cluster {
                model.extend_from_slice(&token.encode());
                model.extend_from_slice(&rank.to_le_bytes());
            }

            model.extend(transitions);
        }

        model.into_boxed_slice()
    }

    pub fn build(recipe: Recipe) -> anyhow::Result<Self> {
        // Read documents from the dataset files.

        let mut documents = Vec::with_capacity(recipe.files.len());

        for file in recipe.files {
            let dataset = std::fs::read_to_string(file.path)?;

            for document in dataset.split(&file.delimiter) {
                documents.push(document.trim().to_string());
            }
        }

        // Prepare tokenizer.

        #[allow(irrefutable_let_patterns)]
        let RecipeTokenizer::WordTokenizer { lowercase, punctuation } = recipe.tokenizer else {
            anyhow::bail!("only word-tokenizer is currently supported");
        };

        let tokenizer = WordTokenizer { lowercase, punctuation };

        // Tokenize documents.

        let mut tokens = HashMap::<String, T>::new();
        let mut token = T::zero();

        let start_token = ("<|start|>", token);
        let stop_token = ("<|stop|>", token.inc());

        tokens.insert(start_token.0.to_string(), start_token.1);
        tokens.insert(stop_token.0.to_string(), stop_token.1);

        token = token.inc().inc();

        let documents = documents.into_iter()
            .map(|document| {
                let mut tokenized_document = Vec::new();

                tokenized_document.push(start_token.1);

                for word in tokenizer.encode(document.as_bytes()) {
                    let word = word?;

                    if !tokens.contains_key(&word) {
                        tokens.insert(word.clone(), token);

                        token = token.inc();
                    }

                    let token = tokens.get(&word).unwrap_or(&token);

                    tokenized_document.push(*token);
                }

                tokenized_document.push(stop_token.1);

                Ok(tokenized_document.into_boxed_slice())
            })
            .collect::<anyhow::Result<Vec<Box<[T]>>>>()?;

        // Create tokens map.

        let tokens_map = TokensMap::<SIZE, T>::from_words(tokens.keys())?;

        // Count transitions for every document.

        let mut transitions = HashMap::<&[T], HashMap<(&[T], &[T]), usize>>::new();
        let min_len = recipe.from_depth + recipe.to_depth;

        for document in &documents {
            if document.len() < min_len {
                continue;
            }

            let mut document_transitions = HashMap::<(&[T], &[T]), usize>::new();

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

            transitions.insert(document, document_transitions);
        }

        // Create transitions map for the whole dataset.

        let mut cummulative_transitions = HashMap::<&(&[T], &[T]), usize>::new();

        for document_transitions in transitions.values() {
            for (transition, count) in document_transitions.iter() {
                *cummulative_transitions.entry(transition)
                    .or_default() += *count;
            }
        }

        let total_transitions = cummulative_transitions.values()
            .copied()
            .sum::<usize>();

        let cummulative_transitions = cummulative_transitions.into_iter()
            .map(|(transition, count)| {
                let frequency = count as f32 / total_transitions as f32;

                (transition.0, transition.1, (frequency * u16::MAX as f32) as u16)
            })
            .collect::<Vec<_>>();

        let transitions_map = TransitionsMap::<SIZE, T>::from_transitions(cummulative_transitions)?;

        // Clusterize documents.

        let micros = std::time::UNIX_EPOCH.elapsed()
            .unwrap_or_default()
            .as_micros();

        let mut rand = ChaCha12Rng::seed_from_u64((micros & (u64::MAX as u128)) as u64);

        let clusters = clusterize(
            recipe.total_experts,
            recipe.centroids,
            &documents,
            &mut rand
        )?;

        // Assign each document to the closest cluster.

        let mut documents_clusters = vec![Vec::new(); clusters.len()];

        for document in &documents {
            let mut document_cluster = 0;
            let mut distance = f32::MAX;

            for (i, cluster) in clusters.iter().enumerate() {
                let cluster_distance = cluster.distance(document.iter().copied());

                if cluster_distance < distance {
                    document_cluster = i;
                    distance = cluster_distance;
                }
            }

            documents_clusters[document_cluster].push(document);
        }

        // Create experts from clusters.

        let mut experts = Vec::with_capacity(clusters.len());

        for (i, cluster) in clusters.into_iter().enumerate() {
            let mut cluster_transitions = HashMap::<&(&[T], &[T]), usize>::new();

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

            let cluster_transitions = cluster_transitions.into_iter()
                .map(|(transition, count)| {
                    let frequency = count as f32 / total_transitions as f32;

                    (transition.0, transition.1, (frequency * u16::MAX as f32) as u16)
                })
                .collect::<Vec<_>>();

            let transitions = TransitionsMap::<SIZE, T>::from_transitions(cluster_transitions)?;

            experts.push(Expert {
                cluster,
                transitions
            });
        }

        // Build the model.

        Ok(Self {
            keys: recipe.keys,
            tokens: tokens_map,
            transitions: transitions_map,
            active_experts: recipe.active_experts,
            experts: experts.into_boxed_slice()
        })
    }
}

pub type Model8 = Model<1, u8>;
pub type Model16 = Model<2, u16>;
pub type Model32 = Model<4, u32>;
pub type Model64 = Model<8, u64>;
