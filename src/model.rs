use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::str::FromStr;
use std::iter::FusedIterator;

use rand_chacha::rand_core::RngCore;

use crate::rand;
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
    #[inline(always)]
    pub const fn cluster(&self) -> &Cluster<T> {
        &self.cluster
    }

    #[inline(always)]
    pub const fn transitions(&self) -> &TransitionsMap<SIZE, T> {
        &self.transitions
    }

    #[inline]
    pub fn similarity(&self, document: impl IntoIterator<Item = T>) -> f32 {
        self.cluster.similarity(document)
    }

    #[inline]
    pub fn find_transitions(&self, from: impl AsRef<[T]>) -> HashSet<Transition<SIZE, T>> {
        self.transitions.find_transitions(from)
    }
}

#[derive(Debug, Clone)]
pub struct Model<const SIZE: usize, T: Token<SIZE>> {
    keys: HashMap<String, String>,
    tokens: TokensMap<SIZE, T>,
    transitions: TransitionsMap<SIZE, T>,
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

        model.extend_from_slice(&(self.experts.len() as u32).to_le_bytes());

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

    pub fn build(mut recipe: Recipe) -> anyhow::Result<Self> {
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
        let mut words = HashSet::new();
        let mut token = T::zero();

        let start_token = (Self::START_TOKEN, token);
        let stop_token = (Self::STOP_TOKEN, token.inc());

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
                        words.insert(word.clone());

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

        let tokens_map = TokensMap::<SIZE, T>::from_words(words)?;

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

                (transition.0, transition.1, (frequency * u32::MAX as f32) as u32)
            })
            .collect::<Vec<_>>();

        let transitions_map = TransitionsMap::<SIZE, T>::from_transitions(cummulative_transitions)?;

        // Clusterize documents.

        let clusters = clusterize(
            recipe.total_experts,
            recipe.centroids,
            &documents,
            &mut rand()
        )?;

        // Assign each document to the closest cluster.

        let mut documents_clusters = vec![Vec::new(); clusters.len()];

        for document in &documents {
            let mut document_cluster = 0;
            let mut similarity = f32::MIN;

            for (i, cluster) in clusters.iter().enumerate() {
                let cluster_similarity = cluster.similarity(document.iter().copied());

                if cluster_similarity > similarity {
                    document_cluster = i;
                    similarity = cluster_similarity;
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

                    (transition.0, transition.1, (frequency * u32::MAX as f32) as u32)
                })
                .collect::<Vec<_>>();

            let transitions = TransitionsMap::<SIZE, T>::from_transitions(cluster_transitions)?;

            experts.push(Expert {
                cluster,
                transitions
            });
        }

        // Prefill default metadata keys.

        recipe.keys.entry(String::from("model.name"))
            .or_insert(recipe.name.clone());

        recipe.keys.entry(String::from("model.tokens.tokenizer"))
            .or_insert(recipe.tokenizer.to_string());

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

        // Build the model.

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
    pub const fn tokens_ref(&self) -> &TokensMap<SIZE, T> {
        &self.tokens
    }

    #[inline(always)]
    pub const fn transitions_ref(&self) -> &TransitionsMap<SIZE, T> {
        &self.transitions
    }

    #[inline(always)]
    pub const fn experts_ref(&self) -> &[Expert<SIZE, T>] {
        &self.experts
    }

    pub fn get_tokenizer(&self) -> anyhow::Result<Option<RecipeTokenizer>> {
        let Some(tokenizer) = self.keys.get("model.tokens.tokenizer") else {
            return Ok(None);
        };

        Ok(Some(RecipeTokenizer::from_str(tokenizer)?))
    }

    pub fn encode_tokens(&self, text: impl AsRef<str>) -> anyhow::Result<Box<[T]>> {
        #[allow(irrefutable_let_patterns)]
        let Some(RecipeTokenizer::WordTokenizer { lowercase, punctuation }) = self.get_tokenizer()? else {
            anyhow::bail!("only word-tokenizer is currently supported");
        };

        let tokenizer = WordTokenizer { lowercase, punctuation };

        let text = text.as_ref().as_bytes();

        let mut tokens = Vec::new();

        for token in tokenizer.encode(text) {
            if let Some(token) = self.tokens.find_token(token?) {
                tokens.push(token);
            }
        }

        Ok(tokens.into_boxed_slice())
    }

    pub fn generate_tokens<'model, R: RngCore>(
        &'model self,
        sequence: impl Into<Vec<T>>,
        rand: &'model mut R
    ) -> anyhow::Result<TokensGenerator<'model, SIZE, T, R>> {
        let active_experts = self.keys.get("model.experts.active")
            .map(|value| value.parse::<usize>())
            .unwrap_or(Ok(0))?;

        let top_k = self.keys.get("model.inference.top_k")
            .map(|value| value.parse::<usize>())
            .unwrap_or(Ok(10))?;

        let max_tokens = self.keys.get("model.inference.max_tokens")
            .map(|value| value.parse::<usize>())
            .unwrap_or(Ok(200))?;

        let sequence: Vec<T> = sequence.into();

        // if let Some(token) = self.tokens.find_token(Self::START_TOKEN) {
        //     sequence.insert(0, token);
        // }

        Ok(TokensGenerator {
            model: self,
            sequence_ptr: sequence.len() - 1,
            sequence,
            rand,
            active_experts,
            top_k,
            max_tokens
        })
    }
}

pub type Model8 = Model<1, u8>;
pub type Model16 = Model<2, u16>;
pub type Model32 = Model<4, u32>;
pub type Model64 = Model<8, u64>;

#[derive(Debug)]
pub struct TokensGenerator<'model, const SIZE: usize, T: Token<SIZE>, R: RngCore> {
    model: &'model Model<SIZE, T>,
    sequence: Vec<T>,
    sequence_ptr: usize,
    rand: &'model mut R,

    /// Amount of experts to use for each token generation.
    active_experts: usize,

    /// Amount of best match token to randomly choose from.
    top_k: usize,

    /// Maximal amount of tokens to generate.
    max_tokens: usize
}

impl<const SIZE: usize, T: Token<SIZE>, R: RngCore> Iterator for TokensGenerator<'_, SIZE, T, R> {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        if self.top_k == 0 || self.sequence.len() >= self.max_tokens {
            return None;
        }

        if let Some(token) = self.sequence.get(self.sequence_ptr + 1) {
            self.sequence_ptr += 1;

            let token = self.model.tokens.find_word(*token)?;

            if token == Model::<SIZE, T>::STOP_TOKEN {
                return None;
            }

            return Some(token);
        }

        // Find best experts for the current tokens stream.

        let total_experts = self.model.experts.len();

        let mut experts = Vec::with_capacity(total_experts);

        for expert in &self.model.experts {
            let similarity = expert.similarity(self.sequence.iter().copied());

            experts.push((expert, similarity));
        }

        experts.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
        });

        experts.shrink_to(self.active_experts);

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
            .map(|expert| expert.1)
            .sum::<f32>();

        for expert in experts {
            let expert_transitions = expert.0.find_transitions(&self.sequence)
                .into_iter()
                .map(|transition| (
                    transition.from,
                    transition.to,
                    transition.weight as u64,
                    expert.1 / total_similarity * total_experts as f32
                ))
                .collect::<Vec<_>>();

            transitions.extend(expert_transitions);
        }

        // Resolve tokens if it's trivial.

        if transitions.is_empty() {
            return None;
        }

        if transitions.len() == 1 {
            if let Some((_, to, _, _)) = transitions.first() {
                self.sequence.extend_from_slice(to);
                self.sequence_ptr += 1;

                let token = self.model.tokens.find_word(to[0])?;

                if token == Model::<SIZE, T>::STOP_TOKEN {
                    return None;
                }

                return Some(token);
            }
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

        let mut transitions = HashMap::<(Box<[T]>, Box<[T]>), f64>::with_capacity(raw_transitions.len());

        for (from, to, weight) in raw_transitions {
            *transitions.entry((from, to)).or_default() += weight;
        }

        let mut transitions = transitions.into_iter()
            .map(|(k, v)| (k.0, k.1, v))
            .collect::<Vec<_>>();

        transitions.sort_by(|a, b| {
            b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal)
        });

        transitions.shrink_to(self.top_k);

        // Predict the next token.

        let total_weight = transitions.iter()
            .map(|transition| transition.2)
            .sum::<f64>();

        let target_weight = self.rand.next_u32() as f64 / u32::MAX as f64 * total_weight;

        let mut curr_weight = 0.0;

        for (_, to, weight) in &transitions {
            curr_weight += *weight;

            if curr_weight >= target_weight {
                self.sequence.extend_from_slice(to);
                self.sequence_ptr += 1;

                let token = self.model.tokens.find_word(to[0])?;

                if token == Model::<SIZE, T>::STOP_TOKEN {
                    return None;
                }

                return Some(token);
            }
        }

        self.sequence.extend_from_slice(&transitions[0].1);
        self.sequence_ptr += 1;

        let token = self.model.tokens.find_word(transitions[0].1[0])?;

        if token == Model::<SIZE, T>::STOP_TOKEN {
            return None;
        }

        Some(token)
    }
}

impl<const SIZE: usize, T: Token<SIZE>, R: RngCore> FusedIterator for TokensGenerator<'_, SIZE, T, R> {}

// #[derive(Debug)]
// pub struct TextGenerator<'model, const SIZE: usize, T: Token<SIZE>, R: RngCore> {
//     tokens: TokensGenerator<'model, SIZE, T, R>,
//     buf: Vec<u8>
// }

// impl<const SIZE: usize, T: Token<SIZE>, R: RngCore> Read for TextGenerator<'_, SIZE, T, R> {
//     fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
//         if let Some(token) = self.tokens.next() {

//         }
//     }
// }
