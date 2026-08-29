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

use rand_chacha::rand_core::Rng;

use crate::tokens::{pre_tokenize, TokensMap};
use crate::transitions::{Transition, TransitionsMap};
use crate::clustering::Cluster;
use crate::recipe::Recipe;

mod builder;
mod generator;

pub use builder::Progress as BuildProgress;
pub use generator::{TokensGenerator, TokensGeneratorStats};

/// For every multi-word token, return the token id of its bare last word
/// (` chat` -> `chat`, `hi chat` -> `chat`).
fn word_aliases(tokens: &TokensMap) -> HashMap<u16, u16> {
    let mut words = HashMap::<Box<[u8]>, u16>::new();
    let mut multi_word = Vec::new();

    tokens.for_each(|token, word| {
        let Ok(word) = std::str::from_utf8(&word) else {
            return;
        };

        words.insert(word.as_bytes().into(), token);

        if word.contains(char::is_whitespace) {
            multi_word.push((token, word.to_owned()));
        }
    });

    let mut aliases = HashMap::new();

    for (token, word) in multi_word {
        let Some(last_word) = word.split_whitespace().last() else {
            continue;
        };

        if let Some(&id) = words.get(last_word.as_bytes()) {
            aliases.insert(token, id);
        }
    }

    aliases
}

#[derive(Debug, Clone)]
pub struct Expert {
    cluster: Cluster,
    transitions: TransitionsMap
}

impl Expert {
    #[inline(always)]
    pub const fn cluster(&self) -> &Cluster {
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
    /// Model metadata keys.
    keys: HashMap<String, String>,

    /// Map of every token known to the model.
    tokens: TokensMap,

    /// Map of transitions between all the known tokens.
    transitions: TransitionsMap,

    /// List of model experts and their transitions.
    experts: Box<[Expert]>,

    /// From-grams in the transitions maps are keyed on bare last words, so
    /// transition lookups must normalize their token tails the same way.
    word_aliases: HashMap<u16, u16>
}

impl Model {
    pub const START_TOKEN: &str = "<|start|>";
    pub const STOP_TOKEN: &str = "<|stop|>";

    pub const DEFAULT_TOP_K: usize = 10;
    pub const DEFAULT_TEMPERATURE: f32 = 1.0;
    pub const DEFAULT_MAX_TOKENS: usize = 200;
    pub const DEFAULT_EXPERTS_CONTEXT: usize = 32;

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

        let word_aliases = word_aliases(&tokens_map);

        Ok(Self {
            keys,
            tokens: tokens_map,
            transitions: transitions_map,
            experts: experts.into_boxed_slice(),
            word_aliases
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

    #[inline]
    pub fn build(
        recipe: Recipe,
        rng: impl Rng,
        progress: impl Fn(BuildProgress) + Send + Sync
    ) -> anyhow::Result<Self> {
        builder::build(recipe, rng, progress)
    }

    /// Encode given text using model's tokenizer. Unknown tokens will be
    /// skipped entirely.
    pub fn tokenize(
        &self,
        text: impl AsRef<str>
    ) -> Box<[u16]> {
        // Lookup tokenizer settings.
        let make_lowercase = self.keys.get("model.tokenizer.make_lowercase")
            .map(|value| value.as_str())
            .unwrap_or("false") == "true";

        let force_alphanumeric = self.keys.get("model.tokenizer.force_alphanumeric")
            .map(|value| value.as_str())
            .unwrap_or("false") == "true";

        // Pre-tokenize given text.
        let text = pre_tokenize(
            text.as_ref(),
            make_lowercase,
            force_alphanumeric
        );

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

    /// Copy of the given tokens with every token replaced by the token of its
    /// bare last word - matches the from-gram keys stored in the transitions
    /// maps (` chat` -> `chat`, `hi chat` -> `chat`).
    pub fn normalize_tail(&self, tail: &[u16]) -> Box<[u16]> {
        tail.iter()
            .map(|&token| {
                self.word_aliases.get(&token)
                    .copied()
                    .unwrap_or(token)
            })
            .collect()
    }

    /// Get iterator that will generate new tokens to the given prefix, using
    /// provided random numbers generator for seeding.
    #[inline]
    pub fn generate<'model, R: Rng>(
        &'model self,
        content: impl AsRef<str>,
        rng: R
    ) -> anyhow::Result<TokensGenerator<'model, R>> {
        TokensGenerator::new(self, content, rng)
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
}
