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

use std::cmp::Ordering;
use std::collections::HashMap;
use std::iter::FusedIterator;
use std::io::{Read, Write};

use rand_chacha::rand_core::Rng;

use super::Model;

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
    rng: R,
    pending: Vec<u8>,

    stats: TokensGeneratorStats,

    /// Tokens after which the inference must be stopped.
    stop_tokens: Vec<String>,

    /// Amount of experts to use for each token generation.
    active_experts: usize,

    /// Amount of best match token to randomly choose from.
    top_k: usize,

    /// Exponent applied to candidate weights before sampling via
    /// `weight^(1 / temperature)`.
    ///
    /// Values above `1.0` flatten the distribution. Values below `1.0` sharpen
    /// it toward the greedy choice.
    ///
    /// `1.0` samples proportionally to weights.
    temperature: f32,

    /// Maximal amount of tokens to generate.
    max_tokens: usize,

    /// Amount of context tokens to use for experts selection.
    experts_context: usize
}

impl<'model, R: Rng> TokensGenerator<'model, R> {
    pub fn new(
        model: &'model Model,
        content: impl AsRef<str>,
        rng: R
    ) -> anyhow::Result<Self> {
        // Parse model's template, stop tokens and prefill values.

        let template = model.keys.get("model.inference.template")
            .cloned()
            .unwrap_or_else(|| String::from("{{content}}"));

        let mut stop_tokens = Vec::new();
        let mut i = 0;

        while let Some(stop_token) = model.keys.get(&format!("model.inference.stop_tokens[{i}]")) {
            stop_tokens.push(stop_token.clone());

            i += 1;
        }

        let start_token = model.keys.get("model.tokens.start_token")
            .cloned()
            .unwrap_or_else(|| Model::START_TOKEN.to_string());

        let stop_token = model.keys.get("model.tokens.stop_token")
            .cloned()
            .unwrap_or_else(|| Model::STOP_TOKEN.to_string());

        // Format the query according to the template and encode it to tokens.

        let generation_prefix = template
            .replace("{{content}}", content.as_ref())
            .replace("{{start_token}}", &start_token)
            .replace("{{stop_token}}", &stop_token);

        let mut generation_prefix = model.tokenize(generation_prefix);

        if generation_prefix.is_empty() {
            let start_token = model.tokens.find_token(&start_token)
                .ok_or_else(|| anyhow::anyhow!("failed to find start token"))?;

            generation_prefix = Box::new([start_token]);
        }

        // Extend stop tokens with model's start and stop tokens.

        stop_tokens.push(start_token);
        stop_tokens.push(stop_token);

        // Parse inference parameters.

        let top_k = model.keys.get("model.inference.top_k")
            .map(|value| value.parse::<usize>())
            .unwrap_or(Ok(Model::DEFAULT_TOP_K))?
            .max(1);

        let temperature = model.keys.get("model.inference.temperature")
            .map(|value| value.parse::<f32>())
            .unwrap_or(Ok(Model::DEFAULT_TEMPERATURE))
            .map(|value| if value > 0.0 { value } else { 1.0 })?;

        let max_tokens = model.keys.get("model.inference.max_tokens")
            .map(|value| value.parse::<usize>())
            .unwrap_or(Ok(Model::DEFAULT_MAX_TOKENS))?;

        let active_experts = model.keys.get("model.inference.active_experts")
            .or_else(|| model.keys.get("model.experts.active"))
            .map(|value| value.parse::<usize>())
            .unwrap_or(Ok(0))?;

        let experts_context = model.keys.get("model.inference.experts_context")
            .or_else(|| model.keys.get("model.experts.context"))
            .map(|value| value.parse::<usize>())
            .unwrap_or(Ok(Model::DEFAULT_EXPERTS_CONTEXT))?;

        Ok(TokensGenerator {
            model,
            sequence_ptr: generation_prefix.len() - 1,
            sequence: generation_prefix.to_vec(),
            rng,
            pending: Vec::new(),
            stats: TokensGeneratorStats {
                experts_use: HashMap::from_iter({
                    model.experts.iter()
                        .enumerate()
                        .map(|(i, _)| (i, 0))
                })
            },
            stop_tokens,
            active_experts,
            top_k,
            temperature,
            max_tokens,
            experts_context
        })
    }

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

        // Emit buffered tokens of the previous transition as is, without any
        // lookups or randomness.
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
        //
        // Similarities are scored over a bounded recent window of the sequence,
        // not over its whole history - occurrence-summed ranks of ever-growing
        // contexts ossify routing onto whatever topic dominated long ago and
        // drown out the current one.

        let total_experts = self.model.experts.len();

        let experts_context = &self.sequence[
            self.sequence.len().saturating_sub(self.experts_context)..
        ];

        let mut experts = Vec::with_capacity(total_experts);

        for (i, expert) in self.model.experts.iter().enumerate() {
            let similarity = expert.similarity(experts_context.iter().copied());

            experts.push((i, expert, similarity));
        }

        experts.sort_by(|a, b| {
            b.2.partial_cmp(&a.2)
                .unwrap_or(Ordering::Equal)
        });

        experts.truncate(self.active_experts);

        for (i, _, _) in &experts {
            *self.stats.experts_use.entry(*i)
                .or_default() += 1;
        }

        // Lookup tails are normalized the same way as the stored from-grams:
        // every token is represented by the bare last word it contains.
        let normalized = self.model.normalize_tail(&self.sequence);

        // Find transitions from the base model and loaded experts.
        //
        // The context is trimmed down from the full tail until some
        // continuation is found - deep exact matches are preferred, but
        // generation must not die just because the query phrasing differs
        // from anything in the corpus.

        let max_ctx = self.model.transitions.from_count()
            .min(self.sequence.len());

        let mut ctx_len = max_ctx;
        let mut transitions = Vec::new();

        while ctx_len >= 1 {
            transitions = self.model.transitions
                .find_transitions(&normalized[normalized.len() - ctx_len..])
                .into_iter()
                .map(|transition| (
                    transition.from,
                    transition.to,
                    transition.weight as u64,
                    1.0
                ))
                .collect::<Vec<_>>();

            if !transitions.is_empty() {
                break;
            }

            if ctx_len > 1 {
                ctx_len -= 1;

                continue;
            }

            break;
        }

        let total_similarity = experts.iter()
            .map(|expert| expert.2)
            .sum::<f32>();

        for expert in experts {
            let expert_transitions = expert.1.find_transitions(&normalized)
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

        // Stop generation if there's nothing to choose from.
        if transitions.is_empty() {
            return None;
        }

        // Take the only available continuation as is.
        if transitions.len() == 1 {
            let to = &transitions[0].1;

            self.sequence.extend(to);
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

        // Sum duplicate pairs and apply the temperature. BTreeMap keeps the
        // summation and iteration order deterministic - HashSet iteration is
        // randomized per process, which would leak into tie-breaks below.

        let mut candidates = std::collections::BTreeMap::<(Box<[u16]>, Box<[u16]>), f64>::new();

        let temperature_pow = 1.0 / self.temperature as f64;

        for (from, to, weight) in raw_transitions {
            *candidates.entry((from, to))
                .or_default() += weight.powf(temperature_pow);
        }

        let mut transitions = candidates.into_iter()
            .map(|(k, v)| (k.0, k.1, v))
            .collect::<Vec<_>>();

        transitions.sort_by(|a, b| {
            // Deterministic tie-break: lexicographic by (from, to) instead
            // of randomized HashMap order.
            b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
                .then_with(|| a.1.cmp(&b.1))
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
                self.sequence.extend(to);
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

        self.sequence.extend(&transitions[0].1);
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

impl<R: Rng> Read for TokensGenerator<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if !self.pending.is_empty() {
            let n = self.pending.as_slice().read(buf)?;
            if n < self.pending.len() {
                self.pending.drain(..n);
                return Ok(n);
            }
            self.pending.clear();
            return Ok(n);
        }

        if buf.is_empty() {
            return Ok(0);
        }

        let Some(token) = self.next() else {
            return Ok(0);
        };

        if buf.len() >= token.len() {
            buf[..token.len()].copy_from_slice(&token);
            Ok(token.len())
        } else {
            buf.copy_from_slice(&token[..buf.len()]);
            self.pending = token[buf.len()..].to_vec();
            Ok(buf.len())
        }
    }
}
