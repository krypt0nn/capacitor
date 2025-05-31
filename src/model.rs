use std::collections::{HashMap, HashSet};

use crate::tokens::{Token, TokensMap};
use crate::transitions::{Transition, TransitionsMap};
use crate::clustering::Cluster;

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
    keys: HashMap<Box<[u8]>, Box<[u8]>>,
    tokens: TokensMap<SIZE, T>,
    transitions: TransitionsMap<SIZE, T>,
    active_experts: usize,
    experts: Box<[Expert<SIZE, T>]>
}

impl<const SIZE: usize, T: Token<SIZE>> Model<SIZE, T> {
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

        let mut keys = HashMap::<Box<[u8]>, Box<[u8]>>::with_capacity(keys_num);

        let mut offset = 13;
        let mut i = 0;

        while i < keys_num {
            let key_len = model[offset] as usize;

            let value_len = u16::from_le_bytes([
                model[offset + 1],
                model[offset + 2]
            ]) as usize;

            offset += 2;

            let key = &model[offset..offset + key_len];
            let value = &model[offset + key_len..offset + key_len + value_len];

            offset += key_len + value_len;
            i += 1;

            keys.insert(key.into(), value.into());
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
            model.extend(key);
            model.extend(value);
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
}
