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

// `[token - 2 bytes][word_len - 2 bytes][word - variable]`
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TokensMap(Box<[u8]>);

impl TokensMap {
    #[inline(always)]
    pub fn open(map: impl Into<Box<[u8]>>) -> Self {
        Self(map.into())
    }

    #[inline(always)]
    pub fn into_inner(self) -> Box<[u8]> {
        self.0
    }

    pub fn from_words<F: AsRef<[u8]>>(
        words: impl IntoIterator<Item = F>
    ) -> anyhow::Result<Self> {
        let mut unique_words = HashSet::new();
        let mut token = 0_u16;
        let mut map = Vec::new();

        for word in words {
            let word = word.as_ref();

            if word.len() > 65535 {
                anyhow::bail!("BPE words must be shorter than 65536 bytes");
            }

            if unique_words.insert(word.to_vec()) {
                map.extend(token.to_le_bytes());
                map.extend((word.len() as u16).to_le_bytes());
                map.extend(word);

                token += 1;
            }
        }

        Ok(Self(map.into_boxed_slice()))
    }

    pub fn for_each(&self, mut callback: impl FnMut(u16, Box<[u8]>)) {
        let mut i = 0;
        let n = self.0.len();

        while i < n {
            let token = u16::from_le_bytes([
                self.0[i], self.0[i + 1]
            ]);

            let word_len = u16::from_le_bytes([
                self.0[i + 2], self.0[i + 3]
            ]) as usize;

            i += 4;

            let word = self.0[i..i + word_len].to_vec()
                .into_boxed_slice();

            callback(token, word);

            i += word_len;
        }
    }

    pub fn find_token(&self, word: impl AsRef<[u8]>) -> Option<u16> {
        let mut i = 0;
        let n = self.0.len();
        let word = word.as_ref();

        while i < n {
            let token = u16::from_le_bytes([
                self.0[i], self.0[i + 1]
            ]);

            let word_len = u16::from_le_bytes([
                self.0[i + 2], self.0[i + 3]
            ]) as usize;

            i += 4;

            let token_word = &self.0[i..i + word_len];

            if word == token_word {
                return Some(token);
            }

            i += word_len;
        }

        None
    }

    pub fn find_word(&self, token: u16) -> Option<Box<[u8]>> {
        let mut i = 0;
        let n = self.0.len();

        while i < n {
            let curr_token = u16::from_le_bytes([
                self.0[i], self.0[i + 1]
            ]);

            let word_len = u16::from_le_bytes([
                self.0[i + 2], self.0[i + 3]
            ]) as usize;

            i += 4;

            if curr_token == token {
                let word = self.0[i..i + word_len].to_vec()
                    .into_boxed_slice();

                return Some(word);
            }

            i += word_len;
        }

        None
    }

    pub fn as_words_table(&self) -> HashMap<Box<[u8]>, u16> {
        let mut tokens = HashMap::new();

        self.for_each(|token, word| {
            tokens.insert(word, token);
        });

        tokens
    }

    pub fn as_tokens_table(&self) -> HashMap<u16, Box<[u8]>> {
        let mut tokens = HashMap::new();

        self.for_each(|token, word| {
            tokens.insert(token, word);
        });

        tokens
    }

    /// Amount of stored tokens.
    pub const fn len(&self) -> usize {
        let mut i = 2;
        let mut len = 0;

        let n = self.0.len();

        while i < n {
            let word_len = u16::from_le_bytes([
                self.0[i], self.0[i + 1]
            ]) as usize;

            i += 4 + word_len;
            len += 1;
        }

        len
    }

    /// Size of the map in bytes.
    #[inline]
    pub const fn size(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Length of the longest stored token.
    pub const fn max_token_len(&self) -> usize {
        let mut i = 2;
        let mut max_len = 0;

        let n = self.0.len();

        while i < n {
            let word_len = u16::from_le_bytes([
                self.0[i], self.0[i + 1]
            ]) as usize;

            if max_len < word_len {
                max_len = word_len;
            }

            i += 4 + word_len;
        }

        max_len
    }
}

#[test]
fn test_tokens_map() -> anyhow::Result<()> {
    let map = TokensMap::from_words([
        "hello",
        "world"
    ])?;

    let hello_token = map.find_token("hello").unwrap();
    let world_token = map.find_token("world").unwrap();

    assert_ne!(hello_token, world_token);

    assert_eq!(map.len(), 2);
    assert_eq!(map.size(), 18);
    assert_eq!(map.find_word(hello_token).as_deref(), Some(b"hello".as_slice()));
    assert_eq!(map.find_word(world_token).as_deref(), Some(b"world".as_slice()));

    assert!(map.find_token("amogus").is_none());
    assert!(map.find_word(42_u16).is_none());

    let table = map.as_tokens_table();

    assert_eq!(table.len(), 2);
    assert_eq!(table.get(&hello_token), Some(&b"hello".to_vec().into_boxed_slice()));
    assert_eq!(table.get(&world_token), Some(&b"world".to_vec().into_boxed_slice()));

    let table = map.as_words_table();

    assert_eq!(table.len(), 2);
    assert_eq!(table.get(b"hello".as_slice()), Some(&hello_token));
    assert_eq!(table.get(b"world".as_slice()), Some(&world_token));

    Ok(())
}
