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
use std::marker::PhantomData;

pub trait Token<const SIZE: usize>:
    std::fmt::Debug +
    Clone +
    Copy +
    Send +
    Sync +
    PartialEq +
    Eq +
    PartialOrd +
    Ord +
    std::hash::Hash
{
    fn encode(&self) -> [u8; SIZE];
    fn decode(bytes: [u8; SIZE]) -> Self;

    fn zero() -> Self {
        Self::decode([0; SIZE])
    }

    fn inc(&self) -> Self {
        let mut bytes = self.encode();

        #[allow(clippy::needless_range_loop)]
        for i in 0..SIZE {
            if bytes[i] < 255 {
                bytes[i] += 1;

                break;
            }

            else {
                bytes[i] = 0;
            }
        }

        Self::decode(bytes)
    }

    fn nth(n: usize) -> Self {
        let mut token = Self::zero();

        for _ in 0..n {
            token = token.inc();
        }

        token
    }
}

impl Token<1> for u8 {
    #[inline]
    fn encode(&self) -> [u8; 1] {
        [*self]
    }

    #[inline]
    fn decode(bytes: [u8; 1]) -> Self {
        bytes[0]
    }

    #[inline]
    fn zero() -> Self {
        0
    }

    #[inline]
    fn inc(&self) -> Self {
        *self + 1
    }

    #[inline]
    fn nth(n: usize) -> Self {
        (n % 256) as u8
    }
}

impl Token<2> for u16 {
    #[inline]
    fn encode(&self) -> [u8; 2] {
        self.to_le_bytes()
    }

    #[inline]
    fn decode(bytes: [u8; 2]) -> Self {
        u16::from_le_bytes(bytes)
    }

    #[inline]
    fn zero() -> Self {
        0
    }

    #[inline]
    fn inc(&self) -> Self {
        *self + 1
    }

    #[inline]
    fn nth(n: usize) -> Self {
        (n % 65536) as u16
    }
}

impl Token<4> for u32 {
    #[inline]
    fn encode(&self) -> [u8; 4] {
        self.to_le_bytes()
    }

    #[inline]
    fn decode(bytes: [u8; 4]) -> Self {
        u32::from_le_bytes(bytes)
    }

    #[inline]
    fn zero() -> Self {
        0
    }

    #[inline]
    fn inc(&self) -> Self {
        *self + 1
    }

    #[inline]
    fn nth(n: usize) -> Self {
        n as u32
    }
}

impl Token<8> for u64 {
    #[inline]
    fn encode(&self) -> [u8; 8] {
        self.to_le_bytes()
    }

    #[inline]
    fn decode(bytes: [u8; 8]) -> Self {
        u64::from_le_bytes(bytes)
    }

    #[inline]
    fn zero() -> Self {
        0
    }

    #[inline]
    fn inc(&self) -> Self {
        *self + 1
    }

    #[inline]
    fn nth(n: usize) -> Self {
        n as u64
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct QuantizedToken<const SIZE: usize>([u8; SIZE]);

impl<const SIZE: usize> Token<SIZE> for QuantizedToken<SIZE> {
    #[inline(always)]
    fn encode(&self) -> [u8; SIZE] {
        self.0
    }

    #[inline(always)]
    fn decode(bytes: [u8; SIZE]) -> Self {
        Self(bytes)
    }

    #[inline(always)]
    fn zero() -> Self {
        Self([0; SIZE])
    }
}

impl std::fmt::Display for QuantizedToken<1> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0[0])
    }
}

impl std::fmt::Display for QuantizedToken<2> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", u16::from_le_bytes(self.0))
    }
}

impl std::fmt::Display for QuantizedToken<3> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut bytes = [0; 4];

        bytes[..3].copy_from_slice(&self.0);

        write!(f, "{}", u32::from_le_bytes(bytes))
    }
}

impl std::fmt::Display for QuantizedToken<4> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", u32::from_le_bytes(self.0))
    }
}

impl std::fmt::Display for QuantizedToken<8> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", u64::from_le_bytes(self.0))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TokensMap<const SIZE: usize, T: Token<SIZE>> {
    /// `[token - SIZE bytes][word_len - 2 bytes][word - variable]`
    map: Box<[u8]>,

    _token: PhantomData<T>
}

impl<const SIZE: usize, T: Token<SIZE>> TokensMap<SIZE, T> {
    #[inline]
    pub fn open(map: impl Into<Box<[u8]>>) -> Self {
        Self {
            map: map.into(),
            _token: PhantomData
        }
    }

    #[inline]
    pub fn into_inner(self) -> Box<[u8]> {
        self.map
    }

    pub fn from_words<F: AsRef<[u8]>>(
        words: impl IntoIterator<Item = F>
    ) -> anyhow::Result<Self> {
        let mut unique_words = HashSet::new();
        let mut token = T::zero();
        let mut map = Vec::new();

        for word in words {
            let word = word.as_ref();

            if word.len() > 65535 {
                anyhow::bail!("BPE words must be shorter than 65536 bytes");
            }

            if unique_words.insert(word.to_vec()) {
                map.extend(token.encode());
                map.extend((word.len() as u16).to_le_bytes());
                map.extend(word);

                token = token.inc();
            }
        }

        Ok(Self {
            map: map.into_boxed_slice(),
            _token: PhantomData
        })
    }

    pub fn for_each(&self, mut callback: impl FnMut(T, Box<[u8]>)) {
        let mut i = 0;
        let n = self.map.len();

        let mut token_buf = [0; SIZE];

        while i < n {
            token_buf.copy_from_slice(&self.map[i..i + SIZE]);

            let token = T::decode(token_buf);

            let word_len = u16::from_le_bytes([
                self.map[i + SIZE], self.map[i + SIZE + 1]
            ]) as usize;

            i += SIZE + 2;

            let word = self.map[i..i + word_len].to_vec()
                .into_boxed_slice();

            callback(token, word);

            i += word_len;
        }
    }

    pub fn find_token(&self, word: impl AsRef<[u8]>) -> Option<T> {
        let mut i = 0;
        let n = self.map.len();
        let word = word.as_ref();

        let mut token_buf = [0; SIZE];

        while i < n {
            token_buf.copy_from_slice(&self.map[i..i + SIZE]);

            let token = T::decode(token_buf);

            let word_len = u16::from_le_bytes([
                self.map[i + SIZE], self.map[i + SIZE + 1]
            ]) as usize;

            i += SIZE + 2;

            let token_word = &self.map[i..i + word_len];

            if word == token_word {
                return Some(token);
            }

            i += word_len;
        }

        None
    }

    pub fn find_word(&self, token: impl Into<T>) -> Option<Box<[u8]>> {
        let mut i = 0;
        let n = self.map.len();
        let token: T = token.into();

        let mut token_buf = [0; SIZE];

        while i < n {
            token_buf.copy_from_slice(&self.map[i..i + SIZE]);

            let word_len = u16::from_le_bytes([
                self.map[i + SIZE], self.map[i + SIZE + 1]
            ]) as usize;

            i += SIZE + 2;

            if token == T::decode(token_buf) {
                let word = self.map[i..i + word_len].to_vec()
                    .into_boxed_slice();

                return Some(word);
            }

            i += word_len;
        }

        None
    }

    pub fn as_words_table(&self) -> HashMap<Box<[u8]>, T> {
        let mut tokens = HashMap::new();

        self.for_each(|token, word| {
            tokens.insert(word, token);
        });

        tokens
    }

    pub fn as_tokens_table(&self) -> HashMap<T, Box<[u8]>> {
        let mut tokens = HashMap::new();

        self.for_each(|token, word| {
            tokens.insert(token, word);
        });

        tokens
    }

    /// Amount of stored tokens.
    pub const fn len(&self) -> usize {
        let mut i = SIZE;
        let mut len = 0;

        let n = self.map.len();

        while i < n {
            let word_len = u16::from_le_bytes([
                self.map[i], self.map[i + 1]
            ]) as usize;

            i += 2 + word_len + SIZE;
            len += 1;
        }

        len
    }

    /// Size of the map in bytes.
    #[inline]
    pub const fn size(&self) -> usize {
        self.map.len()
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Length of the longest stored token.
    pub const fn max_token_len(&self) -> usize {
        let mut i = SIZE;
        let mut max_len = 0;

        let n = self.map.len();

        while i < n {
            let word_len = u16::from_le_bytes([
                self.map[i], self.map[i + 1]
            ]) as usize;

            if max_len < word_len {
                max_len = word_len;
            }

            i += 2 + word_len + SIZE;
        }

        max_len
    }
}

pub type TokensMap8 = TokensMap<1, u8>;
pub type TokensMap16 = TokensMap<2, u16>;
pub type TokensMap32 = TokensMap<4, u32>;
pub type TokensMap64 = TokensMap<8, u64>;

#[test]
fn test_tokens_map() -> anyhow::Result<()> {
    let map = TokensMap16::from_words([
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
