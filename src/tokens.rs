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

use compact_str::{CompactString, ToCompactString};

/// Split text into pre-token items: single characters or whole `<|tag|>`
/// special tags. Under alphanumeric-only tokenizer every non-alphanumeric
/// character (punctuation or whitespace) acts as a word separator: runs of
/// such characters collapse into a single space instead of being deleted.
///
/// Shared by model building and inference tokenization — these two must never
/// drift apart.
pub fn pre_tokenize(
    text: impl AsRef<str>,
    make_lowercase: bool,
    force_alphanumeric: bool
) -> Vec<CompactString> {
    fn push_char(out: &mut Vec<CompactString>, c: char, make_lowercase: bool) {
        out.push(if make_lowercase {
            c.to_lowercase().collect::<CompactString>()
        } else {
            c.to_compact_string()
        });
    }

    let chars = text.as_ref()
        .chars()
        .collect::<Box<[char]>>();

    let n = chars.len();
    let mut out = Vec::with_capacity(n);

    let mut last_space = true;
    let mut i = 0;

    while i < n {
        // Preserve special tags as separate items.
        if chars[i] == '<' && i + 1 < n && chars[i + 1] == '|' {
            let mut j = i + 2;
            let mut found = false;

            while j < n && (j - i) < 256 {
                if chars[j] == '>' && chars[j - 1] == '|' {
                    found = true;

                    break;
                }

                j += 1;
            }

            // If we found <|token|>, then store it. Otherwise process
            // < symbol as a regular character.
            if found {
                out.push(chars[i..=j].iter().collect::<CompactString>());

                last_space = true;

                i = j + 1;

                continue;
            }
        }

        let c = chars[i];

        if force_alphanumeric {
            if c.is_alphanumeric() {
                push_char(&mut out, c, make_lowercase);

                last_space = false;
            } else if !last_space {
                out.push(CompactString::from(" "));

                last_space = true;
            }
        } else {
            push_char(&mut out, c, make_lowercase);
        }

        i += 1;
    }

    if out.last().is_some_and(|item| item.as_str() == " ") {
        out.pop();
    }

    out
}

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
        let words = words.into_iter();
        let (capacity, _) = words.size_hint();

        let mut unique_words = HashSet::with_capacity(capacity);
        let mut deduped = Vec::with_capacity(capacity);

        for word in words {
            let word = word.as_ref();

            if word.len() > 65535 {
                anyhow::bail!("BPE words must be shorter than 65536 bytes");
            }

            if unique_words.insert(word.to_vec()) {
                deduped.push(word.to_vec());
            }
        }

        // Store the longest tokens first - makes word lookups stop early once
        // candidates become shorter than the searched word, and makes
        // max_token_len trivial.
        #[allow(clippy::unnecessary_sort_by)]
        deduped.sort_by(|a, b| b.len().cmp(&a.len()));

        let mut map = Vec::with_capacity(
            deduped.len() * deduped.get(deduped.len() / 2)
                .map(|word| word.len())
                .unwrap_or(0)
        );

        for (token, word) in deduped.into_iter().enumerate() {
            map.extend((token as u16).to_le_bytes());
            map.extend((word.len() as u16).to_le_bytes());
            map.extend(word);
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

            // Words are stored in descending length order - all the following
            // words are shorter than the current one - and thus shorter than
            // the searched word too, so there's nothing left to match.

            if word_len < word.len() {
                return None;
            }

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
        // Approximate amount of stored tokens.
        let mut tokens = HashMap::with_capacity(self.0.len() / 10);

        self.for_each(|token, word| {
            tokens.insert(word, token);
        });

        tokens
    }

    pub fn as_tokens_table(&self) -> HashMap<u16, Box<[u8]>> {
        // Approximate amount of stored tokens.
        let mut tokens = HashMap::with_capacity(self.0.len() / 10);

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
    ///
    /// Words are stored in descending length order, so it's just the first
    /// record's length.
    pub fn max_token_len(&self) -> usize {
        if self.is_empty() {
            return 0;
        }

        u16::from_le_bytes([self.0[2], self.0[3]]) as usize
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

    // Words are stored in descending length order, so token ids follow it.

    let map = TokensMap::from_words([
        "a",
        "longest",
        "bc"
    ])?;

    assert_eq!(map.max_token_len(), 7);
    assert_eq!(map.find_word(0).as_deref(), Some(b"longest".as_slice()));
    assert_eq!(map.find_word(1).as_deref(), Some(b"bc".as_slice()));
    assert_eq!(map.find_word(2).as_deref(), Some(b"a".as_slice()));
    assert_eq!(map.find_token("longest"), Some(0));

    // Truncated longest word must not be found through any suffix.

    assert_eq!(map.find_token("longes"), None);

    // Single letters exist in the map as their own tokens.

    assert_eq!(map.find_token("a"), Some(2));

    Ok(())
}

#[test]
fn test_pre_tokenize() {
    assert_eq!(
        pre_tokenize("Hello,  world!", true, true),
        ["h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"]
    );

    // Punctuation and underscores act as separators, not glue.
    assert_eq!(
        pre_tokenize("how_is_wayland_an_upgrade?", true, true),
        [
            "h", "o", "w", " ", "i", "s", " ", "w", "a", "y", "l", "a", "n",
            "d", " ", "a", "n", " ", "u", "p", "g", "r", "a", "d", "e"
        ]
    );

    // Special tags are preserved as single items.
    assert_eq!(
        pre_tokenize("a<|tag|>b", false, true),
        ["a", "<|tag|>", "b"]
    );

    // Without force_alphanumeric everything is kept as-is.
    assert_eq!(
        pre_tokenize("a,  b", true, false),
        ["a", ",", " ", " ", "b"]
    );
}
