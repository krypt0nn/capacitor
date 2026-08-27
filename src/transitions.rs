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

use std::collections::HashSet;
use std::cmp::Ordering;

use rayon::prelude::*;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Transition {
    pub from: Box<[u16]>,
    pub to: Box<[u16]>,
    pub weight: u32
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TransitionsMap {
    map: Box<[u8]>,
    from_count: usize,
    to_count: usize,
    record_size: usize
}

impl TransitionsMap {
    pub fn open(map: impl Into<Box<[u8]>>) -> anyhow::Result<Self> {
        let map: Box<[u8]> = map.into();

        if map.len() < 2 {
            anyhow::bail!("transitions map cannot be shorter than 2 bytes");
        }

        let from_count = map[0] as usize;
        let to_count = map[1] as usize;

        let record_size = from_count * 2 + to_count * 2 + 4;

        if !(map.len() - 2).is_multiple_of(record_size) {
            anyhow::bail!("invalid transitions map layout");
        }

        Ok(Self {
            map,
            from_count,
            to_count,
            record_size
        })
    }

    #[inline]
    pub fn into_inner(self) -> Box<[u8]> {
        self.map
    }

    pub fn from_transitions<'tokens>(
        transitions: impl IntoIterator<Item = (&'tokens [u16], &'tokens [u16], u32)>
    ) -> anyhow::Result<Self> {
        let mut transitions = transitions.into_iter().collect::<Vec<_>>();

        if transitions.is_empty() {
            anyhow::bail!("at least 1 transition required");
        }

        let from_tokens = transitions[0].0.len();
        let to_tokens = transitions[0].1.len();

        if !(1..=255).contains(&from_tokens) {
            anyhow::bail!("input n-grams must be greater than 0 and lower than 256");
        }

        if !(1..=255).contains(&to_tokens) {
            anyhow::bail!("output n-grams must be greater than 0 and lower than 256");
        }

        transitions.sort_by(|a, b| {
            match a.0.cmp(b.0) {
                Ordering::Equal => a.1.cmp(b.1),
                ord => ord
            }
        });

        let record_size = from_tokens * 2 + to_tokens * 2 + 4;

        let mut map = Vec::with_capacity(2 + record_size * transitions.len());

        map.push(from_tokens as u8);
        map.push(to_tokens as u8);

        for (from, to, weight) in transitions.drain(..) {
            for token in from.iter().chain(to.iter()) {
                map.extend(token.to_le_bytes());
            }

            map.extend(weight.to_le_bytes());
        }

        Self::open(map)
    }

    /// Amount of tokens in each `from` n-gram.
    #[inline(always)]
    pub const fn from_count(&self) -> usize {
        self.from_count
    }

    /// Amount of tokens in each `to` n-gram.
    #[inline(always)]
    pub const fn to_count(&self) -> usize {
        self.to_count
    }

    /// Amount of transitions stored in the map.
    #[inline]
    pub const fn len(&self) -> usize {
        (self.map.len() - 2) / self.record_size
    }

    /// Amount of bytes stored in the transitions map.
    #[inline(always)]
    pub const fn size(&self) -> usize {
        self.map.len()
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.map.len() < 4
    }

    fn read_transition(&self, idx: usize) -> Transition {
        let offset = 2 + self.record_size * idx;

        let mut from_tokens = Vec::with_capacity(self.from_count);
        let mut to_tokens = Vec::with_capacity(self.to_count);

        let mut i = 0;

        while i < self.from_count {
            let token = u16::from_le_bytes([
                self.map[offset + i * 2], self.map[offset + i * 2 + 1]
            ]);

            from_tokens.push(token);

            i += 1;
        }

        while i < self.from_count + self.to_count {
            let token = u16::from_le_bytes([
                self.map[offset + i * 2], self.map[offset + i * 2 + 1]
            ]);

            to_tokens.push(token);

            i += 1;
        }

        let weight = u32::from_le_bytes([
            self.map[offset + i * 2    ], self.map[offset + i * 2 + 1],
            self.map[offset + i * 2 + 2], self.map[offset + i * 2 + 3]
        ]);

        Transition {
            from: from_tokens.into_boxed_slice(),
            to: to_tokens.into_boxed_slice(),
            weight
        }
    }

    /// Read all the transitions from the map and return a list of them.
    pub fn read_list(&self) -> Box<[Transition]> {
        (0..self.len()).map(|i| self.read_transition(i)).collect()
    }

    /// Use provided comparator to perform binary search over stored transitions.
    ///
    /// Result of the search is a *list* of values. Comparator is allowed to
    /// return a continued series of equal values, e.g. return `Equal` for
    /// numbers 2, 3 from sequence 1, 2, 3, 4, 5.
    pub fn binary_search(
        &self,
        mut comparator: impl FnMut(&Transition) -> Ordering
    ) -> HashSet<Transition> {
        let mut matched = HashSet::new();

        let mut left_idx = 0;
        let mut right_idx = self.len() - 1;

        let mut prev_left_idx = left_idx;
        let mut prev_right_idx = right_idx;

        while left_idx <= right_idx {
            let middle_idx = (left_idx + right_idx).div_ceil(2);

            let transition = self.read_transition(middle_idx);

            match comparator(&transition) {
                Ordering::Equal => {
                    matched.insert(transition);

                    let mut i = middle_idx;

                    if middle_idx > 0 {
                        i -= 1;

                        while i >= left_idx {
                            let transition = self.read_transition(i);

                            if comparator(&transition) != Ordering::Equal {
                                break;
                            }

                            matched.insert(transition);

                            if i == 0 {
                                break;
                            }

                            i -= 1;
                        }
                    }

                    i = middle_idx + 1;

                    while i <= right_idx {
                        let transition = self.read_transition(i);

                        if comparator(&transition) != Ordering::Equal {
                            break;
                        }

                        matched.insert(transition);

                        i += 1;
                    }
                }

                Ordering::Less if middle_idx == left_idx => left_idx = middle_idx + 1,
                Ordering::Greater if middle_idx == right_idx => right_idx = middle_idx - 1,

                Ordering::Less => left_idx = middle_idx,
                Ordering::Greater => right_idx = middle_idx
            }

            if left_idx == prev_left_idx && right_idx == prev_right_idx {
                break;
            }

            prev_left_idx = left_idx;
            prev_right_idx = right_idx;
        }

        matched
    }

    /// Use binary search to find transitions with the given `from` suffix.
    ///
    /// The provided suffix can be shorter than what is stored in the map, and
    /// will be truncated if it's longer than needed.
    pub fn find_transitions(
        &self,
        from: impl AsRef<[u16]>
    ) -> HashSet<Transition> {
        let from_count = self.from_count;
        let from = from.as_ref();

        let match_len = from.len().min(self.from_count);

        let from = &from[from.len() - match_len..];

        if self.from_count == match_len {
            self.binary_search(|transition| {
                transition.from.as_ref().cmp(from)
            })
        } else {
            (0..self.len())
                .into_par_iter()
                .map(|i| self.read_transition(i))
                .filter(|transition| {
                    &transition.from[from_count - match_len..] == from
                })
                .collect()
        }
    }
}

#[test]
fn test_transitions_map() -> anyhow::Result<()> {
    let transitions = TransitionsMap::from_transitions([
        ([1, 2].as_slice(), [3, 4].as_slice(), u32::MAX / 3),
        ([2, 3].as_slice(), [4, 5].as_slice(), u32::MAX / 3),
        ([3, 4].as_slice(), [5, 1].as_slice(), u32::MAX / 3)
    ])?;

    let list = transitions.read_list();

    let transitions = TransitionsMap::open(transitions.into_inner())?;

    assert_eq!(&transitions.read_list(), &list);

    assert_eq!(list.len(), 3);

    assert_eq!(list[0], Transition {
        from: vec![1, 2].into_boxed_slice(),
        to: vec![3, 4].into_boxed_slice(),
        weight: u32::MAX / 3
    });

    assert_eq!(list[1], Transition {
        from: vec![2, 3].into_boxed_slice(),
        to: vec![4, 5].into_boxed_slice(),
        weight: u32::MAX / 3
    });

    assert_eq!(list[2], Transition {
        from: vec![3, 4].into_boxed_slice(),
        to: vec![5, 1].into_boxed_slice(),
        weight: u32::MAX / 3
    });

    assert_eq!(transitions.find_transitions([2]), HashSet::from_iter([list[0].clone()]));
    assert_eq!(transitions.find_transitions([1, 2]), HashSet::from_iter([list[0].clone()]));

    assert_eq!(transitions.find_transitions([3]), HashSet::from_iter([list[1].clone()]));
    assert_eq!(transitions.find_transitions([2, 3]), HashSet::from_iter([list[1].clone()]));

    assert_eq!(transitions.find_transitions([4]), HashSet::from_iter([list[2].clone()]));
    assert_eq!(transitions.find_transitions([3, 4]), HashSet::from_iter([list[2].clone()]));

    let transitions = TransitionsMap::from_transitions([
        ([1, 0].as_slice(), [1].as_slice(), u32::MAX / 5),
        ([2, 0].as_slice(), [2].as_slice(), u32::MAX / 5),
        ([3, 1].as_slice(), [3].as_slice(), u32::MAX / 5),
        ([4, 1].as_slice(), [4].as_slice(), u32::MAX / 5),
        ([5, 1].as_slice(), [5].as_slice(), u32::MAX / 5)
    ])?;

    let list = transitions.read_list();

    assert_eq!(transitions.read_transition(0), list[0]);
    assert_eq!(transitions.read_transition(1), list[1]);
    assert_eq!(transitions.read_transition(2), list[2]);
    assert_eq!(transitions.read_transition(3), list[3]);
    assert_eq!(transitions.read_transition(4), list[4]);

    assert_eq!(transitions.find_transitions([1, 0]), HashSet::from_iter([list[0].clone()]));
    assert_eq!(transitions.find_transitions([2, 0]), HashSet::from_iter([list[1].clone()]));
    assert_eq!(transitions.find_transitions([3, 1]), HashSet::from_iter([list[2].clone()]));
    assert_eq!(transitions.find_transitions([4, 1]), HashSet::from_iter([list[3].clone()]));
    assert_eq!(transitions.find_transitions([5, 1]), HashSet::from_iter([list[4].clone()]));

    assert_eq!(transitions.find_transitions([0]), HashSet::from_iter([
        list[0].clone(),
        list[1].clone()
    ]));

    assert_eq!(transitions.find_transitions([1]), HashSet::from_iter([
        list[2].clone(),
        list[3].clone(),
        list[4].clone()
    ]));

    Ok(())
}
