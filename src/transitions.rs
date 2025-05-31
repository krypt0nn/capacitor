use std::collections::HashSet;
use std::marker::PhantomData;
use std::cmp::Ordering;

use crate::tokens::Token;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Transition<const SIZE: usize, T: Token<SIZE>> {
    pub from: Box<[T]>,
    pub to: Box<[T]>,
    pub weight: u16
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TransitionsMap<const SIZE: usize, T: Token<SIZE>> {
    map: Box<[u8]>,
    from_count: usize,
    to_count: usize,
    record_size: usize,
    _token: PhantomData<T>
}

impl<const SIZE: usize, T: Token<SIZE>> TransitionsMap<SIZE, T> {
    pub fn open(map: impl Into<Box<[u8]>>) -> anyhow::Result<Self> {
        let map: Box<[u8]> = map.into();

        if map.len() < 2 {
            anyhow::bail!("transitions map can't be shorter than 2 bytes");
        }

        let from_count = map[0] as usize;
        let to_count = map[1] as usize;

        let record_size = from_count * SIZE + to_count * SIZE + 2;

        if (map.len() - 2) % record_size != 0 {
            anyhow::bail!("invalid transitions map layout");
        }

        Ok(Self {
            map,
            from_count,
            to_count,
            record_size,
            _token: PhantomData
        })
    }

    #[inline]
    pub fn into_inner(self) -> Box<[u8]> {
        self.map
    }

    pub fn from_transitions<'tokens>(
        transitions: impl IntoIterator<Item = (&'tokens [T], &'tokens [T], u16)>
    ) -> anyhow::Result<Self>
    where T: 'tokens
    {
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
            a.0.cmp(b.0)
        });

        let mut map = Vec::with_capacity(2 + (SIZE * from_tokens + SIZE * to_tokens + 2) * transitions.len());

        map.push(from_tokens as u8);
        map.push(to_tokens as u8);

        for (from, to, frequency) in transitions.drain(..) {
            for token in from {
                map.extend_from_slice(&token.encode());
            }

            for token in to {
                map.extend_from_slice(&token.encode());
            }

            map.extend_from_slice(&frequency.to_le_bytes());
        }

        Self::open(map)
    }

    /// Amount of transitions stored in the map.
    #[inline]
    pub fn len(&self) -> usize {
        (self.map.len() - 2) / (self.from_count * SIZE + self.to_count * SIZE + 2)
    }

    /// Amount of bytes stored in the transitions map.
    #[inline]
    pub fn size(&self) -> usize {
        self.map.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.len() < 3
    }

    fn read_transition(&self, idx: usize) -> Transition<SIZE, T> {
        let offset = 2 + self.record_size * idx;

        let mut from_tokens = Vec::with_capacity(self.from_count);
        let mut to_tokens = Vec::with_capacity(self.to_count);

        let mut token = [0; SIZE];
        let mut i = 0;

        while i < self.from_count {
            token.copy_from_slice(&self.map[offset + SIZE * i..offset + SIZE * (i + 1)]);

            from_tokens.push(T::decode(token));

            i += 1;
        }

        while i < self.from_count + self.to_count {
            token.copy_from_slice(&self.map[offset + SIZE * i..offset + SIZE * (i + 1)]);

            to_tokens.push(T::decode(token));

            i += 1;
        }

        let weight = u16::from_le_bytes([
            self.map[offset + SIZE * i],
            self.map[offset + SIZE * i + 1]
        ]);

        Transition {
            from: from_tokens.into_boxed_slice(),
            to: to_tokens.into_boxed_slice(),
            weight
        }
    }

    /// Read all the transitions from the map and return a list of them.
    pub fn read_list(&self) -> Box<[Transition<SIZE, T>]> {
        (0..self.len()).map(|i| self.read_transition(i)).collect()
    }

    /// Use provided comparator to perform binary search over stored transitions.
    ///
    /// Result of the search is a *list* of values. Comparator is allowed to
    /// return a continued series of equal values, e.g. return `Equal` for
    /// numbers 2, 3 from sequence 1, 2, 3, 4, 5.
    pub fn binary_search(
        &self,
        mut comparator: impl FnMut(&Transition<SIZE, T>) -> Ordering
    ) -> HashSet<Transition<SIZE, T>> {
        let mut matched = HashSet::new();

        let mut left_idx = 0;
        let mut right_idx = self.len() - 1;

        let mut prev_left_idx = left_idx;
        let mut prev_right_idx = right_idx;

        while left_idx <= right_idx {
            let middle_idx = (left_idx + right_idx).div_ceil(2);

            dbg!(left_idx, right_idx, middle_idx);

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
    pub fn find_transitions(&self, from: impl AsRef<[T]>) -> HashSet<Transition<SIZE, T>> {
        let from_count = self.from_count;
        let from = from.as_ref();

        let match_len = from.len().min(self.from_count);

        let from = &from[from.len() - match_len..];

        self.binary_search(|transition| {
            let transition_from = &transition.from[from_count - match_len..];

            dbg!(&from);
            dbg!(&transition_from);
            dbg!(transition_from.cmp(from));

            transition_from.cmp(from)
        })
    }
}

pub type TransitionsMap8 = TransitionsMap<1, u8>;
pub type TransitionsMap16 = TransitionsMap<2, u16>;
pub type TransitionsMap32 = TransitionsMap<4, u32>;
pub type TransitionsMap64 = TransitionsMap<8, u64>;

#[test]
fn test_transitions_map() -> anyhow::Result<()> {
    let transitions = TransitionsMap16::from_transitions([
        ([1, 2].as_slice(), [3, 4].as_slice(), u16::MAX / 3),
        ([2, 3].as_slice(), [4, 5].as_slice(), u16::MAX / 3),
        ([3, 4].as_slice(), [5, 1].as_slice(), u16::MAX / 3)
    ])?;

    let list = transitions.read_list();

    let transitions = TransitionsMap16::open(transitions.into_inner())?;

    assert_eq!(&transitions.read_list(), &list);

    assert_eq!(list.len(), 3);

    assert_eq!(list[0], Transition {
        from: vec![1, 2].into_boxed_slice(),
        to: vec![3, 4].into_boxed_slice(),
        weight: u16::MAX / 3
    });

    assert_eq!(list[1], Transition {
        from: vec![2, 3].into_boxed_slice(),
        to: vec![4, 5].into_boxed_slice(),
        weight: u16::MAX / 3
    });

    assert_eq!(list[2], Transition {
        from: vec![3, 4].into_boxed_slice(),
        to: vec![5, 1].into_boxed_slice(),
        weight: u16::MAX / 3
    });

    assert_eq!(transitions.find_transitions([2]), HashSet::from_iter([list[0].clone()]));
    assert_eq!(transitions.find_transitions([1, 2]), HashSet::from_iter([list[0].clone()]));

    assert_eq!(transitions.find_transitions([3]), HashSet::from_iter([list[1].clone()]));
    assert_eq!(transitions.find_transitions([2, 3]), HashSet::from_iter([list[1].clone()]));

    assert_eq!(transitions.find_transitions([4]), HashSet::from_iter([list[2].clone()]));
    assert_eq!(transitions.find_transitions([3, 4]), HashSet::from_iter([list[2].clone()]));

    let transitions = TransitionsMap16::from_transitions([
        ([1, 0].as_slice(), [1].as_slice(), u16::MAX / 5),
        ([2, 0].as_slice(), [2].as_slice(), u16::MAX / 5),
        ([3, 1].as_slice(), [3].as_slice(), u16::MAX / 5),
        ([4, 1].as_slice(), [4].as_slice(), u16::MAX / 5),
        ([5, 1].as_slice(), [5].as_slice(), u16::MAX / 5)
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
