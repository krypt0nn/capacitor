use std::collections::HashSet;
use std::marker::PhantomData;

pub trait Token<const SIZE: usize>:
    std::fmt::Debug +
    Clone +
    Copy +
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TokensMap<const SIZE: usize, T: Token<SIZE>> {
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

    pub fn from_words<F: ToString>(words: impl IntoIterator<Item = F>) -> anyhow::Result<Self> {
        let mut unique_words = HashSet::new();
        let mut token = T::zero();
        let mut map = Vec::new();

        for word in words {
            let word = word.to_string();

            if word.len() > 255 {
                anyhow::bail!("words must be shorter than 256 bytes long");
            }

            if unique_words.insert(word.clone()) {
                map.extend_from_slice(&token.encode());
                map.push(word.len() as u8);
                map.extend_from_slice(word.as_bytes());

                token = token.inc();
            }
        }

        Ok(Self {
            map: map.into_boxed_slice(),
            _token: PhantomData
        })
    }

    pub fn find_token(&self, word: impl AsRef<str>) -> Option<T> {
        let mut i = 0;
        let n = self.map.len();
        let word = word.as_ref();

        let mut token_buf = [0; SIZE];

        while i < n {
            token_buf.copy_from_slice(&self.map[i..i + SIZE]);

            let token = T::decode(token_buf);
            let word_len = self.map[i + SIZE] as usize;

            i += SIZE + 1;

            let token_word = &self.map[i..i + word_len];

            if word.as_bytes() == token_word {
                return Some(token);
            }

            i += word_len;
        }

        None
    }

    pub fn find_word(&self, token: impl Into<T>) -> Option<String> {
        let mut i = 0;
        let n = self.map.len();
        let token: T = token.into();

        let mut token_buf = [0; SIZE];

        while i < n {
            token_buf.copy_from_slice(&self.map[i..i + SIZE]);

            let word_len = self.map[i + SIZE] as usize;

            i += SIZE + 1;

            if token == T::decode(token_buf) {
                return Some(String::from_utf8_lossy(&self.map[i..i + word_len]).to_string());
            }

            i += word_len;
        }

        None
    }

    /// Amount of stored tokens.
    pub fn len(&self) -> usize {
        let mut i = SIZE;
        let n = self.map.len();
        let mut len = 0;

        while i < n {
            i += self.map[i] as usize + SIZE + 1;
            len += 1;
        }

        len
    }

    /// Size of the map in bytes.
    #[inline]
    pub fn size(&self) -> usize {
        self.map.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
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
    assert_eq!(map.size(), 16);
    assert_eq!(map.find_word(hello_token).as_deref(), Some("hello"));
    assert_eq!(map.find_word(world_token).as_deref(), Some("world"));

    assert!(map.find_token("amogus").is_none());
    assert!(map.find_word(42_u16).is_none());

    Ok(())
}
