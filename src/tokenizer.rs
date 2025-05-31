use std::io::{Read, Write};
use std::iter::FusedIterator;

pub trait Tokenizer {
    type Encoder<'reader>: Iterator<Item = anyhow::Result<String>> + FusedIterator;
    type Decoder<'tokens>: Read;

    fn encode<'reader>(&self, reader: impl Read + 'reader) -> Self::Encoder<'reader>;
    fn decode<'tokens>(&self, tokens: impl IntoIterator<Item = String> + 'tokens) -> Self::Decoder<'tokens>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WordTokenizer {
    /// Convert text to lowercase.
    pub lowercase: bool,

    /// Keep text punctuation.
    pub punctuation: bool
}

impl Default for WordTokenizer {
    fn default() -> Self {
        Self {
            lowercase: false,
            punctuation: true
        }
    }
}

impl Tokenizer for WordTokenizer {
    type Encoder<'reader> = WordTokenizerEncoder<'reader>;
    type Decoder<'tokens> = WordTokenizerDecoder<'tokens>;

    fn encode<'reader>(&self, reader: impl Read + 'reader) -> Self::Encoder<'reader> {
        WordTokenizerEncoder {
            lowercase: self.lowercase,
            punctuation: self.punctuation,
            reader: Box::new(reader),
            finished: false,
            buf: Vec::with_capacity(1024)
        }
    }

    fn decode<'tokens>(&self, tokens: impl IntoIterator<Item = String> + 'tokens) -> Self::Decoder<'tokens> {
        WordTokenizerDecoder {
            tokens: Box::new(tokens.into_iter()),
            next: None,
            buf: Vec::with_capacity(1024)
        }
    }
}

pub struct WordTokenizerEncoder<'reader> {
    lowercase: bool,
    punctuation: bool,
    reader: Box<dyn Read + 'reader>,
    finished: bool,
    buf: Vec<u8>
}

impl Iterator for WordTokenizerEncoder<'_> {
    type Item = anyhow::Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut buf = [0; 1024];

        fn apply(mut word: &str, lowercase: bool, punctuation: bool) -> String {
            word = word.trim();

            if !punctuation {
                word = word.trim_matches(|c: char| c.is_ascii_punctuation());
            }

            if lowercase {
                word.to_lowercase()
            } else {
                word.to_string()
            }
        }

        loop {
            let mut i = 0;
            let n = self.buf.len();

            while i < n {
                if self.buf[i].is_ascii_whitespace() {
                    i += 1;
                } else {
                    break;
                }
            }

            while i < n {
                if self.buf[i].is_ascii_whitespace() {
                    let word = String::from_utf8_lossy(&self.buf[..i])
                        .to_string();

                    self.buf = self.buf[i..].to_vec();

                    return Some(Ok(apply(&word, self.lowercase, self.punctuation)));
                }

                i += 1;
            }

            if self.finished {
                if n > 0 {
                    let word = String::from_utf8_lossy(&self.buf)
                        .trim()
                        .to_string();

                    self.buf.clear();

                    return Some(Ok(apply(&word, self.lowercase, self.punctuation)));
                }

                else {
                    return None;
                }
            }

            match self.reader.read(&mut buf) {
                Ok(n) => {
                    if n > 0 {
                        self.buf.extend_from_slice(&buf[..n]);
                    } else {
                        self.finished = true;
                    }
                }

                Err(err) => return Some(Err(anyhow::anyhow!(err)))
            }
        }
    }
}

impl FusedIterator for WordTokenizerEncoder<'_> {}

pub struct WordTokenizerDecoder<'tokens> {
    tokens: Box<dyn Iterator<Item = String> + 'tokens>,
    next: Option<String>,
    buf: Vec<u8>
}

impl Read for WordTokenizerDecoder<'_> {
    fn read(&mut self, mut buf: &mut [u8]) -> std::io::Result<usize> {
        if self.next.is_none() {
            self.next = self.tokens.next();
        }

        if let Some(word) = self.next.take() {
            self.buf.extend(word.as_bytes());

            self.next = self.tokens.next();

            if self.next.is_some() {
                self.buf.push(b' ');
            }
        }

        if self.buf.is_empty() {
            return Ok(0);
        }

        let n = buf.write(&self.buf)?;

        self.buf = self.buf[n..].to_vec();

        Ok(n)
    }
}

#[allow(clippy::unbuffered_bytes)]
#[test]
fn test_word_tokenizer() -> anyhow::Result<()> {
    let tokenizer = WordTokenizer {
        lowercase: true,
        punctuation: false
    };

    assert_eq!(tokenizer.encode(std::io::Cursor::new(b"Hello,   World!")).collect::<anyhow::Result<Vec<String>>>()?, ["hello", "world"]);
    assert_eq!(tokenizer.decode([String::from("hello"), String::from("world")]).bytes().collect::<std::io::Result<Vec<u8>>>()?, b"hello world");

    Ok(())
}
