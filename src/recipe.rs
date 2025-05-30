use std::str::FromStr;
use std::path::PathBuf;

use anyhow::Context;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct File {
    pub path: PathBuf,
    pub delimiter: String
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tokenizer {
    WordTokenizer {
        lowercase: bool,
        punctuation: bool
    }
}

impl std::fmt::Display for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WordTokenizer { lowercase: false, punctuation: false } => f.write_str("word-tokenizer"),
            Self::WordTokenizer { lowercase: false, punctuation: true } => f.write_str("word-tokenizer[punctuation]"),
            Self::WordTokenizer { lowercase: true, punctuation: false } => f.write_str("word-tokenizer[lowercase]"),
            Self::WordTokenizer { lowercase: true, punctuation: true } => f.write_str("word-tokenizer[lowercase,punctuation]")
        }
    }
}

impl FromStr for Tokenizer {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Some(params) = s.strip_prefix("word-tokenizer") {
            let mut lowercase = false;
            let mut punctuation = false;

            for param in params.trim_matches(['[', ']']).split(',') {
                match param {
                    "lowercase" => lowercase = true,
                    "punctuation" => punctuation = true,

                    _ => anyhow::bail!("unknown word tokenizer parameter: {param}")
                }
            }

            Ok(Self::WordTokenizer {
                lowercase,
                punctuation
            })
        }

        else {
            anyhow::bail!("unknown tokenizer: {s}");
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Recipe {
    pub name: String,
    pub files: Vec<File>,
    pub tokenizer: Tokenizer,
    pub total_experts: usize,
    pub active_experts: usize
}

impl std::fmt::Display for Recipe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let files = self.files.iter()
            .map(|file| {
                if file.delimiter.is_empty() {
                    format!("File {}", file.path.display())
                } else {
                    format!("Split {} File {}", file.delimiter, file.path.display())
                }
            })
            .collect::<Vec<String>>();

        write!(
            f,
            "Name {}\nTokenizer {}\nExperts {}/{}\n\n{}",
            self.name,
            self.tokenizer,
            self.active_experts,
            self.total_experts,
            files.join("\n")
        )
    }
}

impl FromStr for Recipe {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut name = None;
        let mut tokenizer = None;
        let mut files = Vec::new();
        let mut total_experts = 0;
        let mut active_experts = 0;

        for line in s.lines() {
            if line.is_empty() {
                continue;
            }

            else if let Some(value) = line.strip_prefix("Name ") {
                name = Some(value.trim().to_string());
            }

            else if let Some(value) = line.strip_prefix("Tokenizer ") {
                tokenizer = Some(Tokenizer::from_str(value.trim())?);
            }

            else if let Some(value) = line.strip_prefix("File ") {
                files.push(File {
                    path: PathBuf::from(value.trim()),
                    delimiter: String::from("<|document|>")
                });
            }

            else if let Some(value) = line.strip_prefix("Split ") {
                let Some((delimiter, path)) = value.split_once(" File ") else {
                    anyhow::bail!("invalid split file parameter: {line}");
                };

                files.push(File {
                    path: PathBuf::from(path.trim()),
                    delimiter: delimiter.trim().to_string()
                });
            }

            else if let Some(value) = line.strip_prefix("Experts ") {
                let Some((active, total)) = value.split_once("/") else {
                    anyhow::bail!("invalid experts parameter value: {line}");
                };

                active_experts = active.parse()
                    .with_context(|| format!("invalid active experts format: {line}"))?;

                total_experts = total.parse()
                    .with_context(|| format!("invalid total experts format: {line}"))?;
            }

            else {
                anyhow::bail!("unknown model parameter: {line}");
            }
        }

        Ok(Self {
            name: name.ok_or_else(|| anyhow::anyhow!("missing model name"))?,
            files,
            tokenizer: tokenizer.ok_or_else(|| anyhow::anyhow!("missing model tokenizer"))?,
            total_experts,
            active_experts
        })
    }
}

#[test]
fn test_recipe() -> anyhow::Result<()> {
    let recipe = Recipe {
        name: String::from("test recipe"),
        files: vec![
            File {
                path: PathBuf::from("test"),
                delimiter: String::from("</test>")
            }
        ],
        tokenizer: Tokenizer::WordTokenizer {
            lowercase: true,
            punctuation: false
        },
        total_experts: 64,
        active_experts: 4
    };

    assert_eq!(Recipe::from_str(&recipe.to_string())?, recipe);

    Ok(())
}
