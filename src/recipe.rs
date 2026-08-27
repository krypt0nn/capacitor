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

use std::collections::HashMap;
use std::str::FromStr;
use std::path::{Path, PathBuf};

use anyhow::Context;

use crate::model::Model;

fn parse_num(value: &str) -> Option<usize> {
    if let Some(value) = value.strip_suffix("g") {
        value.parse::<f32>()
            .map(|value| (value * 1024.0 * 1024.0 * 1024.0).ceil() as usize)
            .ok()
    }

    else if let Some(value) = value.strip_suffix("m") {
        value.parse::<f32>()
            .map(|value| (value * 1024.0 * 1024.0).ceil() as usize)
            .ok()
    }

    else if let Some(value) = value.strip_suffix("k") {
        value.parse::<f32>()
            .map(|value| (value * 1024.0).ceil() as usize)
            .ok()
    }

    else {
        value.parse::<usize>().ok()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct File {
    /// Path to the dataset file.
    pub path: PathBuf,

    /// Documents delimiter.
    pub delimiter: String,

    /// Shuffle documents within the file.
    pub shuffle: bool
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tokenizer {
    /// Convert text characters to lowercase.
    ///
    /// Default: `false`
    pub make_lowercase: bool,

    /// Keep only alpha-numeric characters.
    ///
    /// Default: `false`
    pub force_alphanumeric: bool,

    /// How many tokens BPE tokenizer should learn.
    ///
    /// Default: `1024`
    pub num_tokens: u16,

    /// How many pre-tokenized words (text letters, roughly equivalent to bytes)
    /// will be used to train the BPE tokenizer. Higher value will produce
    /// better quality tokens in cost of increased build time.
    ///
    /// Default: `256M; 268435456`
    pub num_samples: usize
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self {
            make_lowercase: false,
            force_alphanumeric: false,
            num_tokens: 1024,
            num_samples: 256 * 1024 * 1024
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ngrams {
    /// Number of `from` tokens in `[from] -> [to]` transition (from-depth).
    /// Higher value means a model will predict continuation tokens more
    /// precisely.
    ///
    /// Default: `2`
    pub num_from: usize,

    /// Number of `to` tokens in `[from] -> [to]` transition (to-depth). Higher
    /// value means a model will predict more continuation tokens more
    /// precisely.
    ///
    /// Default: `1`
    pub num_to: usize
}

impl Default for Ngrams {
    fn default() -> Self {
        Self {
            num_from: 2,
            num_to: 1
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Experts {
    /// Amount of experts used at the same time.
    ///
    /// Default: `0` (only shared expert)
    pub num_active: usize,

    /// Total amount of experts to train.
    ///
    /// Default: `0` (only shared expert)
    pub num_total: usize,

    /// Amount of centroids (randomly sampled documents) used per-expert on
    /// model build time to determine its profile.
    ///
    /// Default: `4`
    pub num_centroids: usize
}

impl Default for Experts {
    fn default() -> Self {
        Self {
            num_active: 0,
            num_total: 0,
            num_centroids: 4
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recipe {
    /// Metadata key-values table.
    pub keys: HashMap<String, String>,

    /// Documents tokenizer.
    pub tokenizer: Tokenizer,

    /// Model ngrams.
    pub ngrams: Ngrams,

    /// Model experts.
    pub experts: Experts,

    /// Formatting rule for the user's queries.
    ///
    /// | Pattern           | Description                      |
    /// | ----------------- | -------------------------------- |
    /// | `{{content}}`     | User message (generation prefix) |
    /// | `{{start_token}}` | Model document start token       |
    /// | `{{stop_token}}`  | Model document stop token        |
    ///
    /// By default `{{content}}` template is used.
    pub template: String,

    /// Stop words after which the inference must be killed. Stop words are
    /// not returned to the user.
    pub stop_tokens: Vec<String>,

    /// Dataset files.
    pub files: Vec<File>
}

impl Recipe {
    pub fn relative_to(mut self, folder: impl AsRef<Path>) -> Self {
        for file in &mut self.files {
            file.path = folder.as_ref().join(&file.path);
        }

        self
    }
}

impl Default for Recipe {
    fn default() -> Self {
        Self {
            keys: HashMap::new(),
            tokenizer: Tokenizer::default(),
            ngrams: Ngrams::default(),
            experts: Experts::default(),
            template: String::from("{{content}}"),
            stop_tokens: vec![
                Model::START_TOKEN.to_string(),
                Model::STOP_TOKEN.to_string()
            ],
            files: Vec::new()
        }
    }
}

impl std::fmt::Display for Recipe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stop_tokens = self.stop_tokens.iter()
            .map(|token| format!("Stop {token}"))
            .collect::<Vec<String>>();

        let keys = self.keys.iter()
            .filter(|(key, _)| {
                // Special metadata keys.
                ![
                    "model.name",
                    "model.description",
                    "model.author",
                    "model.license"
                ].contains(&key.as_str())
            })
            .map(|(key, value)| format!("Set {key} = {value}"))
            .collect::<Vec<String>>();

        let files = self.files.iter()
            .map(|file| {
                match (file.delimiter.is_empty(), file.shuffle) {
                    (true,  false) => format!("File {}", file.path.display()),
                    (true,  true)  => format!("Shuffle File {}", file.path.display()),
                    (false, false) => format!("Split {} File {}", file.delimiter, file.path.display()),
                    (false, true)  => format!("Split {} Shuffle File {}", file.delimiter, file.path.display())
                }
            })
            .collect::<Vec<String>>();

        let mut lines = Vec::new();

        if let Some(value) = self.keys.get("model.name") {
            lines.push(format!("Name {value}"));
        }

        if let Some(value) = self.keys.get("model.description") {
            lines.push(format!("Description {value}"));
        }

        if let Some(value) = self.keys.get("model.author") {
            lines.push(format!("Author {value}"));
        }

        if let Some(value) = self.keys.get("model.license") {
            lines.push(format!("License {value}"));
        }

        if !lines.is_empty() {
            lines.push(String::new());
        }

        lines.push(format!(
            "{}{}Tokenizer {}/{}",
            if self.tokenizer.make_lowercase { "Lowercase " } else { "" },
            if self.tokenizer.force_alphanumeric { "Alphanumeric " } else { "" },
            self.tokenizer.num_tokens,
            self.tokenizer.num_samples
        ));

        lines.push(format!("Depth {}/{}", self.ngrams.num_from, self.ngrams.num_to));
        lines.push(format!("Experts {}/{}", self.experts.num_active, self.experts.num_total));
        lines.push(format!("Centroids {}", self.experts.num_centroids));

        lines.push(String::new());
        lines.push(format!("Template {}", self.template));

        if !stop_tokens.is_empty() {
            lines.push(String::new());
            lines.extend(stop_tokens);
        }

        if !keys.is_empty() {
            lines.push(String::new());
            lines.extend(keys);
        }

        if !files.is_empty() {
            lines.push(String::new());
            lines.extend(files);
        }

        f.write_str(&lines.join("\n"))
    }
}

impl FromStr for Recipe {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut recipe = Self::default();

        for line in s.lines() {
            if line.is_empty() {
                continue;
            }

            else if let Some(value) = line.strip_prefix("Name ") {
                recipe.keys.insert(
                    String::from("model.name"),
                    value.trim().to_string()
                );
            }

            else if let Some(value) = line.strip_prefix("Description ") {
                recipe.keys.insert(
                    String::from("model.description"),
                    value.trim().to_string()
                );
            }

            else if let Some(value) = line.strip_prefix("Author ") {
                recipe.keys.insert(
                    String::from("model.author"),
                    value.trim().to_string()
                );
            }

            else if let Some(value) = line.strip_prefix("License ") {
                recipe.keys.insert(
                    String::from("model.license"),
                    value.trim().to_string()
                );
            }

            else if let Some(value) = line.strip_prefix("Tokenizer ") {
                let (num_tokens, num_samples) = value.split_once('/')
                    .ok_or_else(|| anyhow::anyhow!("invalid tokenizer syntax"))?;

                let num_tokens = parse_num(&num_tokens.trim().to_ascii_lowercase())
                    .ok_or_else(|| anyhow::anyhow!("failed to parse tokenizer tokens number in model recipe"))?;

                if num_tokens > u16::MAX as usize {
                    anyhow::bail!("BPE tokenizer cannot have more than 65535 tokens");
                }

                recipe.tokenizer.make_lowercase = false;

                recipe.tokenizer.num_tokens = num_tokens as u16;

                recipe.tokenizer.num_samples = parse_num(&num_samples.trim().to_ascii_lowercase())
                    .ok_or_else(|| anyhow::anyhow!("failed to parse tokenizer samples number in model recipe"))?;
            }

            else if let Some(mut value) = line.strip_prefix("Lowercase ") {
                if let Some(new_value) = value.strip_prefix("Alphanumeric ") {
                    recipe.tokenizer.force_alphanumeric = true;

                    value = new_value;
                }

                let Some(value) = value.strip_prefix("Tokenizer ") else {
                    anyhow::bail!("invalid tokenizer syntax: {line}");
                };

                let (num_tokens, num_samples) = value.split_once('/')
                    .ok_or_else(|| anyhow::anyhow!("invalid tokenizer syntax"))?;

                let num_tokens = parse_num(&num_tokens.trim().to_ascii_lowercase())
                    .ok_or_else(|| anyhow::anyhow!("failed to parse tokenizer tokens number in model recipe"))?;

                if num_tokens > u16::MAX as usize {
                    anyhow::bail!("BPE tokenizer cannot have more than 65535 tokens");
                }

                recipe.tokenizer.make_lowercase = true;

                recipe.tokenizer.num_tokens = num_tokens as u16;

                recipe.tokenizer.num_samples = parse_num(&num_samples.trim().to_ascii_lowercase())
                    .ok_or_else(|| anyhow::anyhow!("failed to parse tokenizer samples number in model recipe"))?;
            }

            else if let Some(mut value) = line.strip_prefix("Alphanumeric ") {
                if let Some(new_value) = value.strip_prefix("Lowercase ") {
                    recipe.tokenizer.make_lowercase = true;

                    value = new_value;
                }

                let Some(value) = value.strip_prefix("Tokenizer ") else {
                    anyhow::bail!("invalid tokenizer syntax: {line}");
                };

                let (num_tokens, num_samples) = value.split_once('/')
                    .ok_or_else(|| anyhow::anyhow!("invalid tokenizer syntax"))?;

                let num_tokens = parse_num(&num_tokens.trim().to_ascii_lowercase())
                    .ok_or_else(|| anyhow::anyhow!("failed to parse tokenizer tokens number in model recipe"))?;

                if num_tokens > u16::MAX as usize {
                    anyhow::bail!("BPE tokenizer cannot have more than 65535 tokens");
                }

                recipe.tokenizer.force_alphanumeric = true;

                recipe.tokenizer.num_tokens = num_tokens as u16;

                recipe.tokenizer.num_samples = parse_num(&num_samples.trim().to_ascii_lowercase())
                    .ok_or_else(|| anyhow::anyhow!("failed to parse tokenizer samples number in model recipe"))?;
            }

            else if let Some(value) = line.strip_prefix("Depth ") {
                let Some((from, to)) = value.split_once("/") else {
                    anyhow::bail!("invalid depth parameter value: {line}");
                };

                recipe.ngrams.num_from = from.parse()
                    .with_context(|| format!("invalid from depth format: {line}"))?;

                recipe.ngrams.num_to = to.parse()
                    .with_context(|| format!("invalid to depth format: {line}"))?;
            }

            else if let Some(value) = line.strip_prefix("Experts ") {
                let Some((active, total)) = value.split_once("/") else {
                    anyhow::bail!("invalid experts parameter value: {line}");
                };

                recipe.experts.num_active = active.parse()
                    .with_context(|| format!("invalid active experts format: {line}"))?;

                recipe.experts.num_total = total.parse()
                    .with_context(|| format!("invalid total experts format: {line}"))?;
            }

            else if let Some(value) = line.strip_prefix("Centroids ") {
                recipe.experts.num_centroids = value.parse()
                    .with_context(|| format!("invalid centroids format: {line}"))?;
            }

            else if let Some(value) = line.strip_prefix("Template ") {
                recipe.template = value.trim().to_string();
            }

            else if let Some(value) = line.strip_prefix("Stop ") {
                let word = value.trim().to_string();

                if !recipe.stop_tokens.contains(&word) {
                    recipe.stop_tokens.push(word);
                }
            }

            else if let Some(value) = line.strip_prefix("Set ") {
                let Some((key, value)) = value.split_once(" = ") else {
                    anyhow::bail!("invalid set key parameter: {line}");
                };

                recipe.keys.insert(key.trim().to_string(), value.trim().to_string());
            }

            else if let Some(value) = line.strip_prefix("File ") {
                recipe.files.push(File {
                    path: PathBuf::from(value.trim()),
                    delimiter: String::from("<|document|>"),
                    shuffle: false
                });
            }

            else if let Some(value) = line.strip_prefix("Shuffle File ") {
                recipe.files.push(File {
                    path: PathBuf::from(value.trim()),
                    delimiter: String::from("<|document|>"),
                    shuffle: true
                });
            }

            else if let Some(value) = line.strip_prefix("Split ") {
                if let Some((delimiter, path)) = value.split_once(" Shuffle File ") {
                    recipe.files.push(File {
                        path: PathBuf::from(path.trim()),
                        delimiter: delimiter.trim().to_string(),
                        shuffle: true
                    });
                }

                else if let Some((delimiter, path)) = value.split_once(" File ") {
                    recipe.files.push(File {
                        path: PathBuf::from(path.trim()),
                        delimiter: delimiter.trim().to_string(),
                        shuffle: false
                    });
                }

                else {
                    anyhow::bail!("invalid split file parameter: {line}");
                }
            }

            else {
                anyhow::bail!("unknown model parameter: {line}");
            }
        }

        Ok(recipe)
    }
}

#[test]
fn test_recipe() -> anyhow::Result<()> {
    let recipe = Recipe {
        keys: HashMap::from_iter([
            (String::from("test"), String::from("123"))
        ]),
        tokenizer: Tokenizer {
            make_lowercase: true,
            force_alphanumeric: true,
            num_tokens: 1024,
            num_samples: 256 * 1024 * 1024
        },
        ngrams: Ngrams {
            num_from: 5,
            num_to: 2
        },
        experts: Experts {
            num_active: 4,
            num_total: 64,
            num_centroids: 4
        },
        template: String::from("{{content}}"),
        stop_tokens: vec![
            String::from("<|start|>"),
            String::from("<|stop|>"),
            String::from("<|document|>")
        ],
        files: vec![
            File {
                path: PathBuf::from("test"),
                delimiter: String::from("</test>"),
                shuffle: true
            }
        ]
    };

    assert_eq!(Recipe::from_str(&recipe.to_string())?, recipe);

    Ok(())
}
