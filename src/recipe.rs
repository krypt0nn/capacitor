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
    pub make_lowercase: bool,

    /// How many tokens BPE tokenizer should learn.
    pub num_tokens: usize,

    /// How many pre-tokenized words will be used to train the BPE tokenizer.
    /// Higher value will produce better quality tokens in cost of increased
    /// build time.
    pub num_samples: usize
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recipe {
    /// Metadata key-values table.
    pub keys: HashMap<String, String>,

    /// Dataset files.
    pub files: Vec<File>,

    /// Documents tokenizer.
    pub tokenizer: Tokenizer,

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

    /// Amount of `from` tokens in transitions.
    pub from_depth: usize,

    /// Amount of `to` tokens in transitions.
    pub to_depth: usize,

    /// Total amount of experts (clusters).
    pub total_experts: usize,

    /// Amount of active experts at a time.
    pub active_experts: usize,

    /// Amount of centroids in each cluster in documents clustering algorithm.
    pub centroids: usize
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
            files: Vec::new(),
            tokenizer: Tokenizer {
                make_lowercase: false,
                num_tokens: 1024,
                num_samples: 256 * 1024 * 1024
            },
            template: String::from("{{content}}"),
            stop_tokens: vec![
                String::from("<|start|>"),
                String::from("<|stop|>"),
                String::from("<|document|>"),
                String::from("<|answer|>")
            ],
            from_depth: 2,
            to_depth: 1,
            total_experts: 4,
            active_experts: 1,
            centroids: 2
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
            "{}Tokenizer {}/{}",
            if self.tokenizer.make_lowercase { "Lowercase " } else { "" },
            self.tokenizer.num_tokens,
            self.tokenizer.num_samples
        ));

        lines.push(format!("Template {}", self.template));
        lines.push(format!("Depth {}/{}", self.from_depth, self.to_depth));
        lines.push(format!("Experts {}/{}", self.active_experts, self.total_experts));
        lines.push(format!("Centroids {}", self.centroids));

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
        let mut tokenizer = Tokenizer {
            make_lowercase: false,
            num_tokens: 1024,
            num_samples: 256 * 1024 * 1024
        };
        let mut template = String::from("{{content}}");
        let mut stop_tokens = Vec::new();
        let mut keys = HashMap::new();
        let mut files = Vec::new();
        let mut from_depth = 1;
        let mut to_depth = 1;
        let mut total_experts = 0;
        let mut active_experts = 0;
        let mut centroids = 0;

        for line in s.lines() {
            if line.is_empty() {
                continue;
            }

            else if let Some(value) = line.strip_prefix("Name ") {
                keys.insert(String::from("model.name"), value.trim().to_string());
            }

            else if let Some(value) = line.strip_prefix("Description ") {
                keys.insert(String::from("model.description"), value.trim().to_string());
            }

            else if let Some(value) = line.strip_prefix("Author ") {
                keys.insert(String::from("model.author"), value.trim().to_string());
            }

            else if let Some(value) = line.strip_prefix("License ") {
                keys.insert(String::from("model.license"), value.trim().to_string());
            }

            else if let Some(value) = line.strip_prefix("Tokenizer ") {
                let (num_tokens, num_samples) = value.split_once('/')
                    .ok_or_else(|| anyhow::anyhow!("invalid tokenizer syntax"))?;

                tokenizer.make_lowercase = false;

                tokenizer.num_tokens = parse_num(&num_tokens.trim_ascii().to_ascii_lowercase())
                    .ok_or_else(|| anyhow::anyhow!("failed to parse tokenizer tokens number in model recipe"))?;

                tokenizer.num_samples = parse_num(&num_samples.trim_ascii().to_ascii_lowercase())
                    .ok_or_else(|| anyhow::anyhow!("failed to parse tokenizer samples number in model recipe"))?;
            }

            else if let Some(value) = line.strip_prefix("Lowercase ")
                && let Some(value) = value.strip_prefix("Tokenizer ")
            {
                let (num_tokens, num_samples) = value.split_once('/')
                    .ok_or_else(|| anyhow::anyhow!("invalid tokenizer syntax"))?;

                tokenizer.make_lowercase = true;

                tokenizer.num_tokens = parse_num(&num_tokens.trim_ascii().to_ascii_lowercase())
                    .ok_or_else(|| anyhow::anyhow!("failed to parse tokenizer tokens number in model recipe"))?;

                tokenizer.num_samples = parse_num(&num_samples.trim_ascii().to_ascii_lowercase())
                    .ok_or_else(|| anyhow::anyhow!("failed to parse tokenizer samples number in model recipe"))?;
            }

            else if let Some(value) = line.strip_prefix("Template ") {
                template = value.trim().to_string();
            }

            else if let Some(value) = line.strip_prefix("Stop ") {
                stop_tokens.push(value.trim().to_string());
            }

            else if let Some(value) = line.strip_prefix("Set ") {
                let Some((key, value)) = value.split_once(" = ") else {
                    anyhow::bail!("invalid set key parameter: {line}");
                };

                keys.insert(key.trim().to_string(), value.trim().to_string());
            }

            else if let Some(value) = line.strip_prefix("File ") {
                files.push(File {
                    path: PathBuf::from(value.trim()),
                    delimiter: String::from("<|document|>"),
                    shuffle: false
                });
            }

            else if let Some(value) = line.strip_prefix("Shuffle File ") {
                files.push(File {
                    path: PathBuf::from(value.trim()),
                    delimiter: String::from("<|document|>"),
                    shuffle: true
                });
            }

            else if let Some(value) = line.strip_prefix("Split ") {
                if let Some((delimiter, path)) = value.split_once(" Shuffle File ") {
                    files.push(File {
                        path: PathBuf::from(path.trim()),
                        delimiter: delimiter.trim().to_string(),
                        shuffle: true
                    });
                }

                else if let Some((delimiter, path)) = value.split_once(" File ") {
                    files.push(File {
                        path: PathBuf::from(path.trim()),
                        delimiter: delimiter.trim().to_string(),
                        shuffle: false
                    });
                }

                else {
                    anyhow::bail!("invalid split file parameter: {line}");
                }
            }

            else if let Some(value) = line.strip_prefix("Depth ") {
                let Some((from, to)) = value.split_once("/") else {
                    anyhow::bail!("invalid depth parameter value: {line}");
                };

                from_depth = from.parse()
                    .with_context(|| format!("invalid from depth format: {line}"))?;

                to_depth = to.parse()
                    .with_context(|| format!("invalid to depth format: {line}"))?;
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

            else if let Some(value) = line.strip_prefix("Centroids ") {
                centroids = value.parse()
                    .with_context(|| format!("invalid centroids format: {line}"))?;
            }

            else {
                anyhow::bail!("unknown model parameter: {line}");
            }
        }

        Ok(Self {
            keys,
            files,
            tokenizer,
            template,
            stop_tokens,
            from_depth,
            to_depth,
            total_experts,
            active_experts,
            centroids
        })
    }
}

#[test]
fn test_recipe() -> anyhow::Result<()> {
    let recipe = Recipe {
        keys: HashMap::from_iter([
            (String::from("test"), String::from("123"))
        ]),
        files: vec![
            File {
                path: PathBuf::from("test"),
                delimiter: String::from("</test>"),
                shuffle: true
            }
        ],
        tokenizer: Tokenizer {
            make_lowercase: true,
            num_tokens: 1024,
            num_samples: 256 * 1024 * 1024
        },
        template: String::from("{{content}}"),
        stop_tokens: vec![
            String::from("<|document|>")
        ],
        from_depth: 5,
        to_depth: 2,
        total_experts: 64,
        active_experts: 4,
        centroids: 4
    };

    assert_eq!(Recipe::from_str(&recipe.to_string())?, recipe);

    Ok(())
}
