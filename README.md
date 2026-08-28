# Capacitor

Logical continuation of my [markov-chains](https://github.com/krypt0nn/markov-chains)
project for text generation using highly modified Markov chains architecture.

## Features

- A generalized recipe file with instructions to build the model from sources;
- BPE-style tokenizer (unicode characters instead of bytes);
- Configurable N-gram `from_num` -> `to_num` transitions table;
- Custom MoE-like architecture (prompt classification, per-class transitions);
- Custom highly efficient model storage format.

## Model building

Every model is defined with a special `capacitorfile` recipe. It is a special
text file format that defines datasets, tokenizer, metadata keys, and other
settings needed to build the model. Recipes can be used to build models from
"raw ingredients" by anyone with a single `capacitor build` command run.

Example recipe:

```
Name example-base
Description An example base model
Author Nikita Podvirnyi <krypt0nn@dawn.wine>
License MIT

Lowercase Tokenizer 16K/16M
Depth 3/1
Experts 1/32
Centroids 2

Template <|content|>{{content}}

Stop <|start|>
Stop <|stop|>
Stop <|document|>
Stop <|username|>
Stop <|content|>

Set model.build.cutoff_date = May 30, 2025
Set model.build.build_date = Aug 28, 2026

Set model.inference.top_k = 8
Set model.inference.temperature = 1.0
Set model.inference.max_tokens = 80
Set model.inference.experts_context = 16

Split <|document|> Shuffle File dataset.txt
```

### Dataset

Datasets are loaded with `File [path]` keyword. It can be modified with
`Split [delimiter]` prefix if your input file contains multiple training samples
(documents). In that case, you combine multiple documents with some unique
string which splits them apart. It's recommended to stick to `<|document|>`
name: `Split <|document|> File [path]`. Finally, you can add `Shuffle` keyword
to randomly shuffle the documents: `Shuffle Split <|document|> File [path]`.

### Tokenizer

BPE-style tokenizer is configured with `Tokenizer [learn_tokens]/[sample_chars]`
keyword: the model will learn at least `learn_tokens` tokens using
`sample_chars` per-tokenized characters. Higher `learn_tokens` value will force
the model to learn not just words, but entire parts of sentences. For example,
`hi chat` can become a single token. Similarly, increasing `sample_chars` will
slow down build time, but increase generated tokens quality.

Tokenizer can be modified with special keywords:

| Keyword        | Description                                         |
| -------------- | --------------------------------------------------- |
| `Lowercase`    | Convert every character into lowercase.             |
| `Alphanumeric` | Remove all non-alpha-numeric, non-space characters. |

`Lowercase Alphanumeric Tokenizer 0/0` is the most minimal tokenizer possible.

### N-grams

Model can learn transitions (continuations) for multiple amount of previous and
next tokens, not just 1-to-1. For example, it can learn this chain:
`[hi] [chat] -> [how] [are] [you]`. This can be controlled with
`Depth [from_tokens]/[to_tokens]`. For our example, `from_tokens = 2`,
`to_tokens = 3`. Higher values will increase model size on disk, but improve
its output quality.

Remember that higher tokens number means tokenizer will learn *words* and
*multiple words* as single tokens, so you can scale both tokens number and
N-grams.

### Experts

Model can randomly select `Centroids [num_centroids]` documents from the dataset
and use a classic clusterizing algorithm to sample the whole dataset between
`Experts [num_active]/[num_total]` total experts. Model will always have a
common table of transitions between every token, and have transitions remembered
for documents stored in experts individually. At inference time, model can
use available tokens to select N active experts and apply their transitions
table on top of shared one to select domain-specific tokens with higher
priority.

## Inference params

| Metadata key                      | Description                                     |
| --------------------------------- | ----------------------------------------------- |
| `model.inference.top_k`           | How many next token variants are considered.    |
| `model.inference.temperature`     | Next token variance from learned probabilities. |
| `model.inference.max_tokens`      | Hard limit of maximal amount of output tokens.  |
| `model.inference.experts_context` | Context window of experts.                      |

Top-K and temperature can be specified via run command args:

```bash
capacitor run <model> --top-k 20 --temperature 1.4
```

Author: [Nikita Podvirnyi](https://github.com/krypt0nn)\
Licensed under [GPL-3.0-or-later](LICENSE)
