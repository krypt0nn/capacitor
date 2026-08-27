# Capacitor

Logical continuation of my [markov-chains](https://github.com/krypt0nn/markov-chains)
project for text generation using highly modified Markov chains architecture.

## Features

- A generalized recipe file with instructions to build the model from sources;
- BPE-style tokenizer (unicode characters instead of bytes);
- Configurable N-gram `from_num` -> `to_num` transitions table;
- Custom MoE-like architecture (prompt classification, per-class transitions);
- Custom highly efficient model storage format.

Author: [Nikita Podvirnyi](https://github.com/krypt0nn)\
Licensed under [GPL-3.0](LICENSE)
