use std::io::{Read, Write};
use std::str::FromStr;
use std::path::PathBuf;

use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha12Rng;

pub mod tokens;
pub mod tokenizer;
pub mod transitions;
pub mod clustering;
pub mod recipe;
pub mod model;

use tokenizer::{Tokenizer, WordTokenizer};
use recipe::{Recipe, Tokenizer as RecipeTokenizer};
use model::Model32;

type Model = Model32;

pub fn rand() -> impl RngCore {
    let micros = std::time::UNIX_EPOCH.elapsed()
        .unwrap_or_default()
        .as_micros();

    ChaCha12Rng::seed_from_u64((micros & (u64::MAX as u128)) as u64)
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);

    match args.next().as_deref() {
        Some("new") => {
            let path = args.next()
                .unwrap_or_else(|| String::from("capacitorfile"));

            let recipe = Recipe::default();

            std::fs::write(path, recipe.to_string())?;

            println!("Saved example capacitor model recipe file");
        }

        Some("build") => {
            let path = args.next()
                .unwrap_or_else(|| String::from("capacitorfile"));

            let mut path = PathBuf::from(path);

            if path.is_dir() {
                path = path.join("capacitorfile");
            }

            if !path.is_file() {
                anyhow::bail!("invalid recipe file path");
            }

            let recipe = std::fs::read_to_string(&path)?;
            let mut recipe = Recipe::from_str(&recipe)?;

            if let Some(parent) = path.parent() {
                recipe = recipe.relative_to(parent);
            }

            let output_path = path.parent()
                .map(|parent| parent.join(&recipe.name))
                .unwrap_or_else(|| PathBuf::from(&recipe.name));

            println!("Building the model...");

            let model = Model::build(recipe)?;

            std::fs::write(&output_path, model.into_bytes())?;

            println!("Model saved as {:?}", output_path);
        }

        Some("run") => {
            let Some(path) = args.next().map(PathBuf::from) else {
                anyhow::bail!("missing model file path");
            };

            if !path.is_file() {
                anyhow::bail!("invalid model file path");
            }

            let mut stdout = std::io::stdout();

            stdout.write_all(b"Loading model...")?;
            stdout.flush()?;

            let model = Model::open(std::fs::read(path)?)?;

            let mut rand = rand();

            #[allow(irrefutable_let_patterns)]
            let Some(RecipeTokenizer::WordTokenizer { lowercase, punctuation }) = model.get_tokenizer()? else {
                anyhow::bail!("only word-tokenizer is currently supported");
            };

            let tokenizer = WordTokenizer { lowercase, punctuation };

            let stdin = std::io::stdin();

            loop {
                stdout.write_all(b"\n\n> ")?;
                stdout.flush()?;

                let mut query = String::new();

                stdin.read_line(&mut query)?;

                let query = model.encode_tokens(query.trim())?;
                let generator = model.generate_tokens(query, &mut rand)?;

                let mut decoder = tokenizer.decode(generator);

                let mut buf = [0; 32];

                loop {
                    let n = decoder.read(&mut buf)?;

                    if n == 0 {
                        break;
                    }

                    stdout.write_all(&buf[..n])?;
                    stdout.flush()?;
                }
            }
        }

        Some("show") => {
            let Some(path) = args.next().map(PathBuf::from) else {
                anyhow::bail!("missing model file path");
            };

            if !path.is_file() {
                anyhow::bail!("invalid model file path");
            }

            let model = Model::open(std::fs::read(path)?)?;

            println!("Base model transitions: {}", model.transitions_ref().read_list().len());

            for (i, expert) in model.experts_ref().iter().enumerate() {
                println!("Expert #{} transitions: {}", i + 1, expert.transitions().len());
            }

            println!();

            if !model.keys_ref().is_empty() {
                println!("Keys:");
            }

            for (key, value) in model.keys_ref() {
                println!("  [{key:?}] = {value:?}");
            }

            let mut stdout = std::io::stdout();
            let stdin = std::io::stdin();

            #[allow(irrefutable_let_patterns)]
            let Some(RecipeTokenizer::WordTokenizer { lowercase, punctuation }) = model.get_tokenizer()? else {
                anyhow::bail!("only word-tokenizer is currently supported");
            };

            let tokenizer = WordTokenizer { lowercase, punctuation };

            loop {
                stdout.write_all(b"\n\n> ")?;
                stdout.flush()?;

                let mut query = String::new();

                stdin.read_line(&mut query)?;

                let query = model.encode_tokens(query.trim())?;

                stdout.write_all(b"Query tokens:\n")?;

                for token in &query {
                    stdout.write_all(format!(" {token}").as_bytes())?;
                }

                stdout.write_all(b"\n\n")?;
                stdout.flush()?;

                let transitions = model.transitions_ref()
                    .find_transitions(&query);

                if !transitions.is_empty() {
                    stdout.write_all(b"Base model:\n")?;
                    stdout.flush()?;

                    for transition in transitions {
                        let to = transition.to.iter()
                            .flat_map(|token| {
                                model.tokens_ref().find_word(*token)
                            });

                        let mut to_str = String::new();

                        tokenizer.decode(to).read_to_string(&mut to_str)?;

                        stdout.write_all(format!("  [{:8}] {to_str}\n", transition.weight).as_bytes())?;
                        stdout.flush()?;
                    }
                }
            }
        }

        Some("set") => {
            let Some(path) = args.next().map(PathBuf::from) else {
                anyhow::bail!("missing model file path");
            };

            let Some(key) = args.next() else {
                anyhow::bail!("missing metadata key");
            };

            let Some(value) = args.next() else {
                anyhow::bail!("missing metadata value");
            };

            if !path.is_file() {
                anyhow::bail!("invalid model file path");
            }

            let mut model = Model::open(std::fs::read(&path)?)?;

            model.keys_mut().insert(key, value);

            std::fs::write(&path, model.into_bytes())?;

            println!("Updated model in {path:?}");
        }

        Some("export-tokens") => {
            let Some(path) = args.next().map(PathBuf::from) else {
                anyhow::bail!("missing model file path");
            };

            if !path.is_file() {
                anyhow::bail!("invalid model file path");
            }

            let model = Model::open(std::fs::read(path)?)?;

            println!("token,word");

            for (token, word) in model.tokens_ref().as_table() {
                println!("{token},\"{}\"", word.replace('"', "\\\""));
            }
        }

        Some("export-transitions") => {
            let Some(path) = args.next().map(PathBuf::from) else {
                anyhow::bail!("missing model file path");
            };

            if !path.is_file() {
                anyhow::bail!("invalid model file path");
            }

            let model = Model::open(std::fs::read(path)?)?;

            println!("from,to,weight");

            for transition in model.transitions_ref().read_list() {
                println!(
                    "\"{:?}\",\"{:?}\",{}",
                    transition.from,
                    transition.to,
                    transition.weight as f64 / u32::MAX as f64
                );
            }
        }

        Some("help") | None => {
            println!("capacitor help - show this help message");
            println!("capacitor new <recipe path> - save new example model file");
            println!("capacitor build <recipe path> - build the model");
            println!("capacitor run <model path> - open model inference interface");
            println!("capacitor show <model path> - show model info");
            println!("capacitor set <model path> <key> <value> - set metadata key value to the model");
            println!("capacitor export-tokens <model path> - export tokens csv table to stdout");
            println!("capacitor export-transitions <model path> - export base model transitions csv table to stdout");
        }

        Some(command) => anyhow::bail!("unknown command: {command}")
    }

    Ok(())
}
