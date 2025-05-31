use std::io::Write;
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
use model::Model16;

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

            let model = Model16::build(recipe)?;

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

            let model = Model16::open(std::fs::read(path)?)?;
            let mut rand = rand();

            #[allow(irrefutable_let_patterns)]
            let RecipeTokenizer::WordTokenizer { lowercase, punctuation } = model.tokenizer() else {
                anyhow::bail!("only word-tokenizer is currently supported");
            };

            let tokenizer = WordTokenizer {
                lowercase: *lowercase,
                punctuation: *punctuation
            };

            let stdin = std::io::stdin();

            loop {
                stdout.write_all(b"\n\n> ")?;
                stdout.flush()?;

                let mut query = String::new();

                stdin.read_line(&mut query)?;

                let query = model.encode_tokens(query.trim())?;
                let generator = model.generate_tokens(query, &mut rand)?;

                let mut decoder = tokenizer.decode(generator);

                std::io::copy(&mut decoder, &mut stdout)?;
            }
        }

        Some("help") | None => {
            println!("capacitor help - show this help message");
            println!("capacitor new <recipe path> - save new example model file");
            println!("capacitor build <recipe path> - build the model");
            println!("capacitor run <model path> - open model evaluation interface");
        }

        Some(command) => anyhow::bail!("unknown command: {command}")
    }

    Ok(())
}
