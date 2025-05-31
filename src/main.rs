use std::str::FromStr;
use std::path::PathBuf;

pub mod tokens;
pub mod tokenizer;
pub mod transitions;
pub mod clustering;
pub mod recipe;
pub mod model;

use recipe::Recipe;
use model::Model16;

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

        Some("help") | None => {
            println!("capacitor help - show this help message");
            println!("capacitor new <recipe path> - save new example model file");
            println!("capacitor build <recipe path> - build the model");
        }

        Some(command) => anyhow::bail!("unknown command: {command}")
    }

    Ok(())
}
