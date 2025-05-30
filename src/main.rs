use std::str::FromStr;
use std::path::PathBuf;

use recipe::Recipe;

pub mod tokens;
pub mod tokenizer;
pub mod recipe;
pub mod clustering;

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

            let path = PathBuf::from(path);

            if !path.is_file() {
                anyhow::bail!("invalid recipe file path");
            }

            let recipe = std::fs::read_to_string(&path)?;
            let recipe = Recipe::from_str(&recipe)?;

            dbg!(recipe);
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
