use std::cmp::Ordering;
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

use tokens::QuantizedToken;
use tokenizer::{Tokenizer, WordTokenizer};
use recipe::{Recipe, Tokenizer as RecipeTokenizer};
use model::Model;

// type QuantizedModel = Model<2, u16>;
type QuantizedModel = Model<3, QuantizedToken<3>>;
// type QuantizedModel = Model<4, u32>;

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
            let mut path = args.next()
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("capacitorfile"));

            if path.is_dir() {
                path = path.join("capacitorfile");
            }

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

            let model = QuantizedModel::build(recipe, |curr, total| {
                println!("Building experts {curr}/{total} ({:.2} %)...", curr as f32 / total as f32 * 100.0);
            })?;

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

            let mut verbose = false;

            for flag in args {
                if ["-v", "--verbose"].contains(&flag.as_str()) {
                    verbose = true;
                }

                else {
                    anyhow::bail!("unknown flag: {flag}");
                }
            }

            let mut stdout = std::io::stdout();

            stdout.write_all(b"Loading model...")?;
            stdout.flush()?;

            let model = QuantizedModel::open(std::fs::read(path)?)?;

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

                let mut generator = model.generate_tokens(query.trim(), &mut rand)?;

                if !verbose {
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

                else {
                    for token in &mut generator {
                        stdout.write_all(token.as_bytes())?;
                        stdout.write_all(b" ")?;
                        stdout.flush()?;
                    }

                    let stats = generator.stats();
                    let mut experts_usage = Vec::new();

                    for i in 0..stats.total_experts() {
                        let Some(usage) = stats.expert_frequency(i) else {
                            continue;
                        };

                        if usage > 0.0 {
                            experts_usage.push((usage, format!("  [Expert {i:3}] {:.2} %\n", usage * 100.0)));
                        }
                    }

                    if !experts_usage.is_empty() {
                        experts_usage.sort_by(|a, b| {
                            b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal)
                        });

                        stdout.write_all(b"\n\nExperts use:\n")?;

                        for (_, line) in experts_usage {
                            stdout.write_all(line.as_bytes())?;
                        }

                        stdout.flush()?;
                    }
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

            let model = QuantizedModel::open(std::fs::read(path)?)?;

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

            let mut model = QuantizedModel::open(std::fs::read(&path)?)?;

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

            let model = QuantizedModel::open(std::fs::read(path)?)?;

            println!("token,word");

            for (token, word) in model.tokens_ref().as_tokens_table() {
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

            let model = QuantizedModel::open(std::fs::read(path)?)?;

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

        #[cfg(feature = "http-api")]
        Some("serve") => {
            use std::sync::Mutex;
            use std::collections::HashMap;
            use std::time::{Instant, UNIX_EPOCH};

            use rouille::Response;
            use serde_json::json;

            let address = args.next().unwrap_or_else(|| String::from("0.0.0.0:8080"));

            let models = Mutex::new(HashMap::new());

            struct ModelInfo {
                pub model: QuantizedModel,
                pub loaded_at: Instant,
                pub loaded_timestamp: u64,
                pub requests_count: usize
            }

            println!("Started HTTP REST API server at {address}");

            rouille::start_server(address, move |request| {
                if request.method() != "GET" {
                    return Response::empty_404();
                }

                println!("[{}] {}", request.remote_addr(), request.raw_url());

                match request.url().as_str() {
                    "/api/v1/load" => {
                        let Some(path) = request.get_param("model") else {
                            return Response::json(&json!({
                                "type": "error",
                                "error": "model path is missing"
                            }));
                        };

                        let model = match std::fs::read(&path) {
                            Ok(model) => model,
                            Err(err) => {
                                return Response::json(&json!({
                                    "type": "error",
                                    "error": "failed to load model",
                                    "context": err.to_string(),
                                    "path": &path
                                }));
                            }
                        };

                        let model = match QuantizedModel::open(model) {
                            Ok(model) => model,
                            Err(err) => {
                                return Response::json(&json!({
                                    "type": "error",
                                    "error": "failed to load model",
                                    "context": err.to_string(),
                                    "path": &path
                                }));
                            }
                        };

                        match models.lock() {
                            Ok(mut models) => {
                                models.insert(path, ModelInfo {
                                    model,
                                    loaded_at: Instant::now(),
                                    loaded_timestamp: UNIX_EPOCH.elapsed()
                                        .unwrap_or_default()
                                        .as_secs(),
                                    requests_count: 0
                                });

                                Response::json(&json!({
                                    "status": "success"
                                }))
                            }

                            Err(err) => {
                                Response::json(&json!({
                                    "type": "error",
                                    "error": "failed to lock models mutex",
                                    "context": err.to_string()
                                }))
                            }
                        }
                    }

                    "/api/v1/unload" => {
                        let Some(path) = request.get_param("model") else {
                            return Response::json(&json!({
                                "type": "error",
                                "error": "model path is missing"
                            }));
                        };

                        match models.lock() {
                            Ok(mut models) => {
                                models.remove(&path);

                                Response::json(&json!({
                                    "status": "success"
                                }))
                            }

                            Err(err) => {
                                Response::json(&json!({
                                    "type": "error",
                                    "error": "failed to lock models mutex",
                                    "context": err.to_string()
                                }))
                            }
                        }
                    }

                    "/api/v1/stats" => {
                        match models.lock() {
                            Ok(models) => {
                                let mut stats = Vec::new();

                                let now = UNIX_EPOCH.elapsed()
                                    .unwrap_or_default();

                                for (path, info) in models.iter() {
                                    let mut experts = Vec::new();
                                    let mut total_size = 0;

                                    total_size += info.model.tokens_ref().size();
                                    total_size += info.model.transitions_ref().size();

                                    for expert in info.model.experts_ref() {
                                        experts.push(json!({
                                            "transitions": {
                                                "len": expert.transitions().len(),
                                                "size": expert.transitions().size()
                                            }
                                        }));

                                        total_size += expert.transitions().size();
                                    }

                                    stats.push(json!({
                                        "path": path,
                                        "model": {
                                            "total_size": total_size,
                                            "tokens": {
                                                "len": info.model.tokens_ref().len(),
                                                "size": info.model.tokens_ref().size()
                                            },
                                            "base": {
                                                "transitions": {
                                                    "len": info.model.transitions_ref().len(),
                                                    "size": info.model.transitions_ref().size()
                                                }
                                            },
                                            "experts": experts,
                                            "metadata": info.model.keys_ref()
                                        },
                                        "stats": {
                                            "time": {
                                                "server_time": now.as_secs(),
                                                "loaded_time": info.loaded_timestamp,
                                                "running_time": info.loaded_at.elapsed().as_secs()
                                            },
                                            "requests": {
                                                "count": info.requests_count
                                            }
                                        }
                                    }));
                                }

                                Response::json(&json!({
                                    "status": "success",
                                    "stats": stats
                                }))
                            }

                            Err(err) => {
                                Response::json(&json!({
                                    "type": "error",
                                    "error": "failed to lock models mutex",
                                    "context": err.to_string()
                                }))
                            }
                        }
                    }

                    "/api/v1/generate" => {
                        let Some(path) = request.get_param("model") else {
                            return Response::json(&json!({
                                "type": "error",
                                "error": "model path is missing"
                            }));
                        };

                        let Some(query) = request.get_param("query") else {
                            return Response::json(&json!({
                                "type": "error",
                                "error": "query is missing"
                            }));
                        };

                        match models.lock() {
                            Ok(mut models) => {
                                let Some(model) = models.get_mut(&path) else {
                                    return Response::json(&json!({
                                        "type": "error",
                                        "error": "model is not loaded",
                                        "path": &path
                                    }));
                                };

                                model.requests_count += 1;

                                let Ok(tokenizer) = model.model.get_tokenizer() else {
                                    return Response::json(&json!({
                                        "type": "error",
                                        "error": "failed to load model tokenizer",
                                        "path": &path
                                    }));
                                };

                                #[allow(irrefutable_let_patterns)]
                                let Some(RecipeTokenizer::WordTokenizer { lowercase, punctuation }) = tokenizer else {
                                    return Response::json(&json!({
                                        "type": "error",
                                        "error": "failed to load model tokenizer: only word-tokenizer is currently supported",
                                        "path": &path
                                    }));
                                };

                                let tokenizer = WordTokenizer { lowercase, punctuation };

                                let mut rand = rand();

                                let generator = match model.model.generate_tokens(query.trim(), &mut rand) {
                                    Ok(generator) => generator,
                                    Err(err) => {
                                        return Response::json(&json!({
                                            "type": "error",
                                            "error": "failed to load tokens generator",
                                            "context": err.to_string(),
                                            "path": &path,
                                            "query": &query
                                        }));
                                    }
                                };

                                let mut decoder = tokenizer.decode(generator);
                                let mut response = String::new();

                                if let Err(err) = decoder.read_to_string(&mut response) {
                                    return Response::json(&json!({
                                        "type": "error",
                                        "error": "failed to generate response",
                                        "context": err.to_string(),
                                        "path": &path,
                                        "query": &query
                                    }));
                                }

                                Response::json(&json!({
                                    "status": "success",
                                    "response": response
                                }))
                            }

                            Err(err) => {
                                Response::json(&json!({
                                    "type": "error",
                                    "error": "failed to lock models mutex",
                                    "context": err.to_string()
                                }))
                            }
                        }
                    }

                    _ => Response::empty_404()
                }
            })
        }

        Some("help") | None => {
            println!("capacitor help - show this help message");
            println!("capacitor new <recipe path> - save new example model file");
            println!("capacitor build <recipe path> - build the model");
            println!("capacitor run <model path> [--verbose] - open model inference interface");
            println!("capacitor show <model path> - show model info");
            println!("capacitor set <model path> <key> <value> - set metadata key value to the model");
            println!("capacitor export-tokens <model path> - export tokens csv table to stdout");
            println!("capacitor export-transitions <model path> - export base model transitions csv table to stdout");

            #[cfg(feature = "http-api")]
            println!("capacitor serve <address> - start HTTP REST API");
        }

        Some(command) => anyhow::bail!("unknown command: {command}")
    }

    Ok(())
}
