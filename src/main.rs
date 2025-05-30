use rand_chacha::rand_core::SeedableRng;

pub mod tokens;
pub mod tokenizer;
pub mod recipe;
pub mod clustering;

fn main() {
    let mut rand = rand_chacha::ChaCha20Rng::seed_from_u64(42);

    let documents = [
        vec!["hello", "world"].into_boxed_slice(),
        vec!["hi", "how", "are", "you"].into_boxed_slice(),
        vec!["who", "are", "you"].into_boxed_slice(),
        vec!["hello", "amogus"].into_boxed_slice(),
        // vec!["aboba"].into_boxed_slice()
    ];

    dbg!(clustering::clusterize(3, 1, &documents, &mut rand).unwrap());
}
