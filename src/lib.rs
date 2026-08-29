// SPDX-License-Identifier: GPL-3.0-or-later
//
// capacitor
// Copyright (C) 2026  Nikita Podvirnyi <krypt0nn@vk.ru>
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

use std::path::Path;
use std::io::Write;

use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha12Rng;

pub mod tokens;
pub mod transitions;
pub mod clustering;
pub mod recipe;
pub mod model;

use recipe::Recipe;
use model::{BuildProgress, Model, TokensGenerator};

#[inline]
fn get_rng() -> ChaCha12Rng {
    let micros = std::time::UNIX_EPOCH.elapsed()
        .unwrap_or_default()
        .as_micros();

    ChaCha12Rng::seed_from_u64((micros & (u64::MAX as u128)) as u64)
}

/// Build new model.
#[inline]
pub fn build(
    recipe: Recipe,
    updater: impl Fn(BuildProgress) + Send + Sync
) -> anyhow::Result<Model> {
    Model::build(recipe, get_rng(), updater)
}

/// Load model.
#[inline]
pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Model> {
    Model::open(std::fs::read(path)?)
}

/// Get tokens generator.
#[inline]
pub fn generator<'model>(
    model: &'model Model,
    prompt: impl AsRef<str>
) -> anyhow::Result<TokensGenerator<'model, ChaCha12Rng>> {
    model.generate(prompt, get_rng())
}

/// Generate new tokens directly into provided buffer.
#[inline]
pub fn generate(
    model: &Model,
    prompt: impl AsRef<str>,
    writer: &mut impl Write
) -> anyhow::Result<()> {
    std::io::copy(&mut generator(model, prompt)?, writer)?;

    Ok(())
}
