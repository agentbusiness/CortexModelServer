//! Recipes: one model on one engine with the exact settings of the app it comes from. They are
//! shipped data (`data/recipes/<engine>.toml`, the app's source, read from the live folder when
//! it has them), not records.

use crate::util;
use cc_protocol::contracts::serve::{Recipe, RecipeKeys};
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

#[derive(Deserialize, Default)]
struct File {
    #[serde(default)]
    recipe: Vec<Raw>,
}

#[derive(Deserialize, Default)]
struct Raw {
    id: String,
    name: String,
    engine: String,
    #[serde(default)]
    origin: String,
    #[serde(default = "chat")]
    kind: String,
    #[serde(default)]
    family: String,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    platforms: Vec<String>,
    #[serde(default)]
    backends: Vec<String>,
    #[serde(default)]
    repo: String,
    #[serde(default)]
    revision: String,
    #[serde(default)]
    weights_bytes: u64,
    #[serde(default)]
    compute_bytes: u64,
    #[serde(default)]
    kv_bytes_per_token: u64,
    #[serde(default)]
    min_gpu_bytes: u64,
    #[serde(default)]
    ctx_default: u32,
    #[serde(default)]
    ctx_max: u32,
    #[serde(default)]
    ctx_per_slot: bool,
    #[serde(default)]
    ctx_min_per_slot: u32,
    #[serde(default)]
    reserve_bytes: u64,
    #[serde(default)]
    slots_default: u32,
    #[serde(default)]
    slots_max: u32,
    #[serde(default)]
    vision: bool,
    #[serde(default)]
    speculative: String,
    #[serde(default)]
    config: Option<toml::Table>,
    #[serde(default)]
    keys: Option<toml::Table>,
    #[serde(default = "yes")]
    auto: bool,
}

fn yes() -> bool {
    true
}

fn chat() -> String {
    "chat".into()
}

/// Every recipe in `dir` (`*.toml`), in file and then listed order. A file that does not parse is
/// logged and skipped; the others still load.
pub fn load(dir: &Path) -> Vec<Recipe> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|entries| entries.filter_map(Result::ok).map(|entry| entry.path()).filter(|path| path.extension().is_some_and(|ext| ext == "toml")).collect())
        .unwrap_or_default();
    files.sort();
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for path in files {
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(error) => {
                tracing::warn!(path = %path.display(), %error, "a recipe file cannot be read");
                continue;
            }
        };
        let file: File = match toml::from_str(&text) {
            Ok(file) => file,
            Err(error) => {
                tracing::warn!(path = %path.display(), %error, "a recipe file does not parse; its recipes are left out");
                continue;
            }
        };
        for raw in file.recipe {
            if raw.id.is_empty() || raw.engine.is_empty() || !seen.insert(raw.id.clone()) {
                tracing::warn!(id = %raw.id, path = %path.display(), "a recipe without an id or engine, or named twice, is left out");
                continue;
            }
            out.push(convert(raw));
        }
    }
    out
}

fn table(table: Option<toml::Table>) -> Value {
    table.and_then(|table| serde_json::to_value(table).ok()).unwrap_or(Value::Object(Default::default()))
}

fn convert(raw: Raw) -> Recipe {
    let keys = table(raw.keys);
    let key = |name: &str| keys.get(name).and_then(Value::as_str).unwrap_or_default().to_owned();
    Recipe {
        id: raw.id,
        name: raw.name,
        engine: raw.engine,
        origin: raw.origin,
        kind: raw.kind,
        family: raw.family,
        tags: raw.tags,
        platforms: raw.platforms,
        backends: raw.backends,
        repo: raw.repo,
        revision: raw.revision,
        weights_bytes: raw.weights_bytes,
        compute_bytes: raw.compute_bytes,
        kv_bytes_per_token: raw.kv_bytes_per_token,
        min_gpu_bytes: raw.min_gpu_bytes,
        ctx_default: raw.ctx_default,
        ctx_max: raw.ctx_max.max(raw.ctx_default),
        ctx_per_slot: raw.ctx_per_slot,
        ctx_min_per_slot: raw.ctx_min_per_slot,
        ctx_min: 0,
        reserve_bytes: raw.reserve_bytes,
        // 0: the engine's own fitter picks the slots (CortexAgent's).
        slots_default: raw.slots_default,
        slots_max: raw.slots_max.max(raw.slots_default).max(1),
        vision: raw.vision,
        speculative: raw.speculative,
        config: table(raw.config),
        keys: RecipeKeys {
            slots: key("slots"),
            ctx: key("ctx"),
            cache_ram: key("cache_ram"),
            checkpoints: key("checkpoints"),
            kv_type: key("kv_type"),
            speculative: key("speculative"),
            draft_max: key("draft_max"),
            idle_unload_s: key("idle_unload_s"),
            memory_limit: key("memory_limit"),
        },
        auto: raw.auto,
        available: false,
        reason: String::new(),
        downloaded: false,
    }
}

/// Mark what can run here: this platform, a backend one of the GPUs has, the engine installed
/// and enabled. And whether the weights are in `~/.cortex-models` already.
pub fn mark(recipes: &mut [Recipe], backends: &BTreeSet<String>, engines: &BTreeSet<String>) {
    let platform = util::platform();
    for recipe in recipes {
        recipe.downloaded = downloaded(recipe);
        let reason = if !recipe.platforms.is_empty() && !recipe.platforms.iter().any(|p| p == platform) {
            format!("runs on {} only", recipe.platforms.join(" and "))
        } else if !recipe.backends.is_empty() && !recipe.backends.iter().any(|b| backends.contains(b) || b == "cpu") {
            format!("needs a {} GPU", recipe.backends.join(" or ").to_uppercase())
        } else if !engines.contains(&recipe.engine) {
            format!("the engine `{}` is not installed or is disabled", recipe.engine)
        } else {
            String::new()
        };
        recipe.available = reason.is_empty();
        recipe.reason = reason;
    }
}

/// The recipe's weights folder: `~/.cortex-models/<org>-<repo>/<revision>`.
pub fn folder(recipe: &Recipe) -> Option<PathBuf> {
    if recipe.repo.is_empty() || recipe.revision.is_empty() {
        return None;
    }
    Some(util::models_root().join(recipe.repo.replace('/', "-")).join(&recipe.revision))
}

fn downloaded(recipe: &Recipe) -> bool {
    folder(recipe).is_some_and(|dir| std::fs::read_dir(&dir).is_ok_and(|mut entries| entries.next().is_some()))
}
