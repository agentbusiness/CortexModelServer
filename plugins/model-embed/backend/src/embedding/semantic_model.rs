//! The model the semantic index embeds with, and where its files come from.
//!
//! The model is pinned to one Hugging Face revision and each of its files to
//! a size and SHA-256, so every install embeds with the same weights and
//! tokenizer and its vectors stay comparable. A packaged build names a
//! directory holding the three files; any other build finds them in the
//! shared models folder (`crate::shared_models`), downloading them once.

mod local_semantic_model_files;
mod provision_semantic_model_files;

pub use local_semantic_model_files::local_semantic_model_files;
pub use provision_semantic_model_files::provision_semantic_model_files;

use std::path::PathBuf;

use crate::shared_models::PinnedModelFile;

/// The semantic embedding model's name, which is also its identity in caches.
pub const SEMANTIC_EMBEDDING_MODEL_NAME: &str = "bekko-embedding-v1-a8m";

/// Hugging Face repository the semantic embedding model is published in.
pub const SEMANTIC_EMBEDDING_REPO_ID: &str = "hotchpotch/bekko-embedding-v1-a8m";

/// The pinned repository revision the semantic embedding model is read from.
pub const SEMANTIC_EMBEDDING_REVISION: &str = "c721113d59a1d91b447450324f51c4b3332c924a";

/// The encoder configuration file.
pub const SEMANTIC_EMBEDDING_CONFIG_FILE: &str = "config.json";

/// The encoder weights file.
pub const SEMANTIC_EMBEDDING_WEIGHTS_FILE: &str = "model.safetensors";

/// The tokenizer file.
pub const SEMANTIC_EMBEDDING_TOKENIZER_FILE: &str = "tokenizer.json";

/// The three files at the pinned revision, as the shared models folder holds
/// and checks them.
pub const SEMANTIC_EMBEDDING_FILES: [PinnedModelFile; 3] = [
    PinnedModelFile {
        name: SEMANTIC_EMBEDDING_CONFIG_FILE,
        bytes: 1_207,
        sha256: "950f344b034bb96a6ac3da045b1cd5ba037c08d8e4978471d407324622f8878e",
    },
    PinnedModelFile {
        name: SEMANTIC_EMBEDDING_WEIGHTS_FILE,
        bytes: 211_952_896,
        sha256: "5db74812b69eb6c17845f5ecaab533e6e0816877fcf9639a25dd187862d0e885",
    },
    PinnedModelFile {
        name: SEMANTIC_EMBEDDING_TOKENIZER_FILE,
        bytes: 34_363_442,
        sha256: "8bd47075711f75a143d1b78e01a41cc65c1c591b00d3cfeffc23db07adce1392",
    },
];

/// Directory a packaged build ships the three model files in.
const SEMANTIC_MODEL_DIR_ENV: &str = "CORTEX_SEMANTIC_MODEL_DIR";

/// Local paths of the semantic embedding model's files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticModelFiles {
    /// The encoder configuration (`config.json`).
    pub config: PathBuf,
    /// The encoder weights (`model.safetensors`).
    pub weights: PathBuf,
    /// The tokenizer (`tokenizer.json`).
    pub tokenizer: PathBuf,
}
