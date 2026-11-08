//! The model this engine serves and its files in the shared models folder (`~/.cortex-models`,
//! `<org>-<repo>/<revision>/<file>`): `bekko-embedding-v1-a8m`, the embedder every Cortex app
//! uses (the knowledge app's semantic embedder, CortexCode's and CortexTerm's memory, CortexAgent),
//! ModernBERT on candle. Its pinned files and their download are the knowledge app's own
//! (`embedding/semantic_model`).

use crate::embedding::{SEMANTIC_EMBEDDING_FILES, SEMANTIC_EMBEDDING_REPO_ID, SEMANTIC_EMBEDDING_REVISION};
use crate::shared_models::shared_model_dir;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy)]
pub struct Spec {
    /// The `model` setting.
    pub id: &'static str,
    pub repo: &'static str,
    pub revision: &'static str,
    pub dims: u32,
    pub family: &'static str,
}

pub const BEKKO: Spec = Spec {
    id: "bekko-embedding-v1-a8m",
    repo: SEMANTIC_EMBEDDING_REPO_ID,
    revision: SEMANTIC_EMBEDDING_REVISION,
    dims: 384,
    family: "modernbert",
};

pub fn find(id: &str) -> Option<Spec> {
    (id == BEKKO.id).then_some(BEKKO)
}

impl Spec {
    pub fn dir(&self) -> PathBuf {
        shared_model_dir(self.repo, self.revision)
    }

    /// `<repo>@<revision[..7]>`.
    pub fn model_id(&self) -> String {
        format!("{}@{}", self.repo, &self.revision[..7])
    }

    /// Bytes of every file.
    pub fn bytes(&self) -> u64 {
        SEMANTIC_EMBEDDING_FILES.iter().map(|file| file.bytes).sum()
    }

    /// The weights' bytes (what `model.info` reports as `file_bytes`).
    pub fn weights_bytes(&self) -> u64 {
        SEMANTIC_EMBEDDING_FILES.iter().find(|f| f.name.ends_with(".safetensors")).map_or(0, |f| f.bytes)
    }

    /// Every file is in place at its pinned size.
    pub fn installed(&self) -> bool {
        let dir = self.dir();
        SEMANTIC_EMBEDDING_FILES
            .iter()
            .all(|file| std::fs::metadata(dir.join(file.name)).is_ok_and(|meta| meta.is_file() && meta.len() == file.bytes))
    }

    /// The tokenizer the model reads with.
    pub fn tokenizer(&self) -> PathBuf {
        self.dir().join("tokenizer.json")
    }
}
