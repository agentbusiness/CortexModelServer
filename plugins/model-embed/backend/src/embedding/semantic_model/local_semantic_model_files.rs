use std::path::Path;

use super::{
    SEMANTIC_EMBEDDING_CONFIG_FILE, SEMANTIC_EMBEDDING_FILES, SEMANTIC_EMBEDDING_REPO_ID,
    SEMANTIC_EMBEDDING_REVISION, SEMANTIC_EMBEDDING_TOKENIZER_FILE,
    SEMANTIC_EMBEDDING_WEIGHTS_FILE, SEMANTIC_MODEL_DIR_ENV, SemanticModelFiles,
};
use crate::shared_models::shared_model_dir;

/// The semantic embedding model's files when they are already on this machine.
///
/// A packaged build's directory (named by `CORTEX_SEMANTIC_MODEL_DIR`) comes
/// first; otherwise the pinned revision's folder in the shared models folder,
/// each file at its pinned size. `None` when neither holds all three files.
pub fn local_semantic_model_files() -> Option<SemanticModelFiles> {
    if let Some(dir) = std::env::var_os(SEMANTIC_MODEL_DIR_ENV).filter(|dir| !dir.is_empty())
        && let Some(files) = files_in(Path::new(&dir))
    {
        return Some(files);
    }
    let shared = shared_model_dir(SEMANTIC_EMBEDDING_REPO_ID, SEMANTIC_EMBEDDING_REVISION);
    let pinned = SEMANTIC_EMBEDDING_FILES.iter().all(|file| {
        std::fs::metadata(shared.join(file.name))
            .is_ok_and(|meta| meta.is_file() && meta.len() == file.bytes)
    });
    if pinned { files_in(&shared) } else { None }
}

fn files_in(dir: &Path) -> Option<SemanticModelFiles> {
    let files = SemanticModelFiles {
        config: dir.join(SEMANTIC_EMBEDDING_CONFIG_FILE),
        weights: dir.join(SEMANTIC_EMBEDDING_WEIGHTS_FILE),
        tokenizer: dir.join(SEMANTIC_EMBEDDING_TOKENIZER_FILE),
    };
    (files.config.is_file() && files.weights.is_file() && files.tokenizer.is_file())
        .then_some(files)
}
