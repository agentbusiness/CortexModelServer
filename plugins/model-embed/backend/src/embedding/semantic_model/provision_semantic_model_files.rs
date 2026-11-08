use embed_common::utils::error::{Error, Result};

use super::super::update_embedding_progress;
use super::{
    SEMANTIC_EMBEDDING_CONFIG_FILE, SEMANTIC_EMBEDDING_FILES, SEMANTIC_EMBEDDING_REPO_ID,
    SEMANTIC_EMBEDDING_REVISION, SEMANTIC_EMBEDDING_TOKENIZER_FILE,
    SEMANTIC_EMBEDDING_WEIGHTS_FILE, SemanticModelFiles, local_semantic_model_files,
};
use crate::shared_models::{fetch_shared_model_file, shared_model_dir};

/// The semantic embedding model's files, downloading the pinned revision into
/// the shared models folder when this machine does not hold them yet. Each
/// file is checked against its pinned size and SHA-256 before it is moved
/// into place; a file another Cortex app already fetched is used as is.
///
/// # Errors
///
/// Returns an error when the files are not local and cannot be downloaded.
pub fn provision_semantic_model_files() -> Result<SemanticModelFiles> {
    if let Some(files) = local_semantic_model_files() {
        return Ok(files);
    }
    for file in &SEMANTIC_EMBEDDING_FILES {
        update_embedding_progress(format!(
            "downloading {} from {SEMANTIC_EMBEDDING_REPO_ID}",
            file.name
        ));
        let url = format!(
            "https://huggingface.co/{SEMANTIC_EMBEDDING_REPO_ID}/resolve/{SEMANTIC_EMBEDDING_REVISION}/{}",
            file.name
        );
        fetch_shared_model_file(
            SEMANTIC_EMBEDDING_REPO_ID,
            SEMANTIC_EMBEDDING_REVISION,
            file,
            |sink| {
                let response = ureq::get(&url)
                    .call()
                    .map_err(|err| format!("cannot reach {url}: {err}"))?;
                std::io::copy(&mut response.into_body().into_reader(), sink)
                    .map(|_| ())
                    .map_err(|err| format!("download of {} stopped: {err}", file.name))
            },
        )
        .map_err(|err| {
            Error::Internal(format!(
                "Failed to download '{}' from '{SEMANTIC_EMBEDDING_REPO_ID}' at revision {SEMANTIC_EMBEDDING_REVISION}: {err}",
                file.name
            ))
        })?;
    }
    update_embedding_progress(format!(
        "downloaded model files for {SEMANTIC_EMBEDDING_REPO_ID}"
    ));
    let directory = shared_model_dir(SEMANTIC_EMBEDDING_REPO_ID, SEMANTIC_EMBEDDING_REVISION);
    Ok(SemanticModelFiles {
        config: directory.join(SEMANTIC_EMBEDDING_CONFIG_FILE),
        weights: directory.join(SEMANTIC_EMBEDDING_WEIGHTS_FILE),
        tokenizer: directory.join(SEMANTIC_EMBEDDING_TOKENIZER_FILE),
    })
}
