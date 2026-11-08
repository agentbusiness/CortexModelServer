//! Model cache management and download from HuggingFace Hub.
//!
//! Downloads ONNX model and tokenizer files on first use and caches them
//! locally. The cache is managed by `hf-hub` and defaults to
//! `~/.cache/huggingface/hub/`. Override with the `HF_HOME` or
//! `HUGGINGFACE_HUB_CACHE` environment variables.

use std::fs;
use std::path::{Path, PathBuf};

use hf_hub::api::sync::ApiBuilder;
use embed_common::utils::error::{Error, Result};

use super::huggingface_hub_cache_dir::huggingface_hub_cache_dir;
use super::model_config::{EmbeddingModelConfig, ResolveInfo};

use super::update_embedding_progress as update_embedder_progress;

/// Resolved local paths to a downloaded (or already local) model.
pub(crate) struct ResolvedModel {
    pub model_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub name: String,
}

fn cached_hub_snapshot_dir_in(cache_dir: &Path, repo_id: &str) -> Option<PathBuf> {
    let repo_dir_name = format!("models--{}", repo_id.replace('/', "--"));
    let snapshots_dir = cache_dir.join(repo_dir_name).join("snapshots");
    let entries = fs::read_dir(&snapshots_dir).ok()?;
    let mut candidates: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_dir())
        .collect();
    candidates.sort();
    candidates.pop()
}

fn cached_hub_snapshot_dir(repo_id: &str) -> Option<PathBuf> {
    cached_hub_snapshot_dir_in(&huggingface_hub_cache_dir(), repo_id)
}

fn cached_hub_file(repo_id: &str, relative_path: &str) -> Option<PathBuf> {
    let snapshot_dir = cached_hub_snapshot_dir(repo_id)?;
    let candidate = snapshot_dir.join(relative_path);
    candidate.exists().then_some(candidate)
}

fn resolve_from_local_cache(
    repo_id: &str,
    model_file: &str,
    tokenizer_file: &str,
    name: String,
) -> Option<ResolvedModel> {
    let model_path = cached_hub_file(repo_id, model_file)?;
    let tokenizer_path = cached_hub_file(repo_id, tokenizer_file)?;
    Some(ResolvedModel {
        model_path,
        tokenizer_path,
        name,
    })
}

/// Resolves model files to local paths, downloading from HuggingFace Hub if necessary.
///
/// For [`Local`](EmbeddingModelConfig::Local) configs, returns the paths directly.
/// For [`HuggingFace`](EmbeddingModelConfig::HuggingFace) configs,
/// downloads the model and tokenizer files from HuggingFace Hub on first use.
/// Subsequent calls return cached paths instantly.
pub(crate) fn resolve(config: &EmbeddingModelConfig) -> Result<ResolvedModel> {
    let name = config.display_name();

    match config.resolve_info() {
        ResolveInfo::Local {
            model_path,
            tokenizer_path,
        } => Ok(ResolvedModel {
            model_path: model_path.clone(),
            tokenizer_path: tokenizer_path.clone(),
            name,
        }),
        ResolveInfo::Hub {
            repo_id,
            model_file,
            tokenizer_file,
        } => {
            update_embedder_progress(format!("checking local Hugging Face cache for {}", repo_id));
            if let Some(resolved) =
                resolve_from_local_cache(repo_id, model_file, tokenizer_file, name.clone())
            {
                update_embedder_progress(format!("using local Hugging Face cache for {}", repo_id));
                return Ok(resolved);
            }

            update_embedder_progress(format!(
                "local cache miss for {}; initializing Hugging Face client",
                repo_id
            ));
            let api = ApiBuilder::new()
                .with_cache_dir(huggingface_hub_cache_dir())
                .build()
                .map_err(|e| {
                    Error::Internal(format!("Failed to initialize HuggingFace Hub client: {e}"))
                })?;

            let repo = api.model(repo_id.to_string());

            update_embedder_progress(format!(
                "downloading model file {} from {}",
                model_file, repo_id
            ));
            let model_path = repo.get(model_file).map_err(|e| {
                Error::Internal(format!(
                    "Failed to download model file '{model_file}' from '{repo_id}': {e}"
                ))
            })?;

            update_embedder_progress(format!(
                "downloading tokenizer file {} from {}",
                tokenizer_file, repo_id
            ));
            let tokenizer_path = repo.get(tokenizer_file).map_err(|e| {
                Error::Internal(format!(
                    "Failed to download tokenizer file '{tokenizer_file}' from '{repo_id}': {e}"
                ))
            })?;

            update_embedder_progress(format!("downloaded model files for {}", repo_id));
            Ok(ResolvedModel {
                model_path,
                tokenizer_path,
                name,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn local_config_resolves_directly() {
        let config = EmbeddingModelConfig::Local {
            model_path: "/some/path/model.onnx".into(),
            tokenizer_path: "/some/path/tokenizer.json".into(),
        };
        let resolved = resolve(&config).unwrap();
        assert_eq!(resolved.model_path, PathBuf::from("/some/path/model.onnx"));
        assert_eq!(
            resolved.tokenizer_path,
            PathBuf::from("/some/path/tokenizer.json")
        );
        assert_eq!(resolved.name, "model");
    }

    #[test]
    fn finds_hub_model_in_local_cache_layout() {
        let temp = tempdir().unwrap();
        let hub_dir = temp.path().join("hub");
        let snapshot_dir = hub_dir
            .join("models--org--model")
            .join("snapshots")
            .join("snapshot-1");
        fs::create_dir_all(snapshot_dir.join("onnx")).unwrap();
        fs::write(snapshot_dir.join("onnx/model.onnx"), b"fake-model").unwrap();
        fs::write(snapshot_dir.join("tokenizer.json"), b"fake-tokenizer").unwrap();

        assert_eq!(
            cached_hub_snapshot_dir_in(&hub_dir, "org/model").unwrap(),
            snapshot_dir
        );
    }
}
