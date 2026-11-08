//! ONNX embedding model configuration.

use std::path::PathBuf;

/// Where an ONNX embedding model's files come from.
///
/// Use [`Local`](Self::Local) for your own ONNX models already on disk, or
/// [`HuggingFace`](Self::HuggingFace) for any HuggingFace-hosted model.
///
/// # Examples
///
/// ```no_run
/// use crate::embedding::{EmbeddingModelConfig, OnnxEmbeddingModel};
///
/// # fn main() -> embed_common::utils::error::Result<()> {
/// // Local: use your own ONNX model files
/// let model = OnnxEmbeddingModel::from_config(EmbeddingModelConfig::Local {
///     model_path: "path/to/model.onnx".into(),
///     tokenizer_path: "path/to/tokenizer.json".into(),
/// })?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum EmbeddingModelConfig {
    /// Custom model from a HuggingFace repository.
    HuggingFace {
        /// Repository ID (e.g., `"org/model"`).
        repo_id: String,
        /// Path within the repo to the ONNX model file (e.g., `"onnx/model.onnx"`).
        model_file: String,
        /// Path within the repo to the tokenizer file (e.g., `"tokenizer.json"`).
        tokenizer_file: String,
    },
    /// Custom model from local file paths.
    Local {
        /// Path to the `.onnx` model file.
        model_path: PathBuf,
        /// Path to the `tokenizer.json` file.
        tokenizer_path: PathBuf,
    },
}

impl EmbeddingModelConfig {
    /// Returns a human-readable name for this model configuration.
    ///
    /// This name is used as the registry key when an engine database loads the model.
    #[must_use]
    pub fn display_name(&self) -> String {
        match self {
            Self::HuggingFace { repo_id, .. } => repo_id.clone(),
            Self::Local {
                model_path: path, ..
            } => path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("custom-model")
                .to_string(),
        }
    }

    /// Returns the HuggingFace repo ID and file paths for download resolution.
    pub(crate) fn resolve_info(&self) -> ResolveInfo<'_> {
        match self {
            Self::HuggingFace {
                repo_id,
                model_file,
                tokenizer_file,
            } => ResolveInfo::Hub {
                repo_id,
                model_file,
                tokenizer_file,
            },
            Self::Local {
                model_path,
                tokenizer_path,
            } => ResolveInfo::Local {
                model_path,
                tokenizer_path,
            },
        }
    }
}

/// Internal enum for download resolution dispatch.
pub(crate) enum ResolveInfo<'a> {
    Hub {
        repo_id: &'a str,
        model_file: &'a str,
        tokenizer_file: &'a str,
    },
    Local {
        model_path: &'a PathBuf,
        tokenizer_path: &'a PathBuf,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn huggingface_display_name_is_repo_id() {
        let config = EmbeddingModelConfig::HuggingFace {
            repo_id: "org/my-model".into(),
            model_file: "model.onnx".into(),
            tokenizer_file: "tokenizer.json".into(),
        };
        assert_eq!(config.display_name(), "org/my-model");
    }

    #[test]
    fn local_display_name_from_file_stem() {
        let config = EmbeddingModelConfig::Local {
            model_path: "/path/to/my-model.onnx".into(),
            tokenizer_path: "/path/to/tokenizer.json".into(),
        };
        assert_eq!(config.display_name(), "my-model");
    }
}
