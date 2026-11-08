//! GLiNER-relex's files in the models folder every Cortex app shares (`~/.cortex-models`, laid out
//! `<org>-<repo>/<revision>/<file>`, `cc-shared-models`): the pinned revision the knowledge app runs, each file
//! at its pinned size and SHA-256 (the knowledge app's `local_models/catalog.rs`).
//!
//! A missing file is downloaded from Hugging Face at the pinned revision with the revision folder's
//! `.download.lock` held (another app or another lane of this engine wanting the same file waits,
//! then finds it), into `<file>.part`, hashed on the way, and renamed into place only when its size
//! and SHA-256 match. Files are only read once in place: lanes share them.

use futures_util::StreamExt;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tokio::io::AsyncWriteExt;

pub const REPO: &str = "knowledgator/gliner-relex-large-v1.0";
pub const REVISION: &str = "4aedc9226a5ac9e2f6b5ea3e91c1ee577c88a290";

pub struct Pinned {
    pub name: &'static str,
    pub bytes: u64,
    pub sha256: &'static str,
}

/// The files the relex lane reads (`gliner_config.json`, `model.safetensors`, `tokenizer.json`).
pub const FILES: [Pinned; 3] = [
    Pinned { name: "gliner_config.json", bytes: 3_565, sha256: "009031313c27dbdc255d4e9d85bdfb89f17259ccf4c82e8982ac763c249c1cdb" },
    Pinned { name: "model.safetensors", bytes: 1_866_369_448, sha256: "7c5bd751e1b24e4254d70fe4355a986cd65400676ce3735f7752429fcc26960a" },
    Pinned { name: "tokenizer.json", bytes: 8_649_416, sha256: "d9f64f3126faa24e3ef6bdc47e7120eef528a49cd8b447bdd6865bcf2043cd65" },
];

pub fn total_bytes() -> u64 {
    FILES.iter().map(|file| file.bytes).sum()
}

/// `<models>/<org>-<repo>/<revision>`.
pub fn model_dir(models: &Path) -> PathBuf {
    cc_shared_models::revision_dir(models, REPO, REVISION)
}

/// Every file is in place at its pinned size.
pub fn installed(dir: &Path) -> bool {
    FILES.iter().all(|file| std::fs::metadata(dir.join(file.name)).is_ok_and(|meta| meta.is_file() && meta.len() == file.bytes))
}

/// Where an install stands (for `model.info` and the probe).
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct Progress {
    pub phase: String,
    pub file: String,
    pub bytes: u64,
    pub total: u64,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub error: String,
}

/// Download what is missing. `progress` follows it.
pub async fn install(models: &Path, progress: &Mutex<Option<Progress>>) -> Result<(), String> {
    let dir = model_dir(models);
    let set = |p: Progress| {
        if let Ok(mut slot) = progress.lock() {
            *slot = Some(p);
        }
    };
    cc_shared_models::ensure_root(models).map_err(|e| format!("cannot create {}: {e}", models.display()))?;
    let lock_dir = dir.clone();
    set(Progress { phase: "waiting for the download lock".into(), total: total_bytes(), ..Default::default() });
    let _lock = tokio::task::spawn_blocking(move || cc_shared_models::lock(&lock_dir))
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| format!("cannot lock {}: {e}", dir.display()))?;
    let client = reqwest::Client::builder().build().map_err(|e| e.to_string())?;
    let mut done = 0u64;
    for file in &FILES {
        let target = dir.join(file.name);
        // Looked at again with the lock held: another app may have just fetched it.
        if std::fs::metadata(&target).is_ok_and(|meta| meta.is_file() && meta.len() == file.bytes) {
            done += file.bytes;
            continue;
        }
        let url = format!("https://huggingface.co/{REPO}/resolve/{REVISION}/{}", file.name);
        let part = dir.join(format!("{}.part", file.name));
        let response = client.get(&url).send().await.and_then(|r| r.error_for_status()).map_err(|e| format!("cannot reach {url}: {e}"))?;
        let mut out = tokio::fs::File::create(&part).await.map_err(|e| format!("cannot write {}: {e}", part.display()))?;
        let mut hasher = Sha256::new();
        let mut written = 0u64;
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| format!("download of {} stopped: {e}", file.name))?;
            hasher.update(&chunk);
            out.write_all(&chunk).await.map_err(|e| format!("cannot write {}: {e}", part.display()))?;
            written += chunk.len() as u64;
            set(Progress { phase: "downloading".into(), file: file.name.into(), bytes: done + written, total: total_bytes(), error: String::new() });
        }
        out.flush().await.map_err(|e| e.to_string())?;
        drop(out);
        let digest: String = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect();
        if written != file.bytes || digest != file.sha256 {
            let _ = std::fs::remove_file(&part);
            return Err(format!("{} does not match its pin ({written} bytes, sha256 {digest})", file.name));
        }
        std::fs::rename(&part, &target).map_err(|e| format!("cannot move {} into place: {e}", part.display()))?;
        done += file.bytes;
        tracing::info!(file = file.name, bytes = written, "model file downloaded");
    }
    set(Progress { phase: "installed".into(), bytes: done, total: total_bytes(), ..Default::default() });
    Ok(())
}
