//! What the copied embedders (`embedding/semantic_model`) call `crate::shared_models` in the knowledge app's
//! adapters: the same names, on this repository's `cc-shared-models` (`~/.cortex-models`, laid out
//! `<org>-<repo>/<revision>/<file>`, the same `.download.lock`). `PinnedModelFile` is the knowledge app's,
//! copied; the folder can be moved with the plugin's `models_dir` setting ([`set_root`]).

use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::OnceLock;

static ROOT: OnceLock<PathBuf> = OnceLock::new();

/// Use `root` instead of `~/.cortex-models` (once, before the first model lookup).
pub fn set_root(root: PathBuf) {
    let _ = ROOT.set(root);
}

/// The shared models folder.
pub fn cortex_models_dir() -> PathBuf {
    ROOT.get().cloned().unwrap_or_else(cc_shared_models::root)
}

// Copied from the knowledge app's adapters crate, src/shared_models/pinned_model_file.rs.
/// One file of a Hugging Face repository at a pinned revision, as a download
/// checks it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PinnedModelFile {
    /// The file's path in the repository, subfolders included.
    pub name: &'static str,
    /// Its size in bytes.
    pub bytes: u64,
    /// Lower-case hex SHA-256 of its bytes.
    pub sha256: &'static str,
}

/// `<root>/<org>-<repo>/<revision>`.
pub fn shared_model_dir(repository: &str, revision: &str) -> PathBuf {
    cc_shared_models::revision_dir(&cortex_models_dir(), repository, revision)
}

/// The path of `file` of `repository` at `revision`, downloading it first when it is not there at
/// its pinned size: with the revision folder's lock held, `download` writes into `<file>.part`
/// (hashed on the way), renamed into place only when its size and SHA-256 match.
pub fn fetch_shared_model_file(
    repository: &str,
    revision: &str,
    file: &PinnedModelFile,
    download: impl FnOnce(&mut dyn Write) -> Result<(), String>,
) -> Result<PathBuf, String> {
    let root = cortex_models_dir();
    cc_shared_models::ensure_root(&root).map_err(|err| format!("cannot create '{}': {err}", root.display()))?;
    let directory = shared_model_dir(repository, revision);
    let _lock = cc_shared_models::lock(&directory).map_err(|err| format!("cannot lock '{}': {err}", directory.display()))?;
    let target = directory.join(file.name);
    if std::fs::metadata(&target).is_ok_and(|meta| meta.is_file() && meta.len() == file.bytes) {
        return Ok(target);
    }
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent).map_err(|err| format!("cannot create '{}': {err}", parent.display()))?;
    }
    let partial = directory.join(format!("{}.part", file.name));
    let output = File::create(&partial).map_err(|err| format!("cannot write '{}': {err}", partial.display()))?;
    let mut sink = Hashing { output, hasher: Sha256::new(), written: 0 };
    let result = download(&mut sink).and_then(|()| sink.output.flush().map_err(|err| format!("cannot write '{}': {err}", partial.display())));
    let Hashing { output, hasher, written } = sink;
    drop(output);
    let digest: String = hasher.finalize().iter().map(|byte| format!("{byte:02x}")).collect();
    let checked = result.and_then(|()| {
        if written != file.bytes {
            Err(format!("{} has {written} bytes, expected {}", file.name, file.bytes))
        } else if digest != file.sha256 {
            Err(format!("{} does not match its pinned SHA-256", file.name))
        } else {
            Ok(())
        }
    });
    match checked {
        Ok(()) => std::fs::rename(&partial, &target).map(|()| target).map_err(|err| format!("cannot move '{}' into place: {err}", partial.display())),
        Err(err) => {
            let _ = std::fs::remove_file(&partial);
            Err(err)
        }
    }
}

struct Hashing {
    output: File,
    hasher: Sha256,
    written: u64,
}

impl Write for Hashing {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let count = self.output.write(buffer)?;
        self.hasher.update(&buffer[..count]);
        self.written += count as u64;
        Ok(count)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.output.flush()
    }
}
