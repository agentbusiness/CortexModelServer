//! The variants (Gemma 4 E4B / E2B, the E4B report lane, Qwen3.8-27B Q4 / Q3; CortexModelServer
//! patch), their pinned files, and getting them onto disk in the shared models
//! folder (`cc_shared_models`): with the revision's download lock held, a file already there at
//! its pinned size (another app may have just fetched it) is used when its SHA-256 matches, else it
//! is downloaded with resume to `<file>.part`, checked, and renamed into place. Every check is
//! recorded beside the file in `<file>.digest`.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy)]
pub struct FileSpec {
    pub name: &'static str,
    pub bytes: u64,
    pub sha256: &'static str,
    // CortexModelServer patch: `(repo, revision)` of a file that is not in the variant's own
    // repository (Qwen's DFlash 2 drafter is z-lab's).
    pub from: Option<(&'static str, &'static str)>,
}

// CortexModelServer patch: how llama-server is started for a variant, each the exact command line
// of the app it comes from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Profile {
    /// CortexTerm's / the knowledge app's Gemma 4: MTP drafts, one unified KV pool, two sliding-window
    /// checkpoints 2048 apart, a 1 GB RAM prompt cache, no cache reuse, no context shift.
    GemmaMtp,
    /// CortexCode's Linux Qwen3.8-27B: DFlash 2 drafts, the vision projector, ctx per slot, and
    /// llama-server's own cache defaults.
    QwenDflash,
}

impl Profile {
    pub fn kv_unified(self) -> bool {
        self == Profile::GemmaMtp
    }

    pub fn speculative(self) -> &'static str {
        match self {
            Profile::GemmaMtp => "draft-mtp",
            Profile::QwenDflash => "draft-dflash",
        }
    }
}

// CortexModelServer patch: a variant's own server settings (`config::Settings`), as its origin
// runs it.
#[derive(Debug, Clone, Copy)]
pub struct Defaults {
    pub ctx: u32,
    pub parallel: u32,
    pub ubatch: u32,
    pub kv_type: &'static str,
    pub draft_max: u32,
    pub cache_ram: Option<u32>,
    pub checkpoints: Option<u32>,
    pub max_tokens: Option<u32>,
    pub greedy: bool,
}

/// CortexTerm's model-llama and the knowledge app's office lane (`model-worker/src/llama/serve.rs`).
const OFFICE: Defaults = Defaults {
    ctx: 32768,
    parallel: 2,
    ubatch: 2048,
    kv_type: "f16",
    draft_max: 4,
    cache_ram: Some(1024),
    checkpoints: Some(2),
    max_tokens: None,
    greedy: false,
};

#[derive(Debug)]
pub struct Spec {
    pub id: &'static str,
    pub label: &'static str,
    pub tier: &'static str,
    pub repo: &'static str,
    pub revision: &'static str,
    pub quant: &'static str,
    pub weights: FileSpec,
    /// The draft model (`--spec-draft-model`): Gemma's MTP head, Qwen's DFlash 2 drafter.
    pub drafter: FileSpec,
    pub notes: &'static str,
    // CortexModelServer patch: the fields below.
    pub profile: Profile,
    pub defaults: Defaults,
    /// The vision projector (`--mmproj`): image input.
    pub mmproj: Option<FileSpec>,
    /// The projector stays on the GPU (else `--no-mmproj-offload`).
    pub mmproj_on_gpu: bool,
    pub family: &'static str,
    /// The mesh tag the model is lent under (`office`, `coding`).
    pub tag: &'static str,
    /// The trained context, reported until the server says.
    pub ctx_native: u32,
}

const E4B_WEIGHTS: FileSpec = FileSpec {
    name: "gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf",
    bytes: 4_215_695_776,
    sha256: "df0fd4ee07072c607c29a0a1cb4f98918426cca12f45a2776bdd6ee6d09a4de3",
    from: None,
};

const E4B_DRAFTER: FileSpec = FileSpec {
    name: "mtp-gemma-4-E4B-it.gguf",
    bytes: 59_678_016,
    sha256: "423074e537504b4f9ec5eafed5c639fac82c96631626efccacdd3c4039b20605",
    from: None,
};

pub const E4B: Spec = Spec {
    id: "gemma-4-e4b-qat",
    label: "Gemma 4 E4B (QAT, UD-Q4_K_XL, MTP drafts)",
    tier: "e4b",
    repo: "unsloth/gemma-4-E4B-it-qat-GGUF",
    revision: "8c5a9e4fd5482e2be20fe0bf013b4c262a8f4265",
    quant: "UD-Q4_K_XL",
    weights: E4B_WEIGHTS,
    drafter: E4B_DRAFTER,
    notes: "The default above 8 GB of memory (4.2 GB of weights, memory-mapped).",
    profile: Profile::GemmaMtp,
    defaults: OFFICE,
    mmproj: None,
    mmproj_on_gpu: false,
    family: "gemma4",
    tag: "office",
    ctx_native: 131_072,
};

pub const E2B: Spec = Spec {
    id: "gemma-4-e2b-qat",
    label: "Gemma 4 E2B (QAT, UD-Q4_K_XL, MTP drafts)",
    tier: "e2b",
    repo: "unsloth/gemma-4-E2B-it-qat-GGUF",
    revision: "66a399f68ddd113b06dff02fca9523e55465d11d",
    quant: "UD-Q4_K_XL",
    weights: FileSpec {
        name: "gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf",
        bytes: 2_620_370_976,
        sha256: "e531007218dfab990486a5de7676a6932d6ea8dea233d1f698d7c21cf8a16889",
        from: None,
    },
    drafter: FileSpec {
        name: "mtp-gemma-4-E2B-it.gguf",
        bytes: 59_235_648,
        sha256: "586f2460b909008640981ec34060aa864e03c144fbabfb3173c4335087e4aae0",
        from: None,
    },
    notes: "The default on 8 GB Macs (2.6 GB of weights, memory-mapped).",
    profile: Profile::GemmaMtp,
    defaults: OFFICE,
    mmproj: None,
    mmproj_on_gpu: false,
    family: "gemma4",
    tag: "office",
    ctx_native: 131_072,
};

// CortexModelServer patch: the knowledge app's report lane (its service crate, `src/local_report_route/
// report_lane.rs`, `model-worker/src/llama/serve.rs` and `generate_thinking.rs`): the same E4B
// files, one process with one slot of the trained context (131,072), thinking per request
// (llama-server's `--reasoning auto`, each call's `enable_thinking` and budget), the answer capped at
// 16,384 tokens, greedy.
pub const E4B_REPORT: Spec = Spec {
    id: "gemma-4-e4b-qat-report",
    label: "Gemma 4 E4B (QAT, UD-Q4_K_XL, MTP drafts), report lane 131k",
    tier: "e4b",
    repo: "unsloth/gemma-4-E4B-it-qat-GGUF",
    revision: "8c5a9e4fd5482e2be20fe0bf013b4c262a8f4265",
    quant: "UD-Q4_K_XL",
    weights: E4B_WEIGHTS,
    drafter: E4B_DRAFTER,
    notes: "The knowledge app's report lane: one slot of 131,072 tokens, answers capped at 16,384, greedy.",
    profile: Profile::GemmaMtp,
    defaults: Defaults { ctx: 131_072, parallel: 1, max_tokens: Some(16_384), greedy: true, ..OFFICE },
    mmproj: None,
    mmproj_on_gpu: false,
    family: "gemma4",
    tag: "office",
    ctx_native: 131_072,
};

// CortexModelServer patch: CortexCode's Linux Qwen (its `plugins/model-llama`, catalog.rs and
// config.rs there): unsloth UD quants with the F16 vision projector, z-lab's DFlash 2 drafter at
// most 7 tokens ahead, 2 slots of 32k each, ubatch 512, q8_0 KV with flash attention.
const QWEN_REPO: &str = "unsloth/Qwen3.8-27B-GGUF";
const QWEN_REVISION: &str = "4ca720788d1e01f1bff70c033e0d0028fd02e502";

const QWEN: Defaults = Defaults {
    ctx: 32768,
    parallel: 2,
    ubatch: 512,
    kv_type: "q8_0",
    draft_max: 7,
    cache_ram: None,
    checkpoints: None,
    max_tokens: None,
    greedy: false,
};

const DFLASH2: FileSpec = FileSpec {
    name: "Qwen3.8-27B-DFlash2-Q4_K_M.gguf",
    bytes: 1_143_006_816,
    sha256: "1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd",
    from: Some(("z-lab/Qwen3.8-27B-DFlash2-GGUF", "2d9571f8ce46e151f61c6499c99dee6079e1d610")),
};

const QWEN_MMPROJ: FileSpec = FileSpec {
    name: "mmproj-F16.gguf",
    bytes: 927_607_488,
    sha256: "cbb841a9ee0636b2ec172f5bb8df2ea8dfeb01e90fe7c6126581d662a0b4e43e",
    from: None,
};

pub const QWEN_Q4: Spec = Spec {
    id: "qwen3.8-27b-q4",
    label: "Qwen3.8-27B (UD-Q4_K_XL, DFlash 2 drafts)",
    tier: "27b",
    repo: QWEN_REPO,
    revision: QWEN_REVISION,
    quant: "UD-Q4_K_XL",
    weights: FileSpec {
        name: "Qwen3.8-27B-UD-Q4_K_XL.gguf",
        bytes: 17_559_178_144,
        sha256: "3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e",
        from: None,
    },
    drafter: DFLASH2,
    notes: "CortexCode's Linux Qwen on a GPU with 24 GB or more (17.6 GB of weights).",
    profile: Profile::QwenDflash,
    defaults: QWEN,
    mmproj: Some(QWEN_MMPROJ),
    mmproj_on_gpu: true,
    family: "qwen3.8",
    tag: "coding",
    ctx_native: 262_144,
};

pub const QWEN_Q3: Spec = Spec {
    id: "qwen3.8-27b-q3",
    label: "Qwen3.8-27B (UD-Q3_K_XL, DFlash 2 drafts)",
    tier: "27b",
    repo: QWEN_REPO,
    revision: QWEN_REVISION,
    quant: "UD-Q3_K_XL",
    weights: FileSpec {
        name: "Qwen3.8-27B-UD-Q3_K_XL.gguf",
        bytes: 13_146_393_504,
        sha256: "8c2a45ff85e7674ca185ec8eb6cdeab0e617ed9d8018caed0b64380eb2a67a5e",
        from: None,
    },
    drafter: DFLASH2,
    notes: "CortexCode's Linux Qwen on a 16 GB GPU (13.1 GB of weights; the vision projector runs on the CPU).",
    profile: Profile::QwenDflash,
    defaults: QWEN,
    mmproj: Some(QWEN_MMPROJ),
    mmproj_on_gpu: false,
    family: "qwen3.8",
    tag: "coding",
    ctx_native: 262_144,
};

// CortexModelServer patch: the report lane and the two Qwen variants added.
pub const CATALOG: [&Spec; 5] = [&E4B, &E2B, &E4B_REPORT, &QWEN_Q4, &QWEN_Q3];

pub fn find(id: &str) -> Option<&'static Spec> {
    CATALOG.into_iter().find(|spec| spec.id == id)
}

/// The variant for this machine: E4B above 8 GiB of memory, E2B at 8 GiB or less.
pub fn for_memory(physical_bytes: u64) -> &'static Spec {
    if physical_bytes > 8 << 30 { &E4B } else { &E2B }
}

impl Spec {
    /// `<models>/<org>-<repo>/<revision>`.
    pub fn dir(&self, models: &Path) -> PathBuf {
        models.join(self.repo.replace('/', "-")).join(self.revision)
    }

    // CortexModelServer patch: a file's own folder (its repository's revision) and path.
    pub fn file_dir(&self, models: &Path, file: &FileSpec) -> PathBuf {
        match file.from {
            Some((repo, revision)) => models.join(repo.replace('/', "-")).join(revision),
            None => self.dir(models),
        }
    }

    pub fn path(&self, models: &Path, file: &FileSpec) -> PathBuf {
        self.file_dir(models, file).join(file.name)
    }

    // CortexModelServer patch: the projector, when the variant has one.
    pub fn files(&self) -> Vec<FileSpec> {
        let mut files = vec![self.weights, self.drafter];
        files.extend(self.mmproj);
        files
    }

    pub fn model_id(&self) -> String {
        format!("{}@{}:{}", self.repo, &self.revision[..7], self.quant)
    }

    fn url(&self, file: &FileSpec) -> String {
        let (repo, revision) = file.from.unwrap_or((self.repo, self.revision));
        format!("https://huggingface.co/{repo}/resolve/{revision}/{}", file.name)
    }
}

#[derive(Serialize, Deserialize, PartialEq)]
struct DigestRecord {
    bytes: u64,
    modified_ns: u128,
    sha256: String,
}

fn modified_ns(path: &Path) -> Option<u128> {
    std::fs::metadata(path).ok()?.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok().map(|d| d.as_nanos())
}

fn digest_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".digest");
    path.with_file_name(name)
}

/// Whether `path` is the pinned file by its recorded check (no hashing).
fn recorded_ok(path: &Path, file: &FileSpec) -> bool {
    let Ok(meta) = std::fs::metadata(path) else { return false };
    if meta.len() != file.bytes {
        return false;
    }
    let Some(record) = std::fs::read(digest_path(path)).ok().and_then(|b| serde_json::from_slice::<DigestRecord>(&b).ok()) else {
        return false;
    };
    record.bytes == file.bytes && Some(record.modified_ns) == modified_ns(path) && record.sha256 == file.sha256
}

fn record(path: &Path, sha256: &str) {
    if let (Ok(meta), Some(modified)) = (std::fs::metadata(path), modified_ns(path)) {
        let record = DigestRecord { bytes: meta.len(), modified_ns: modified, sha256: sha256.to_owned() };
        if let Ok(text) = serde_json::to_vec_pretty(&record) {
            let _ = std::fs::write(digest_path(path), text);
        }
    }
}

fn sha256_file(path: &Path, on_progress: &mut dyn FnMut(u64)) -> Result<String, String> {
    let mut input = std::fs::File::open(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 8 << 20];
    let mut done = 0u64;
    let mut last = Instant::now();
    loop {
        let read = input.read(&mut buffer).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        done += read as u64;
        if last.elapsed() >= Duration::from_millis(500) {
            on_progress(done);
            last = Instant::now();
        }
    }
    Ok(hasher.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Whether every file of the variant is in place and checked (by record, no hashing).
pub fn installed(spec: &Spec, models: &Path) -> bool {
    // CortexModelServer patch: each file in its own repository's folder.
    spec.files().iter().all(|file| recorded_ok(&spec.path(models, file), file))
}

/// `model.install` progress.
#[derive(Debug, Clone, Serialize)]
pub struct InstallProgress {
    pub variant: String,
    /// verify | download | done | failed
    pub phase: String,
    pub file: String,
    pub bytes: u64,
    pub total: u64,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub detail: String,
}

pub type Report = std::sync::Arc<dyn Fn(InstallProgress) + Send + Sync>;

/// Make the variant present and checked under `models`. Returns its directory.
pub async fn install(spec: &'static Spec, models: PathBuf, report: Report) -> Result<PathBuf, String> {
    // CortexModelServer patch: each file's own folder is locked while it is checked or fetched (the
    // drafter of a Qwen variant is in another repository), as CortexCode's model-llama does.
    let total: u64 = spec.files().iter().map(|f| f.bytes).sum();
    let mut before = 0u64;
    for file in spec.files() {
        let dir = spec.file_dir(&models, &file);
        // Another app downloading the same revision holds the lock: wait for it, then look again.
        let failed = |e: std::io::Error| format!("cannot lock {}: {e}", dir.join(cc_shared_models::LOCK).display());
        let _lock = match cc_shared_models::try_lock(&dir).map_err(failed)? {
            Some(lock) => lock,
            None => {
                let detail = "waiting while another Cortex app downloads this model".to_owned();
                report(InstallProgress { variant: spec.id.into(), phase: "download".into(), file: file.name.into(), bytes: before, total, detail });
                let locked = dir.clone();
                tokio::task::spawn_blocking(move || cc_shared_models::lock(&locked)).await.map_err(|e| e.to_string())?.map_err(failed)?
            }
        };
        let target = dir.join(file.name);
        let say = {
            let report = report.clone();
            move |phase: &str, done: u64, detail: String| {
                report(InstallProgress {
                    variant: spec.id.into(),
                    phase: phase.into(),
                    file: file.name.into(),
                    bytes: before + done,
                    total,
                    detail,
                })
            }
        };
        if recorded_ok(&target, &file) {
            before += file.bytes;
            continue;
        }
        if std::fs::metadata(&target).is_ok_and(|m| m.len() == file.bytes) {
            if verify(&target, &say).await? == file.sha256 {
                record(&target, file.sha256);
                before += file.bytes;
                continue;
            }
            tracing::warn!(file = %target.display(), "a model file does not match its pinned digest; downloading it again");
            let _ = std::fs::remove_file(&target);
        }
        let part = dir.join(format!("{}.part", file.name));
        download(&spec.url(&file), &file, &part, &say).await?;
        let sha = verify(&part, &say).await?;
        if sha != file.sha256 {
            let _ = std::fs::remove_file(&part);
            let message = format!("{} has sha256 {sha}, expected {}", part.display(), file.sha256);
            say("failed", 0, message.clone());
            return Err(message);
        }
        std::fs::rename(&part, &target).map_err(|e| format!("cannot move {} into place: {e}", part.display()))?;
        record(&target, &sha);
        before += file.bytes;
    }
    let dir = spec.dir(&models);
    report(InstallProgress { variant: spec.id.into(), phase: "done".into(), file: String::new(), bytes: total, total, detail: dir.display().to_string() });
    Ok(dir)
}

/// The SHA-256 of `path`, hashed on a blocking thread.
async fn verify(path: &Path, say: &(impl Fn(&str, u64, String) + Clone + Send + Sync + 'static)) -> Result<String, String> {
    let (path, say) = (path.to_path_buf(), say.clone());
    tokio::task::spawn_blocking(move || sha256_file(&path, &mut |done| say("verify", done, String::new()))).await.map_err(|e| e.to_string())?
}

/// Fetch `file` into `part`, resuming what an earlier attempt left there.
async fn download(url: &str, file: &FileSpec, part: &Path, say: &(dyn Fn(&str, u64, String) + Send + Sync)) -> Result<(), String> {
    use futures_util::StreamExt;
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .read_timeout(Duration::from_secs(120))
        .build()
        .map_err(|e| format!("http client: {e}"))?;
    let mut attempts = 0u32;
    loop {
        attempts += 1;
        let have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
        if have > file.bytes {
            let _ = std::fs::remove_file(&part);
            continue;
        }
        if have == file.bytes {
            break;
        }
        let mut request = client.get(url);
        if have > 0 {
            request = request.header(reqwest::header::RANGE, format!("bytes={have}-"));
        }
        let outcome: Result<(), String> = async {
            let response = request.send().await.map_err(|e| format!("download request failed: {e}"))?;
            let status = response.status();
            if !status.is_success() {
                return Err(format!("download of {url} returned HTTP {status}"));
            }
            let resumed = status == reqwest::StatusCode::PARTIAL_CONTENT;
            let mut out = std::fs::OpenOptions::new()
                .create(true)
                .append(resumed)
                .write(true)
                .truncate(!resumed)
                .open(&part)
                .map_err(|e| format!("cannot write {}: {e}", part.display()))?;
            let mut written = if resumed { have } else { 0 };
            let mut stream = response.bytes_stream();
            let mut last = Instant::now();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|e| format!("download interrupted: {e}"))?;
                if chunk.len() as u64 > file.bytes - written {
                    return Err("the server sent more than the pinned file size".into());
                }
                out.write_all(&chunk).map_err(|e| format!("cannot write {}: {e}", part.display()))?;
                written += chunk.len() as u64;
                if last.elapsed() >= Duration::from_secs(1) {
                    say("download", written, String::new());
                    last = Instant::now();
                }
            }
            out.flush().map_err(|e| e.to_string())?;
            if written != file.bytes {
                return Err(format!("download ended at {written} of {} bytes", file.bytes));
            }
            Ok(())
        }
        .await;
        match outcome {
            Ok(()) => {}
            Err(error) if attempts < 20 => {
                say("download", have, format!("retrying after: {error}"));
                tokio::time::sleep(Duration::from_secs(u64::from(attempts.min(10)) * 2)).await;
            }
            Err(error) => return Err(error),
        }
    }
    Ok(())
}
