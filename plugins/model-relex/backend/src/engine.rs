//! The engine: one GLiNER-relex worker process (this executable run as `worker`, the knowledge app's relex
//! lane in `relex/` answering JSON lines on stdin/stdout, exactly as the knowledge app's pool drives it),
//! started on the first `model.extract`, ended after `idle_unload_s` without one, so the model's
//! GPU memory goes with the process. One slot: passages run one at a time, the rest queue in
//! arrival order.

use crate::config::Config;
use crate::metrics::Metrics;
use crate::store;
use cc_protocol::contracts::model::{ModelInfo, SlotState, SlotsResult, kinds, methods, tags};
use cc_protocol::{RpcError, codes, to_value};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio_util::sync::CancellationToken;

/// Subtokens per encoder pass (`relex::WINDOW_TOKENS`).
pub const WINDOW_TOKENS: u32 = 768;
/// What `model.info` reports as the engine.
pub const ENGINE_REVISION: &str = "candle 0.11 + model-worker relex @ 4ea0c825 (copied as is)";
const MODEL_NAME: &str = "gliner-relex-large-v1.0";

struct Worker {
    child: Child,
    stdin: ChildStdin,
    stdout: Lines<BufReader<ChildStdout>>,
    device: String,
    load_ms: u64,
}

pub struct Engine {
    config: Config,
    models: PathBuf,
    dir: PathBuf,
    lane: String,
    /// The one slot: holding it is running a passage (or loading the worker).
    worker: tokio::sync::Mutex<Option<Worker>>,
    queued: AtomicU32,
    busy: AtomicBool,
    loaded: AtomicBool,
    next_id: AtomicU64,
    last_used: Mutex<Instant>,
    device: Mutex<String>,
    load_ms: AtomicU64,
    stderr_tail: Arc<Mutex<String>>,
    metrics: Metrics,
    install: Mutex<Option<store::Progress>>,
    installing: AtomicBool,
}

fn now_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

/// The device the worker will pick (the same test as `relex::pick_device`), before it runs.
fn expected_device() -> &'static str {
    if candle_core::utils::metal_is_available() {
        "metal"
    } else if candle_core::utils::cuda_is_available() {
        "cuda"
    } else {
        "cpu"
    }
}

impl Engine {
    pub fn new(config: Config) -> Arc<Self> {
        let models = config.models_dir.as_ref().map(PathBuf::from).unwrap_or_else(cc_shared_models::root);
        let dir = store::model_dir(&models);
        let lane = std::env::var("CORTEX_LANE").unwrap_or_default();
        let engine = Arc::new(Self {
            config,
            models,
            dir,
            lane,
            worker: tokio::sync::Mutex::new(None),
            queued: AtomicU32::new(0),
            busy: AtomicBool::new(false),
            loaded: AtomicBool::new(false),
            next_id: AtomicU64::new(1),
            last_used: Mutex::new(Instant::now()),
            device: Mutex::new(expected_device().to_owned()),
            load_ms: AtomicU64::new(0),
            stderr_tail: Arc::new(Mutex::new(String::new())),
            metrics: Metrics::default(),
            install: Mutex::new(None),
            installing: AtomicBool::new(false),
        });
        engine.clone().watch_idle();
        engine
    }

    pub fn installed(&self) -> bool {
        store::installed(&self.dir)
    }

    pub fn install_progress(&self) -> Option<store::Progress> {
        self.install.lock().ok().and_then(|p| p.clone())
    }

    /// Download the missing files once, in the background.
    pub fn install_in_background(self: &Arc<Self>) {
        if !self.config.download || self.installed() || self.installing.swap(true, Ordering::SeqCst) {
            return;
        }
        let engine = self.clone();
        tokio::spawn(async move {
            if let Err(error) = store::install(&engine.models, &engine.install).await {
                tracing::warn!(%error, "GLiNER-relex was not installed");
                if let Ok(mut slot) = engine.install.lock() {
                    let mut progress = slot.clone().unwrap_or_default();
                    progress.phase = "failed".into();
                    progress.error = error;
                    *slot = Some(progress);
                }
            }
            engine.installing.store(false, Ordering::SeqCst);
        });
    }

    /// End the worker after `idle_unload_s` without a request.
    fn watch_idle(self: Arc<Self>) {
        if self.config.idle_unload_s == 0 {
            return;
        }
        let idle = Duration::from_secs(self.config.idle_unload_s);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(5)).await;
                let quiet = self.last_used.lock().map(|at| at.elapsed() >= idle).unwrap_or(false);
                if !quiet || self.queued.load(Ordering::SeqCst) > 0 {
                    continue;
                }
                let Ok(mut slot) = self.worker.try_lock() else { continue };
                if let Some(mut worker) = slot.take() {
                    let _ = worker.child.kill().await;
                    self.loaded.store(false, Ordering::SeqCst);
                    tracing::info!(idle_s = idle.as_secs(), "GLiNER-relex unloaded after being idle");
                }
            }
        });
    }

    pub async fn shutdown(&self) {
        if let Some(mut worker) = self.worker.lock().await.take() {
            let _ = worker.child.kill().await;
        }
        self.loaded.store(false, Ordering::SeqCst);
    }

    async fn start_worker(&self) -> Result<Worker, String> {
        let exe = std::env::current_exe().map_err(|e| format!("cannot find this executable: {e}"))?;
        let started = Instant::now();
        let mut child = Command::new(exe)
            .arg("worker")
            .arg("--model")
            .arg(&self.dir)
            .env("LAB_RELEX_DTYPE", &self.config.encoder_dtype)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| format!("cannot start the relex worker: {e}"))?;
        let stdin = child.stdin.take().ok_or("no worker stdin")?;
        let stdout = child.stdout.take().ok_or("no worker stdout")?;
        if let Some(stderr) = child.stderr.take() {
            let tail = self.stderr_tail.clone();
            tokio::spawn(async move {
                let mut lines = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    eprintln!("relex worker: {line}");
                    if let Ok(mut last) = tail.lock() {
                        *last = line;
                    }
                }
            });
        }
        let mut stdout = BufReader::new(stdout).lines();
        // The worker loads the model and runs one warm pass (Metal compiles its kernels, 8-13 s)
        // before it says it is ready.
        loop {
            match stdout.next_line().await {
                Ok(Some(line)) => {
                    let Ok(ready) = serde_json::from_str::<Value>(&line) else { continue };
                    if ready["ready"].as_bool() == Some(true) {
                        let device = ready["device"].as_str().unwrap_or("cpu").to_owned();
                        let load_ms = ready["load_ms"].as_u64().unwrap_or_else(|| started.elapsed().as_millis() as u64);
                        tracing::info!(%device, load_ms, "GLiNER-relex loaded");
                        return Ok(Worker { child, stdin, stdout, device, load_ms });
                    }
                }
                Ok(None) | Err(_) => {
                    let _ = child.wait().await;
                    let tail = self.stderr_tail.lock().map(|t| t.clone()).unwrap_or_default();
                    return Err(format!("the relex worker ended while loading: {tail}"));
                }
            }
        }
    }

    /// One passage through the worker (`{"id", "relex": payload}` → `{"id", "text", "stats"}`).
    async fn run(&self, payload: Value) -> Result<Value, RpcError> {
        let queued_at = Instant::now();
        self.queued.fetch_add(1, Ordering::SeqCst);
        let mut slot = self.worker.lock().await;
        self.queued.fetch_sub(1, Ordering::SeqCst);
        self.busy.store(true, Ordering::SeqCst);
        let queue_ms = now_ms(queued_at);
        let outcome = self.run_locked(&mut slot, payload).await;
        self.busy.store(false, Ordering::SeqCst);
        if let Ok(mut at) = self.last_used.lock() {
            *at = Instant::now();
        }
        if let Ok((_, work_ms, tokens)) = &outcome {
            self.metrics.record(*tokens, *work_ms, queue_ms);
        }
        outcome.map(|(answer, _, _)| answer)
    }

    async fn run_locked(&self, slot: &mut Option<Worker>, payload: Value) -> Result<(Value, f64, u64), RpcError> {
        if slot.is_none() {
            let worker = self.start_worker().await.map_err(|e| RpcError::new(codes::UNAVAILABLE, e).retryable())?;
            if let Ok(mut device) = self.device.lock() {
                device.clone_from(&worker.device);
            }
            self.load_ms.store(worker.load_ms, Ordering::SeqCst);
            self.loaded.store(true, Ordering::SeqCst);
            *slot = Some(worker);
        }
        let worker = slot.as_mut().expect("started above");
        let id = self.next_id.fetch_add(1, Ordering::SeqCst).to_string();
        let started = Instant::now();
        let line = format!("{}\n", json!({ "id": id, "relex": payload }));
        let answer = async {
            worker.stdin.write_all(line.as_bytes()).await.map_err(|e| e.to_string())?;
            worker.stdin.flush().await.map_err(|e| e.to_string())?;
            loop {
                let Some(line) = worker.stdout.next_line().await.map_err(|e| e.to_string())? else {
                    return Err("the relex worker ended".to_owned());
                };
                let Ok(answer) = serde_json::from_str::<Value>(&line) else { continue };
                if answer["id"].as_str() == Some(id.as_str()) || answer["id"].is_null() {
                    return Ok(answer);
                }
            }
        }
        .await;
        let answer = match answer {
            Ok(answer) => answer,
            Err(error) => {
                // The process is gone or out of step: start a fresh one on the next passage.
                if let Some(mut worker) = slot.take() {
                    let _ = worker.child.kill().await;
                }
                self.loaded.store(false, Ordering::SeqCst);
                let tail = self.stderr_tail.lock().map(|t| t.clone()).unwrap_or_default();
                return Err(RpcError::new(codes::INTERNAL, format!("{error}: {tail}")).retryable());
            }
        };
        if let Some(error) = answer["error"].as_str() {
            return Err(RpcError::new(codes::INTERNAL, error.to_owned()));
        }
        let tokens = answer["stats"]["tokens"].as_u64().unwrap_or(0);
        let text = answer.get("text").cloned().unwrap_or(Value::String(String::new()));
        let stats = answer.get("stats").cloned().unwrap_or_else(|| json!({}));
        Ok((json!({ "text": text, "stats": stats }), now_ms(started), tokens))
    }

    /// `model.extract`: `{"relex": {text, kinds, relations?, threshold?, relation_threshold?}}`, the
    /// thresholds defaulting to the config's. Cancelling the call drops the answer; the passage
    /// under way finishes in the worker (it cannot be stopped mid-pass without losing the model).
    pub async fn extract(self: &Arc<Self>, params: Value, cancel: CancellationToken) -> Result<Value, RpcError> {
        let Some(mut payload) = params.get("relex").cloned().filter(Value::is_object) else {
            return Err(RpcError::invalid_params("no `relex` passage"));
        };
        if payload.get("text").and_then(Value::as_str).is_none() || payload.get("kinds").and_then(Value::as_array).is_none() {
            return Err(RpcError::invalid_params("a `relex` passage needs `text` and `kinds`"));
        }
        if payload.get("threshold").is_none_or(Value::is_null) {
            payload["threshold"] = json!(self.config.threshold);
        }
        if payload.get("relation_threshold").is_none_or(Value::is_null) {
            payload["relation_threshold"] = json!(self.config.relation_threshold);
        }
        if !self.installed() {
            self.install_in_background();
            let detail = match self.install_progress() {
                Some(p) if !p.error.is_empty() => format!("GLiNER-relex is not installed: {}", p.error),
                Some(p) => format!("GLiNER-relex is installing: {} {:.2} of {:.2} GB", p.phase, p.bytes as f64 / 1e9, p.total as f64 / 1e9),
                None => format!("GLiNER-relex is not installed in {}", self.dir.display()),
            };
            return Err(RpcError::new(codes::UNAVAILABLE, detail).retryable());
        }
        let engine = self.clone();
        let running = tokio::spawn(async move { engine.run(payload).await });
        tokio::select! {
            joined = running => joined.map_err(|e| RpcError::new(codes::INTERNAL, e.to_string()))?,
            _ = cancel.cancelled() => Err(RpcError::cancelled()),
        }
    }

    pub fn info(&self) -> ModelInfo {
        let loaded = self.loaded.load(Ordering::SeqCst);
        let mut extra = json!({
            "source": "local",
            "holder": "cortexmodelserver",
            "kind": kinds::EXTRACT,
            "lane": self.lane,
            "state": if loaded { "loaded" } else if self.installed() { "unloaded" } else { "not installed" },
            "repo": store::REPO,
            "revision": store::REVISION,
            "encoder_dtype": self.config.encoder_dtype,
            "heads_dtype": "f32",
            "bilstm_device": "cpu",
            "window_tokens": WINDOW_TOKENS,
            "threshold": self.config.threshold,
            "relation_threshold": self.config.relation_threshold,
            "idle_unload_s": self.config.idle_unload_s,
        });
        if loaded {
            extra["load_ms"] = json!(self.load_ms.load(Ordering::SeqCst));
        }
        if let Some(progress) = self.install_progress() {
            extra["install"] = serde_json::to_value(progress).unwrap_or(Value::Null);
        }
        ModelInfo {
            model_id: format!("{}@{}:{}", store::REPO, &store::REVISION[..7], self.config.encoder_dtype),
            variant: MODEL_NAME.into(),
            path: self.dir.display().to_string(),
            family: "gliner-relex".into(),
            tags: vec![tags::BILSTM.into()],
            quant: self.config.encoder_dtype.clone(),
            file_bytes: store::total_bytes(),
            ctx_total: WINDOW_TOKENS,
            ctx_native: WINDOW_TOKENS,
            slots: 1,
            kv_type: String::new(),
            speculative: String::new(),
            backend: self.device.lock().map(|d| d.clone()).unwrap_or_default(),
            tensor_api: None,
            engine_revision: ENGINE_REVISION.into(),
            extra,
            vision: None,
            kind: kinds::EXTRACT.into(),
        }
    }

    pub fn slots(&self) -> SlotsResult {
        let busy = self.busy.load(Ordering::SeqCst);
        let queued = self.queued.load(Ordering::SeqCst);
        SlotsResult {
            slots: vec![SlotState { slot: 0, cache_key: None, busy, n_tokens: 0, priority: None }],
            queued,
            hibernated: Vec::new(),
            queued_by_priority: cc_protocol::contracts::model::QueueDepth { foreground: 0, agent: 0, background: queued },
            busy: u32::from(busy),
            pool_tokens: 0,
            reserved_tokens: 0,
            metrics: Some(self.metrics.live()),
            loaded: self.loaded.load(Ordering::SeqCst),
        }
    }

    pub async fn dispatch(self: &Arc<Self>, method: &str, params: Value, cancel: CancellationToken) -> Result<Value, RpcError> {
        match method {
            methods::INFO => to_value(&self.info()),
            methods::SLOTS => to_value(&self.slots()),
            methods::METRICS => to_value(&self.metrics.live()),
            methods::EXTRACT => self.extract(params, cancel).await,
            other if other.starts_with("model.") => Err(RpcError::new(
                codes::UNAVAILABLE,
                format!("{MODEL_NAME} is an extractor (kind extract): it answers model.extract, not {other}"),
            )),
            other => Err(RpcError::unknown_method(other)),
        }
    }
}
