//! The engine: bekko (`store::BEKKO`), loaded in process on the first
//! `model.embed` (the knowledge app's embedders in `embedding/`, as they are), dropped after
//! `idle_unload_s` without one. One slot: requests run one at a time (the embedders are not safe
//! to call concurrently), the rest queue in arrival order.

use crate::config::Config;
use crate::embedding::{
    EmbeddingModel, EmbeddingOptions, ModernBertEmbeddingModel, local_semantic_model_files, provision_semantic_model_files,
    set_embedding_progress_handler,
};
use crate::metrics::Metrics;
use crate::store::{self, Spec};
use cc_protocol::contracts::model::{EmbedParams, EmbedResult, ModelInfo, QueueDepth, SlotState, SlotsResult, kinds, methods, tags};
use cc_protocol::{RpcError, codes, parse_params, to_value};
use serde_json::{Value, json};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

pub const ENGINE_REVISION: &str = "candle 0.10.2 + embedding/modernbert @ 4ea0c825 (copied, renames only)";

struct Loaded {
    model: Arc<dyn EmbeddingModel>,
    /// The model's own tokenizer, read again here to count tokens and cut texts to `truncate`.
    tokenizer: Arc<tokenizers::Tokenizer>,
}

pub struct Engine {
    config: Config,
    spec: Spec,
    lane: String,
    slot: tokio::sync::Mutex<Option<Loaded>>,
    queued: AtomicU32,
    busy: AtomicBool,
    loaded: AtomicBool,
    load_ms: AtomicU64,
    target: Mutex<String>,
    last_used: Mutex<Instant>,
    metrics: Metrics,
    /// The embedders' own progress stages and an install's outcome.
    stage: Arc<Mutex<String>>,
    install_error: Mutex<String>,
    installing: AtomicBool,
}

fn ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

/// `text` cut after its first `limit` tokens (by the tokenizer's byte offsets), or as it is.
fn cut(text: &str, encoding: &tokenizers::Encoding, limit: usize) -> String {
    if encoding.len() <= limit {
        return text.to_owned();
    }
    let mut end = encoding.get_offsets().iter().take(limit).map(|(_, end)| *end).max().unwrap_or(0).min(text.len());
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].to_owned()
}

impl Engine {
    pub fn new(config: Config) -> Arc<Self> {
        if let Some(dir) = &config.models_dir {
            crate::shared_models::set_root(dir.into());
        }
        if config.max_input_tokens > 0 && config.max_input_tokens != 512 {
            // SAFETY: set once, before the model loads, while nothing else in this process reads
            // or writes the environment (one lane per process; the embedders read it at load).
            unsafe { std::env::set_var("CORTEX_EMBED_MAX_INPUT_TOKENS", config.max_input_tokens.to_string()) };
        }
        let spec = store::find(&config.model).unwrap_or_else(|| {
            tracing::warn!(model = %config.model, "unknown embedding model; serving {}", store::BEKKO.id);
            store::BEKKO
        });
        let stage = Arc::new(Mutex::new(String::new()));
        let progress = stage.clone();
        set_embedding_progress_handler(Some(Arc::new(move |text: String| {
            if let Ok(mut stage) = progress.lock() {
                *stage = text;
            }
        })));
        let engine = Arc::new(Self {
            config,
            spec,
            lane: std::env::var("CORTEX_LANE").unwrap_or_default(),
            slot: tokio::sync::Mutex::new(None),
            queued: AtomicU32::new(0),
            busy: AtomicBool::new(false),
            loaded: AtomicBool::new(false),
            load_ms: AtomicU64::new(0),
            target: Mutex::new(String::new()),
            last_used: Mutex::new(Instant::now()),
            metrics: Metrics::default(),
            stage,
            install_error: Mutex::new(String::new()),
            installing: AtomicBool::new(false),
        });
        engine.clone().watch_idle();
        engine
    }

    pub fn installed(&self) -> bool {
        self.spec.installed()
    }

    pub fn install_state(&self) -> (String, String) {
        (self.stage.lock().map(|s| s.clone()).unwrap_or_default(), self.install_error.lock().map(|s| s.clone()).unwrap_or_default())
    }

    /// Download the missing files once, in the background.
    pub fn install_in_background(self: &Arc<Self>) {
        if !self.config.download || self.installed() || self.installing.swap(true, Ordering::SeqCst) {
            return;
        }
        let engine = self.clone();
        tokio::task::spawn_blocking(move || {
            let outcome = provision_semantic_model_files().map(|_| ()).map_err(|e| e.to_string());
            if let Err(error) = outcome {
                tracing::warn!(%error, model = engine.spec.id, "the embedding model was not installed");
                if let Ok(mut slot) = engine.install_error.lock() {
                    *slot = error;
                }
            }
            engine.installing.store(false, Ordering::SeqCst);
        });
    }

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
                let Ok(mut slot) = self.slot.try_lock() else { continue };
                if let Some(loaded) = slot.take() {
                    drop(loaded);
                    self.loaded.store(false, Ordering::SeqCst);
                    tracing::info!(model = self.spec.id, idle_s = idle.as_secs(), "embedding model unloaded after being idle");
                }
            }
        });
    }

    pub async fn shutdown(&self) {
        self.slot.lock().await.take();
        self.loaded.store(false, Ordering::SeqCst);
    }

    /// Load the model (blocking): the knowledge app's embedder with this lane's options.
    fn load(spec: Spec, config: &Config) -> Result<Loaded, String> {
        let options = EmbeddingOptions::default()
            .with_batch_size(config.batch_size.max(1))
            .with_intra_threads(config.threads.max(1))
            .with_prefer_gpu(config.prefer_gpu);
        let files = local_semantic_model_files().ok_or_else(|| format!("{} is not installed", spec.id))?;
        let model: Arc<dyn EmbeddingModel> =
            Arc::new(ModernBertEmbeddingModel::from_files(spec.id, &files, options).map_err(|e| e.to_string())?);
        let mut tokenizer = tokenizers::Tokenizer::from_file(spec.tokenizer()).map_err(|e| format!("tokenizer.json: {e}"))?;
        tokenizer.with_padding(None);
        let _ = tokenizer.with_truncation(None);
        Ok(Loaded { model, tokenizer: Arc::new(tokenizer) })
    }

    async fn run(self: Arc<Self>, params: EmbedParams) -> Result<EmbedResult, RpcError> {
        let queued_at = Instant::now();
        self.queued.fetch_add(1, Ordering::SeqCst);
        let mut slot = self.slot.lock().await;
        self.queued.fetch_sub(1, Ordering::SeqCst);
        self.busy.store(true, Ordering::SeqCst);
        let queue_ms = ms(queued_at);
        let outcome = self.run_locked(&mut slot, params).await;
        self.busy.store(false, Ordering::SeqCst);
        if let Ok(mut at) = self.last_used.lock() {
            *at = Instant::now();
        }
        if let Ok((result, work_ms)) = &outcome {
            self.metrics.record(result.tokens, *work_ms, queue_ms);
        }
        outcome.map(|(result, _)| result)
    }

    async fn run_locked(&self, slot: &mut Option<Loaded>, params: EmbedParams) -> Result<(EmbedResult, f64), RpcError> {
        if slot.is_none() {
            let (spec, config) = (self.spec, self.config.clone());
            let started = Instant::now();
            let loaded = tokio::task::spawn_blocking(move || Self::load(spec, &config))
                .await
                .map_err(|e| RpcError::new(codes::INTERNAL, e.to_string()))?
                .map_err(|e| RpcError::new(codes::UNAVAILABLE, e).retryable())?;
            let target = format!("{}/{}", loaded.model.backend(), loaded.model.execution_target());
            tracing::info!(model = spec.id, %target, load_ms = started.elapsed().as_millis() as u64, "embedding model loaded");
            if let Ok(mut slot) = self.target.lock() {
                *slot = target;
            }
            self.load_ms.store(started.elapsed().as_millis() as u64, Ordering::SeqCst);
            self.loaded.store(true, Ordering::SeqCst);
            *slot = Some(loaded);
        }
        let loaded = slot.as_ref().expect("loaded above");
        let (model, tokenizer) = (loaded.model.clone(), loaded.tokenizer.clone());
        let limit = params.truncate.map_or(usize::MAX, |t| t.max(1) as usize).min(self.config.max_input_tokens.max(1));
        let model_id = self.spec.model_id();
        let started = Instant::now();
        let result = tokio::task::spawn_blocking(move || -> Result<EmbedResult, String> {
            let encodings = tokenizer.encode_batch(params.texts.iter().map(String::as_str).collect::<Vec<_>>(), true).map_err(|e| format!("tokenizing: {e}"))?;
            let tokens: u64 = encodings.iter().map(|e| e.len().min(limit) as u64).sum();
            let texts: Vec<String> = params.texts.iter().zip(&encodings).map(|(text, encoding)| cut(text, encoding, limit)).collect();
            let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
            // The embedder mean-pools and L2-normalizes: the vectors are unit length whatever
            // `normalize` asks.
            let vectors = model.embed(&refs).map_err(|e| e.to_string())?;
            Ok(EmbedResult { dims: model.dimensions() as u32, vectors, model_id, tokens })
        })
        .await
        .map_err(|e| RpcError::new(codes::INTERNAL, e.to_string()))?
        .map_err(|e| RpcError::new(codes::INTERNAL, e))?;
        Ok((result, ms(started)))
    }

    /// `model.embed`. Cancelling the call drops the answer; the batch under way finishes.
    pub async fn embed(self: &Arc<Self>, params: Value, cancel: CancellationToken) -> Result<Value, RpcError> {
        let params: EmbedParams = parse_params(params)?;
        if params.texts.is_empty() {
            return to_value(&EmbedResult { vectors: Vec::new(), dims: self.spec.dims, model_id: self.spec.model_id(), tokens: 0 });
        }
        if !self.installed() {
            self.install_in_background();
            let (stage, error) = self.install_state();
            let detail = if !error.is_empty() {
                format!("{} is not installed: {error}", self.spec.id)
            } else if self.installing.load(Ordering::SeqCst) {
                format!("{} is installing ({stage})", self.spec.id)
            } else {
                format!("{} is not installed in {}", self.spec.id, self.spec.dir().display())
            };
            return Err(RpcError::new(codes::UNAVAILABLE, detail).retryable());
        }
        let running = tokio::spawn(self.clone().run(params));
        tokio::select! {
            joined = running => to_value(&joined.map_err(|e| RpcError::new(codes::INTERNAL, e.to_string()))??),
            _ = cancel.cancelled() => Err(RpcError::cancelled()),
        }
    }

    pub fn info(&self) -> ModelInfo {
        let loaded = self.loaded.load(Ordering::SeqCst);
        let target = self.target.lock().map(|t| t.clone()).unwrap_or_default();
        // CortexModelServer patch: the GPU is CUDA in the Linux build that has it.
        let gpu = if cfg!(feature = "cuda") { "cuda" } else { "metal" };
        let backend = if loaded {
            if target.ends_with("gpu") { gpu } else { "cpu" }
        } else if self.config.prefer_gpu && (cfg!(target_os = "macos") || cfg!(feature = "cuda")) {
            gpu
        } else {
            "cpu"
        };
        let (stage, error) = self.install_state();
        let mut extra = json!({
            "source": "local",
            "holder": "cortexmodelserver",
            "kind": kinds::EMBED,
            "lane": self.lane,
            "state": if loaded { "loaded" } else if self.installed() { "unloaded" } else { "not installed" },
            "repo": self.spec.repo,
            "revision": self.spec.revision,
            "runtime": "candle",
            "dims": self.spec.dims,
            "bytes": self.spec.bytes(),
            "pooling": "mean",
            "normalized": true,
            "batch_size": self.config.batch_size,
            "max_input_tokens": self.config.max_input_tokens,
            "idle_unload_s": self.config.idle_unload_s,
        });
        if loaded {
            extra["load_ms"] = json!(self.load_ms.load(Ordering::SeqCst));
            extra["target"] = json!(target);
        }
        if !stage.is_empty() {
            extra["stage"] = json!(stage);
        }
        if !error.is_empty() {
            extra["install_error"] = json!(error);
        }
        ModelInfo {
            model_id: self.spec.model_id(),
            variant: self.spec.id.into(),
            path: self.spec.dir().display().to_string(),
            family: self.spec.family.into(),
            tags: vec![tags::EMBED.into()],
            quant: "f32".into(),
            file_bytes: self.spec.weights_bytes(),
            ctx_total: self.config.max_input_tokens as u32,
            ctx_native: self.config.max_input_tokens as u32,
            slots: 1,
            kv_type: String::new(),
            speculative: String::new(),
            backend: backend.into(),
            tensor_api: None,
            engine_revision: ENGINE_REVISION.into(),
            extra,
            vision: None,
            kind: kinds::EMBED.into(),
        }
    }

    pub fn slots(&self) -> SlotsResult {
        let busy = self.busy.load(Ordering::SeqCst);
        let queued = self.queued.load(Ordering::SeqCst);
        SlotsResult {
            slots: vec![SlotState { slot: 0, cache_key: None, busy, n_tokens: 0, priority: None }],
            queued,
            hibernated: Vec::new(),
            queued_by_priority: QueueDepth { foreground: 0, agent: 0, background: queued },
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
            methods::EMBED => self.embed(params, cancel).await,
            other if other.starts_with("model.") => Err(RpcError::new(
                codes::UNAVAILABLE,
                format!("{} is an embedder (kind embed): it answers model.embed, not {other}", self.spec.id),
            )),
            other => Err(RpcError::unknown_method(other)),
        }
    }
}
