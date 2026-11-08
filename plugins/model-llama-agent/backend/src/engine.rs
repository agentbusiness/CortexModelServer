//! Where each `model.*` call goes: this lane's model on CortexAgent's llama.cpp.
//!
//! The model is loaded on the first call that needs it (installing it first when needed), fitted to
//! the memory free at that moment (CortexAgent's fitter, `fit`), and stopped at critical memory
//! pressure or, when `idle_unload_s` is set, after that long without a request.
//!
//! CortexModelServer patch: one instance of this plugin runs per lane (the frame starts several at
//! once, each with its own config, CORTEX_PLUGIN_DATA and CORTEX_LANE). Nothing is borrowed from or
//! lent to the mesh here (ct's `borrow` is removed), there is no quantized-KV trial-and-fallback
//! (CortexAgent's fit chooses the KV type), and the chosen variant is kept per lane.
//!
//! A watcher looks every 5 s.

use crate::catalog::{self, InstallProgress, Spec};
use crate::config::Config;
use crate::gpu::{self, Gpu};
use crate::local;
use crate::progress::{Publisher, Tracker};
use crate::server::{Launch, Server};
use crate::store;
use cc_protocol::contracts::model::{
    ChatParams, CompleteParams, HoldParams, ModelInfo, PrewarmParams, SelectParams, SelectResult, SlotState, SlotsResult, TokenizeParams, Variant, VariantsResult, VisionInfo,
    kinds, methods, tags,
};
use cc_protocol::{Ctx, ItemSink, RpcError, codes, parse_params, to_value};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

// CortexModelServer: from CortexAgent (vendor/llama.cpp/REVISION and PATCHES.md).
pub const ENGINE_REVISION: &str = "llama.cpp unslothai 468ad58 (b11876) + CortexAgent vendor/llama.cpp/PATCHES.md (4 sites)";
const WATCH_EVERY: Duration = Duration::from_secs(5);

/// Where stream items go: the caller over CPP/1, or stdout (the `call` command).
pub enum Sink {
    Frame(ItemSink),
    Print,
}

impl Sink {
    pub fn send(&self, item: Value) -> bool {
        match self {
            Sink::Frame(sink) => sink.send(item),
            Sink::Print => {
                println!("{item}");
                true
            }
        }
    }
}

pub struct Engine {
    config: Config,
    data_dir: PathBuf,
    models: PathBuf,
    publish: Publisher,
    memory: u64,
    /// CortexModelServer patch: this instance's lane (`CORTEX_LANE`; "" outside the frame's lanes).
    lane: String,
    /// The GPU the model runs on; None: this machine runs no model.
    gpu: Option<Gpu>,
    variant: Mutex<&'static Spec>,
    /// The loaded llama-server. The lock is held while loading, so one load runs at a time.
    server: tokio::sync::Mutex<Option<Arc<Server>>>,
    ctx: Option<Ctx>,
    installing: tokio::sync::Mutex<()>,
    install: Arc<Mutex<Option<InstallProgress>>>,
    state: Mutex<&'static str>,
    last_request: Mutex<Instant>,
}

#[cfg(target_os = "macos")]
fn sysctl_u64(name: &std::ffi::CStr) -> Option<u64> {
    let mut value = 0u64;
    let mut length = std::mem::size_of::<u64>();
    // SAFETY: NUL-terminated key, writable u64 output whose size is passed, no new value.
    let status = unsafe { libc::sysctlbyname(name.as_ptr(), (&raw mut value).cast(), &mut length, std::ptr::null_mut(), 0) };
    if status != 0 {
        return None;
    }
    Some(if length == 4 { value & 0xffff_ffff } else { value })
}

/// `kern.memorystatus_vm_pressure_level` 4 = critical.
#[cfg(target_os = "macos")]
fn pressure_critical() -> bool {
    sysctl_u64(c"kern.memorystatus_vm_pressure_level") == Some(4)
}

/// Linux has no pressure level: critical when under 3% of memory is still available.
#[cfg(not(target_os = "macos"))]
fn pressure_critical() -> bool {
    match (crate::fit::available_memory_bytes(), crate::fit::physical_memory_bytes()) {
        (Some(available), Some(total)) => available * 100 < total * 3,
        _ => false,
    }
}

impl Engine {
    pub async fn new(config: Config, home: PathBuf, data_dir: PathBuf, publish: Publisher, ctx: Option<Ctx>) -> Arc<Engine> {
        let _ = std::fs::create_dir_all(&data_dir);
        crate::server::reclaim_stale(&data_dir);
        let models = match &config.models_dir {
            Some(dir) => PathBuf::from(dir),
            None => {
                let root = cc_shared_models::root();
                cc_shared_models::move_in(&home.join("models"), &root);
                root
            }
        };
        let lane = std::env::var("CORTEX_LANE").unwrap_or_default();
        let gpu = crate::server::binary(&config).ok().and_then(|bin| gpu::find(&bin));
        let memory = crate::fit::physical_memory_bytes().unwrap_or(0);
        // CortexModelServer: from CortexAgent: the lane's chosen model (`model.select`, kept per
        // lane), else its config, else CortexAgent's recommendation for this machine (ct: E4B above
        // 8 GiB, E2B otherwise).
        let (selected, late) = match &ctx {
            Some(ctx) => match store::variant(ctx, &lane, Duration::from_secs(5)).await {
                Ok(id) => (id, false),
                Err(()) => (None, true),
            },
            None => (None, false),
        };
        let working_set = gpu.as_ref().map_or(0, |gpu| gpu.memory_bytes);
        let variant = selected
            .and_then(|id| catalog::find(&id))
            .or_else(|| catalog::find(&config.variant))
            .unwrap_or_else(|| catalog::recommend(working_set, memory));
        tracing::info!(variant = variant.id, lane = %lane, memory, working_set, "model-llama-agent starting");
        let engine = Arc::new(Engine {
            config,
            data_dir,
            models,
            publish,
            memory,
            lane,
            gpu,
            variant: Mutex::new(variant),
            server: tokio::sync::Mutex::new(None),
            ctx,
            installing: tokio::sync::Mutex::new(()),
            install: Arc::new(Mutex::new(None)),
            state: Mutex::new("unloaded"),
            last_request: Mutex::new(Instant::now()),
        });
        if late && let Some(ctx) = engine.ctx.clone() {
            let weak = Arc::downgrade(&engine);
            let lane = engine.lane.clone();
            tokio::spawn(async move {
                let Ok(Some(id)) = store::variant(&ctx, &lane, Duration::from_secs(24 * 3600)).await else { return };
                let (Some(engine), Some(spec)) = (weak.upgrade(), catalog::find(&id)) else { return };
                if engine.spec().id != spec.id && engine.server.lock().await.is_none() {
                    tracing::info!(variant = spec.id, "the chosen model was read from the settings");
                    *engine.variant.lock().unwrap_or_else(|p| p.into_inner()) = spec;
                }
            });
        }
        let watcher = Arc::downgrade(&engine);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(WATCH_EVERY).await;
                let Some(engine) = watcher.upgrade() else { return };
                engine.watch_once().await;
            }
        });
        engine
    }

    fn spec(&self) -> &'static Spec {
        *self.variant.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn set_state(&self, state: &'static str) {
        *self.state.lock().unwrap_or_else(|p| p.into_inner()) = state;
    }

    fn state(&self) -> &'static str {
        *self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    async fn watch_once(self: &Arc<Self>) {
        let Some(server) = self.server.try_lock().ok().and_then(|slot| slot.clone()) else { return };
        if !server.alive() {
            tracing::warn!("llama-server exited; it loads again on the next call");
            drop(server);
            *self.server.lock().await = None;
            self.set_state("unloaded");
        } else if pressure_critical() {
            drop(server);
            let engine = self.clone();
            tokio::spawn(async move { engine.unload("memory pressure is critical", Duration::from_secs(10)).await });
        } else if self.config.idle_unload_s > 0 {
            // CortexModelServer: from CortexAgent: 0 (the default) keeps the model loaded.
            let idle = self.last_request.lock().unwrap_or_else(|p| p.into_inner()).elapsed();
            if idle >= Duration::from_secs(self.config.idle_unload_s) && Arc::strong_count(&server) <= 2 {
                drop(server);
                self.unload("idle", Duration::from_secs(1)).await;
            }
        }
    }

    async fn unload(&self, reason: &str, wait: Duration) -> f64 {
        let started = Instant::now();
        let Some(server) = self.server.lock().await.take() else { return 0.0 };
        self.set_state("unloading");
        tracing::info!(reason, "unloading the model");
        while Arc::strong_count(&server) > 1 && started.elapsed() < wait {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        server.stop();
        drop(server);
        self.set_state("unloaded");
        started.elapsed().as_secs_f64() * 1000.0
    }

    /// The loaded server, loading it (and installing the model) first when needed.
    async fn local(&self) -> Result<Arc<Server>, RpcError> {
        *self.last_request.lock().unwrap_or_else(|p| p.into_inner()) = Instant::now();
        if self.gpu.is_none() {
            return Err(no_gpu());
        }
        let mut slot = self.server.lock().await;
        if let Some(server) = slot.as_ref()
            && server.alive()
        {
            return Ok(server.clone());
        }
        if pressure_critical() {
            return Err(RpcError::new(codes::BUSY, "memory pressure is critical; the model is not loaded now").retryable());
        }
        let spec = self.spec();
        let dir = self.ensure_installed(spec).await?;
        self.set_state("loading");
        let gpu = self.gpu.as_ref().ok_or_else(no_gpu)?;
        let launch = Launch {
            spec,
            dir,
            config: &self.config,
            data_dir: self.data_dir.clone(),
            device_memory: gpu.memory_bytes,
            unified: gpu.backend == "metal",
        };
        let server = match Server::start(launch).await {
            Ok(server) => Arc::new(server),
            Err(error) => {
                self.set_state("unloaded");
                return Err(RpcError::new(codes::UNAVAILABLE, error).retryable());
            }
        };
        self.set_state("ready");
        *slot = Some(server.clone());
        Ok(server)
    }

    async fn ensure_installed(&self, spec: &'static Spec) -> Result<PathBuf, RpcError> {
        if self.gpu.is_none() {
            return Err(no_gpu());
        }
        if catalog::installed(spec, &self.models) {
            return Ok(spec.dir(&self.models));
        }
        let _guard = self.installing.lock().await;
        if catalog::installed(spec, &self.models) {
            return Ok(spec.dir(&self.models));
        }
        self.set_state("downloading");
        let publish = self.publish.clone();
        let current = self.install.clone();
        let report: catalog::Report = Arc::new(move |progress: InstallProgress| {
            if let Ok(value) = serde_json::to_value(&progress) {
                publish("model.install", value);
            }
            *current.lock().unwrap_or_else(|p| p.into_inner()) = Some(progress);
        });
        let result = catalog::install(spec, self.models.clone(), report).await;
        *self.install.lock().unwrap_or_else(|p| p.into_inner()) = None;
        self.set_state("unloaded");
        result.map_err(|error| RpcError::new(codes::UNAVAILABLE, format!("installing {}: {error}", spec.id)).retryable())
    }

    pub fn install_in_background(self: &Arc<Self>) {
        let engine = self.clone();
        tokio::spawn(async move {
            if let Err(error) = engine.ensure_installed(engine.spec()).await {
                tracing::warn!(%error, "model install failed");
            }
        });
    }

    pub fn installed(&self) -> bool {
        catalog::installed(self.spec(), &self.models)
    }

    pub fn install_progress(&self) -> Option<InstallProgress> {
        self.install.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    pub fn without_gpu(&self) -> bool {
        self.gpu.is_none()
    }

    fn extra(&self, source: &str, holder: Value, server: Option<&Server>) -> Value {
        let mut extra = json!({
            "source": source,
            "holder": holder,
            "tier": self.spec().tier,
            "state": self.state(),
            "memory_bytes": self.memory,
            "lane": self.lane,
            "origin": "CortexAgent",
        });
        if let Some(progress) = self.install_progress() {
            extra["install"] = serde_json::to_value(progress).unwrap_or(Value::Null);
        }
        if let Some(server) = server {
            extra["load_ms"] = json!(server.load_ms);
            extra["cache_ram_mib"] = json!((server.launch.prompt_cache_bytes >> 20).max(1));
            extra["ctx_checkpoints"] = json!(server.launch.context_checkpoints);
            extra["draft_max"] = json!(server.launch.draft_max);
            if let Some(fit) = &server.fit {
                extra["fit"] = json!({
                    "within_budget": fit.within_budget,
                    "native_context_tokens": fit.native_context_tokens,
                    "checkpoint_bytes": fit.checkpoint_bytes,
                    "device_bytes": fit.device_bytes,
                    "ram_bytes": fit.ram_bytes,
                    "device_budget_bytes": fit.budget.device_bytes,
                    "ram_budget_bytes": fit.budget.ram_bytes,
                });
            }
        }
        extra["input_modalities"] = if self.config.vision { json!(["text", "image"]) } else { json!(["text"]) };
        extra
    }

    fn local_info(&self, server: Option<&Server>) -> ModelInfo {
        let spec = server.map(|s| s.spec).unwrap_or_else(|| self.spec());
        let dir = spec.dir(&self.models);
        let extra = match server {
            Some(server) => self.extra("local", json!("cortexmodelserver"), Some(server)),
            None => self.extra("none", Value::Null, None),
        };
        let vision = server.map_or(self.config.vision, |s| s.launch.projector.is_some());
        ModelInfo {
            model_id: spec.model_id(),
            variant: spec.id.into(),
            path: dir.join(spec.weights.name).display().to_string(),
            family: "gemma4".into(),
            tags: vec![tags::OFFICE.into()],
            quant: spec.quant.into(),
            file_bytes: spec.weights.bytes,
            ctx_total: server.map_or(self.config.ctx, |s| s.launch.context_tokens),
            ctx_native: server.map(|s| s.ctx_native).filter(|n| *n > 0).unwrap_or(match spec.tier {
                "26b-a4b" => 262_144,
                _ => 131_072,
            }),
            slots: server.map_or(u32::from(self.config.slots), |s| u32::from(s.launch.slots)),
            kv_type: server.map(|s| s.kv_type.clone()).unwrap_or_else(|| self.config.kv_type.clone()),
            speculative: "draft-mtp".into(),
            backend: self.gpu.as_ref().map_or("none", |gpu| gpu.backend).into(),
            tensor_api: None,
            engine_revision: ENGINE_REVISION.into(),
            extra,
            vision: vision.then(|| VisionInfo {
                available: true,
                loaded: server.is_some(),
                idle_unload_s: self.config.idle_unload_s,
                ..VisionInfo::default()
            }),
            // CortexModelServer patch: model@1 now carries `kind` (ct's plugin predates it).
            kind: kinds::CHAT.into(),
        }
    }

    pub async fn info(&self) -> Result<Value, RpcError> {
        if self.gpu.is_none() {
            return Err(no_gpu());
        }
        let server = self.server.try_lock().ok().and_then(|slot| slot.clone());
        to_value(&self.local_info(server.as_deref()))
    }

    fn variants(&self) -> VariantsResult {
        if self.gpu.is_none() {
            return VariantsResult { variants: Vec::new() };
        }
        let active = self.spec().id;
        VariantsResult {
            variants: catalog::CATALOG
                .iter()
                .map(|spec| {
                    let installed = catalog::installed(spec, &self.models);
                    Variant {
                        id: spec.id.into(),
                        label: spec.label.into(),
                        path: spec.dir(&self.models).join(spec.weights.name).display().to_string(),
                        installed,
                        active: spec.id == active,
                        notes: spec.notes.into(),
                        repo: Some(spec.repo.into()),
                        revision: Some(spec.revision.into()),
                        sha256: Some(spec.weights.sha256.into()),
                        verified: installed.then_some(true),
                        bytes: Some(spec.files().iter().map(|f| f.bytes).sum()),
                        mtp: Some(true),
                    }
                })
                .collect(),
        }
    }

    async fn select(&self, id: &str) -> Result<SelectResult, RpcError> {
        let started = Instant::now();
        let spec = catalog::find(id).ok_or_else(|| RpcError::invalid_params(format!("unknown variant `{id}`")))?;
        let previous = self.spec().id.to_owned();
        if previous == spec.id {
            let server = self.server.lock().await.clone();
            return Ok(SelectResult { variant: spec.id.into(), previous, changed: false, reload_ms: 0.0, unload_ms: 0.0, load_ms: 0.0, info: self.local_info(server.as_deref()) });
        }
        self.ensure_installed(spec).await?;
        let was_loaded = self.server.lock().await.is_some();
        let unload_ms = self.unload("variant switch", Duration::from_secs(600)).await;
        *self.variant.lock().unwrap_or_else(|p| p.into_inner()) = spec;
        if let Some(ctx) = &self.ctx {
            store::save_variant(ctx, &self.lane, spec.id);
        }
        let mut load_ms = 0.0;
        if was_loaded {
            let server = self.local().await?;
            load_ms = server.load_ms;
        }
        let server = self.server.lock().await.clone();
        Ok(SelectResult {
            variant: spec.id.into(),
            previous,
            changed: true,
            reload_ms: started.elapsed().as_secs_f64() * 1000.0,
            unload_ms,
            load_ms,
            info: self.local_info(server.as_deref()),
        })
    }

    /// `model.progress` request ids carry the lane, so lanes never share one.
    fn tracker(&self, key: &str, priority: cc_protocol::contracts::model::Priority, work: &str) -> Tracker {
        Tracker::new(self.publish.clone(), &self.lane, key, priority, work)
    }

    pub async fn dispatch(self: &Arc<Self>, method: &str, params: Value, items: Option<&Sink>, cancel: CancellationToken) -> Result<Value, RpcError> {
        match method {
            methods::INFO => return self.info().await,
            methods::VARIANTS => return to_value(&self.variants()),
            methods::SELECT => {
                let params: SelectParams = parse_params(params)?;
                return to_value(&self.select(&params.variant).await?);
            }
            methods::CHAT | methods::COMPLETE | methods::TOKENIZE | methods::COUNT | methods::SLOTS | methods::HIBERNATE | methods::HOLD | methods::PREWARM => {}
            other => return Err(RpcError::unknown_method(other)),
        }
        if matches!(method, methods::HIBERNATE | methods::HOLD) {
            return Ok(local_noop(method, &params));
        }
        if method == methods::SLOTS && self.server.try_lock().ok().is_some_and(|slot| slot.is_none()) {
            return to_value(&idle_slots(u32::from(self.config.slots)));
        }
        let server = self.local().await?;
        match method {
            methods::CHAT => {
                let params: ChatParams = parse_params(params)?;
                let key = params.cache_key.clone().unwrap_or_default();
                let mut tracker = self.tracker(&key, params.priority, "generate");
                let result = local::chat(&server, &params, items, &cancel, &mut tracker).await;
                if result.is_err() {
                    tracker.done("failed", None);
                }
                to_value(&result?)
            }
            methods::COMPLETE => {
                let params: CompleteParams = parse_params(params)?;
                let key = params.cache_key.clone().unwrap_or_default();
                let mut tracker = self.tracker(&key, params.priority, "generate");
                let result = local::complete(&server, &params, items, &cancel, &mut tracker).await;
                if result.is_err() {
                    tracker.done("failed", None);
                }
                to_value(&result?)
            }
            methods::PREWARM => {
                let params: PrewarmParams = parse_params(params)?;
                if params.idle_only && Arc::strong_count(&server) > 2 {
                    return Ok(json!({"cache_key": params.cache_key, "skipped": "busy"}));
                }
                let mut tracker = self.tracker(&params.cache_key, cc_protocol::contracts::model::Priority::Background, "prewarm");
                local::prewarm(&server, &params, &cancel, &mut tracker).await
            }
            methods::TOKENIZE => {
                let params: TokenizeParams = parse_params(params)?;
                to_value(&local::tokenize(&server, &params.text).await?)
            }
            methods::COUNT => {
                let params: ChatParams = parse_params(params)?;
                to_value(&local::count(&server, &params).await?)
            }
            methods::SLOTS => to_value(&local::slots(&server).await?),
            _ => Ok(local_noop(method, &params)),
        }
    }

    pub async fn shutdown(&self) {
        if let Some(server) = self.server.lock().await.take() {
            server.stop();
        }
    }
}

fn no_gpu() -> RpcError {
    RpcError::new(codes::UNAVAILABLE, "this machine has no Metal GPU to run CortexAgent's llama.cpp").retryable()
}

fn idle_slots(slots: u32) -> SlotsResult {
    SlotsResult {
        slots: (0..slots).map(|slot| SlotState { slot, cache_key: None, busy: false, n_tokens: 0, priority: None }).collect(),
        queued: 0,
        hibernated: Vec::new(),
        queued_by_priority: Default::default(),
        busy: 0,
        pool_tokens: 0,
        reserved_tokens: 0,
        metrics: None,
        loaded: false,
    }
}

/// `model.hibernate` and `model.hold`: the scheduler saves and restores conversations itself.
fn local_noop(method: &str, params: &Value) -> Value {
    if method == methods::HOLD
        && let Ok(hold) = serde_json::from_value::<HoldParams>(params.clone())
    {
        return json!({"cache_key": hold.cache_key, "hold": hold.hold});
    }
    json!({"cache_key": params["cache_key"], "hibernated": false})
}
