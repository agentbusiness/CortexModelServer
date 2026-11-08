//! The supervised `llama-server` child (CortexAgent's llama.cpp, see agent-server/): started on a free loopback port, health-waited, recorded
//! in `<data>/llama-server.pid` (a stale one left by a killed plugin is reclaimed at start), and
//! killed when it is dropped, unloaded or the plugin ends.

use crate::catalog::Spec;
use crate::config::Config;
use crate::fit::{self, ContextFit, LaunchConfig, MemoryBudget};
use crate::scheduler::{self, Scheduler};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

const PID_FILE: &str = "llama-server.pid";

pub struct Server {
    pub spec: &'static Spec,
    pub base: String,
    pub http: reqwest::Client,
    pub kv_type: String,
    pub load_ms: f64,
    pub ctx_native: u32,
    /// What llama-server was started with (CortexAgent's fitted configuration).
    pub launch: LaunchConfig,
    pub fit: Option<ContextFit>,
    pub scheduler: Arc<Scheduler>,
    child: std::sync::Mutex<Option<tokio::process::Child>>,
    pid: u32,
    pid_file: PathBuf,
}

/// The llama-server binary: beside this executable (the package's `backend/`), else the
/// configured path.
pub fn binary(config: &Config) -> Result<PathBuf, String> {
    if let Some(path) = &config.server_bin {
        return Ok(PathBuf::from(path));
    }
    let exe = std::env::current_exe().map_err(|e| format!("cannot find this executable: {e}"))?;
    let path = exe.parent().map(|dir| dir.join("llama").join("llama-server")).ok_or("the executable has no directory")?;
    if path.is_file() { Ok(path) } else { Err(format!("llama-server is missing beside {}", exe.display())) }
}

fn free_port() -> Result<u16, String> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").map_err(|e| format!("no free port: {e}"))?;
    listener.local_addr().map(|a| a.port()).map_err(|e| e.to_string())
}

fn alive(pid: u32) -> bool {
    // SAFETY: signal 0 only checks that the process exists.
    pid > 0 && unsafe { libc::kill(pid as i32, 0) } == 0
}

fn command_of(pid: u32) -> String {
    std::process::Command::new("ps")
        .args(["-o", "comm=", "-p", &pid.to_string()])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_default()
}

fn terminate(pid: u32) {
    if !alive(pid) {
        return;
    }
    // SAFETY: plain signals to a process this plugin started.
    unsafe { libc::kill(pid as i32, libc::SIGTERM) };
    let started = Instant::now();
    // Reap it when it is this process's child (a zombie still answers signal 0).
    let gone = || {
        // SAFETY: non-blocking wait on one pid; fails harmlessly for a process that is not our child.
        unsafe { libc::waitpid(pid as i32, std::ptr::null_mut(), libc::WNOHANG) };
        !alive(pid)
    };
    while !gone() && started.elapsed() < Duration::from_secs(5) {
        std::thread::sleep(Duration::from_millis(50));
    }
    if alive(pid) {
        unsafe { libc::kill(pid as i32, libc::SIGKILL) };
    }
}

/// Kill a llama-server a previous run of this plugin left behind (its pid file survived).
pub fn reclaim_stale(data_dir: &Path) {
    let path = data_dir.join(PID_FILE);
    let Ok(text) = std::fs::read_to_string(&path) else { return };
    if let Ok(pid) = text.trim().parse::<u32>()
        && alive(pid)
        && command_of(pid).ends_with("llama-server")
    {
        tracing::warn!(pid, "stopping a llama-server left behind by an earlier run");
        terminate(pid);
    }
    remove_if(&path, &text);
}

/// Remove the pid file only while it still names the process it named (a successor may have
/// written its own).
fn remove_if(path: &Path, expected: &str) {
    if std::fs::read_to_string(path).is_ok_and(|now| now == expected) {
        let _ = std::fs::remove_file(path);
    }
}

pub struct Launch<'a> {
    pub spec: &'static Spec,
    pub dir: PathBuf,
    pub config: &'a Config,
    pub data_dir: PathBuf,
    /// The GPU's memory as llama.cpp reports it (Metal: recommendedMaxWorkingSetSize), for the fit.
    pub device_memory: u64,
    /// Device allocations come out of host RAM (Metal).
    pub unified: bool,
}

impl Server {
    // CortexModelServer: from CortexAgent (crates/model/src/model.rs `LocalModel::llama` and
    // llama_server.rs `LlamaServer::start`): the launch configuration is CortexAgent's, fitted once
    // to the memory measured now; llama-server is then started with CortexAgent's arguments and
    // the loopback HTTP settings ct's plugin uses.
    pub async fn start(launch: Launch<'_>) -> Result<Server, String> {
        let Launch { spec, dir, config, data_dir, device_memory, unified } = launch;
        let bin = binary(config)?;
        let port = free_port()?;
        let weights = dir.join(spec.weights.name);
        let mut fit_config = LaunchConfig::for_model(weights.clone(), config.ctx);
        if fit_config.context_tokens == 0 {
            return Err("The model does not declare a valid native context length; no reduced context was substituted.".into());
        }
        fit_config.gpu_layers = config.gpu_layers;
        fit_config.ubatch_tokens = config.ubatch;
        fit_config.draft_max = config.draft_max;
        fit_config.drafter = Some(dir.join(spec.drafter.name));
        if config.vision {
            fit_config.projector = Some(dir.join(spec.projector.name));
        }
        // CortexModelServer patch: the lane's ceilings.
        if config.max_slots > 0 {
            fit_config.max_slots = config.max_slots.clamp(fit::MIN_SLOTS, fit::MAX_SLOTS);
        }
        if config.checkpoints > 0 {
            fit_config.max_checkpoints = config.checkpoints.min(fit::MAX_CONTEXT_CHECKPOINTS);
        }
        if config.cache_ram > 0 {
            fit_config.max_prompt_cache_bytes = (u64::from(config.cache_ram) << 20).min(fit::PROMPT_CACHE_MAX_BYTES);
        }
        fit_config.context_checkpoints = fit_config.max_checkpoints;
        fit_config.prompt_cache_bytes = fit_config.max_prompt_cache_bytes;
        let kv_cache = config.kv_cache()?;
        let explicit_slots = (config.slots > 0).then(|| config.slots.clamp(fit::MIN_SLOTS, fit::MAX_SLOTS));
        fit_config.slots = explicit_slots.unwrap_or(fit::DEFAULT_SLOTS.min(fit_config.max_slots));
        let fitted = match MemoryBudget::measure(device_memory, config.memory_limit, unified) {
            Some(budget) if explicit_slots.is_some() => Some(fit_config.fit_to_memory(budget, kv_cache)),
            Some(budget) => Some(fit_config.fit_slots_to_memory(budget, kv_cache)),
            None => {
                fit_config.kv_cache_type = kv_cache.unwrap_or(fit::DEFAULT_KV_CACHE_TYPE);
                tracing::warn!(context_tokens = fit_config.context_tokens, "memory could not be measured; the native context was kept");
                None
            }
        };
        if let Some(fit) = &fitted {
            fit.log();
        }
        let slot_dir = data_dir.join("slots");
        scheduler::prepare_slot_dir(&slot_dir)?;
        let mut args = fit_config.arguments(&slot_dir);
        // CortexModelServer patch: ct's loopback HTTP settings (CortexAgent has no listener at all);
        // ct's --device, -b, the fixed -c/--parallel and --cache-ram 1024 / --ctx-checkpoints 2 are gone.
        args.extend([
            "--no-webui".into(),
            "--host".into(),
            "127.0.0.1".into(),
            "--port".into(),
            port.to_string(),
        ]);
        let log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(data_dir.join("llama-server.log"))
            .map_err(|e| format!("cannot open the server log: {e}"))?;
        let started = Instant::now();
        tracing::info!(bin = %bin.display(), ?args, "starting llama-server");
        let mut child = tokio::process::Command::new(&bin)
            .args(&args)
            .stdin(std::process::Stdio::null())
            .stdout(log.try_clone().map_err(|e| e.to_string())?)
            .stderr(log)
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| format!("cannot start {}: {e}", bin.display()))?;
        let pid = child.id().unwrap_or(0);
        let pid_file = data_dir.join(PID_FILE);
        let _ = std::fs::write(&pid_file, pid.to_string());
        let base = format!("http://127.0.0.1:{port}");
        let http = reqwest::Client::builder().connect_timeout(Duration::from_secs(5)).build().map_err(|e| e.to_string())?;
        // Health: 503 while loading, 200 when ready.
        let deadline = Instant::now() + Duration::from_secs(300);
        loop {
            if let Ok(Some(status)) = child.try_wait() {
                let _ = std::fs::remove_file(&pid_file);
                return Err(format!("llama-server exited while loading ({status}); see {}", data_dir.join("llama-server.log").display()));
            }
            if let Ok(response) = http.get(format!("{base}/health")).timeout(Duration::from_secs(2)).send().await
                && response.status().is_success()
            {
                break;
            }
            if Instant::now() > deadline {
                let _ = child.start_kill();
                let _ = std::fs::remove_file(&pid_file);
                return Err("llama-server did not become healthy within 300 s".into());
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        let load_ms = started.elapsed().as_secs_f64() * 1000.0;
        let ctx_native = match http.get(format!("{base}/v1/models")).send().await {
            Ok(response) => response.json::<serde_json::Value>().await.ok().and_then(|v| v["data"][0]["meta"]["n_ctx_train"].as_u64()).unwrap_or(0) as u32,
            Err(_) => 0,
        };
        // CortexModelServer: from CortexAgent: one unified pool of `--ctx-size` cells shared by every
        // slot, and the scheduler that pins each request to a slot.
        let scheduler = Scheduler::new(fit_config.slots, u64::from(fit_config.context_tokens), slot_dir);
        scheduler::spawn_slot_spill(Arc::downgrade(&scheduler), http.clone(), base.clone());
        tracing::info!(variant = spec.id, load_ms, port, pid, slots = fit_config.slots, context_tokens = fit_config.context_tokens, "llama-server is ready");
        Ok(Server {
            spec,
            base,
            http,
            kv_type: fit_config.kv_cache_type.as_str().to_owned(),
            load_ms,
            ctx_native,
            launch: fit_config,
            fit: fitted,
            scheduler,
            child: std::sync::Mutex::new(Some(child)),
            pid,
            pid_file,
        })
    }

    pub fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }

    pub fn alive(&self) -> bool {
        alive(self.pid)
    }

    /// Stop the child (graceful, then forced) and forget its pid file.
    pub fn stop(&self) {
        let child = self.child.lock().ok().and_then(|mut slot| slot.take());
        if child.is_some() {
            terminate(self.pid);
            remove_if(&self.pid_file, &self.pid.to_string());
            tracing::info!(variant = self.spec.id, pid = self.pid, "llama-server stopped");
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Kill the child recorded in the pid file (signal handler path: no async, no locks).
pub fn kill_recorded(data_dir: &Path) {
    reclaim_stale(data_dir);
}
