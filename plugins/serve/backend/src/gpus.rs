//! This computer's GPUs and their readings.
//!
//! - macOS: one entry, the unified memory. Models may use the GPU's working set (Metal's
//!   `recommendedMaxWorkingSetSize`, the frame measures it), less a reserve for the system; use,
//!   power and temperature come from the system plugin's sampler (`system.metrics`).
//! - Linux: every NVIDIA GPU from `nvidia-smi` (memory, use, power, temperature). Without it, the
//!   devices llama.cpp sees (`llama-server --list-devices` of the installed engine: CUDA and
//!   Vulkan), with their memory. No GPU: no models (the standard's GPU rule).

use crate::util;
use cc_protocol::Ctx;
use cc_protocol::contracts::serve::Gpu;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// What does not change while the plugin runs (names, memory), measured once.
#[derive(Debug, Clone, Default)]
pub struct Hardware {
    pub gpus: Vec<Gpu>,
    #[allow(dead_code)]
    pub ram_bytes: u64,
    pub chip: String,
}

pub async fn measure(ctx: &Ctx, home: &Path) -> Hardware {
    let machine = ctx.call("frame.machine", json!({})).await.unwrap_or(Value::Null);
    let ram_bytes = machine.get("memory_bytes").and_then(Value::as_u64).unwrap_or(0);
    let chip = machine.get("chip").and_then(Value::as_str).unwrap_or_default().to_owned();
    if cfg!(target_os = "macos") {
        let working_set = machine.get("gpu_working_set_bytes").and_then(Value::as_u64).filter(|bytes| *bytes > 0).unwrap_or(ram_bytes / 4 * 3);
        let gpu = Gpu {
            index: 0,
            name: if chip.is_empty() { "Apple GPU".into() } else { chip.clone() },
            backend: "metal".into(),
            memory_bytes: working_set,
            unified: true,
            uuid: format!("metal:{}", chip.to_lowercase().replace(' ', "-")),
            cores: machine.get("gpu_cores").and_then(Value::as_u64).unwrap_or(0) as u32,
            ..Gpu::default()
        };
        return Hardware { gpus: vec![gpu], ram_bytes, chip };
    }
    let mut gpus = nvidia_smi().await.unwrap_or_default();
    if gpus.is_empty() {
        gpus = list_devices(home).await;
    }
    Hardware { gpus, ram_bytes, chip }
}

/// The readings now: on a Mac from `system.metrics`, on Linux from `nvidia-smi`.
pub async fn read(ctx: &Ctx, hardware: &Hardware) -> Vec<Gpu> {
    let mut gpus = hardware.gpus.clone();
    if cfg!(target_os = "macos") {
        let metrics = tokio::time::timeout(Duration::from_secs(4), ctx.call("system.metrics", json!({}))).await.ok().and_then(Result::ok).unwrap_or(Value::Null);
        if let Some(gpu) = gpus.first_mut() {
            gpu.util = metrics.get("gpu").and_then(Value::as_f64).map(|percent| percent / 100.0).unwrap_or(0.0);
            gpu.power_w = metrics.get("gpu_power").and_then(Value::as_f64).unwrap_or(0.0);
            gpu.temp_c = metrics.get("gpu_temp").and_then(Value::as_f64).unwrap_or(0.0);
        }
        return gpus;
    }
    if let Some(now) = nvidia_smi().await {
        for gpu in &mut gpus {
            if let Some(reading) = now.iter().find(|reading| reading.uuid == gpu.uuid || (gpu.uuid.is_empty() && reading.index == gpu.index)) {
                gpu.used_bytes = reading.used_bytes;
                gpu.util = reading.util;
                gpu.power_w = reading.power_w;
                gpu.temp_c = reading.temp_c;
            }
        }
    }
    gpus
}

async fn output(program: &str, args: &[&str]) -> Option<String> {
    let run = tokio::process::Command::new(program).args(args).kill_on_drop(true).output();
    let output = tokio::time::timeout(Duration::from_secs(8), run).await.ok()?.ok()?;
    output.status.success().then(|| String::from_utf8_lossy(&output.stdout).into_owned())
}

async fn nvidia_smi() -> Option<Vec<Gpu>> {
    let text = output("nvidia-smi", &["--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu,power.draw,temperature.gpu", "--format=csv,noheader,nounits"]).await?;
    let mib = |value: &str| value.trim().parse::<f64>().map(|mib| (mib * 1024.0 * 1024.0) as u64).unwrap_or(0);
    let number = |value: &str| value.trim().parse::<f64>().unwrap_or(0.0);
    let gpus: Vec<Gpu> = text
        .lines()
        .filter_map(|line| {
            let fields: Vec<&str> = line.split(',').collect();
            (fields.len() >= 8).then(|| Gpu {
                index: fields[0].trim().parse().unwrap_or(0),
                name: fields[1].trim().to_owned(),
                backend: "cuda".into(),
                uuid: fields[2].trim().to_owned(),
                memory_bytes: mib(fields[3]),
                used_bytes: mib(fields[4]),
                util: number(fields[5]) / 100.0,
                power_w: number(fields[6]),
                temp_c: number(fields[7]),
                unified: false,
                cores: 0,
            })
        })
        .collect();
    (!gpus.is_empty()).then_some(gpus)
}

/// `llama-server --list-devices` of an installed llama engine: lines like
/// `  Vulkan0: AMD Radeon RX 7900 XTX (24560 MiB, 24000 MiB free)`.
async fn list_devices(home: &Path) -> Vec<Gpu> {
    let Some(server) = find_llama_server(home) else {
        return Vec::new();
    };
    let Some(text) = output(&server.display().to_string(), &["--list-devices"]).await else {
        return Vec::new();
    };
    let mut gpus = Vec::new();
    let mut vulkan_index = 0;
    for line in text.lines() {
        let line = line.trim();
        let Some((device, rest)) = line.split_once(':') else { continue };
        let backend = if device.starts_with("CUDA") {
            "cuda"
        } else if device.starts_with("Vulkan") {
            "vulkan"
        } else {
            continue;
        };
        let index: u32 = device.trim_start_matches(|c: char| c.is_ascii_alphabetic()).parse().unwrap_or(vulkan_index);
        vulkan_index += 1;
        let (name, memory) = rest.rsplit_once('(').unwrap_or((rest, ""));
        let total = memory.split(',').next().and_then(|part| part.trim().strip_suffix("MiB")).and_then(|mib| mib.trim().parse::<u64>().ok()).unwrap_or(0);
        let free = memory.split(',').nth(1).and_then(|part| part.trim().strip_suffix("MiB free)")).and_then(|mib| mib.trim().parse::<u64>().ok()).unwrap_or(total);
        if backend == "cuda" || !gpus.iter().any(|gpu: &Gpu| gpu.name == name.trim()) {
            gpus.push(Gpu {
                index,
                name: name.trim().to_owned(),
                backend: backend.into(),
                memory_bytes: total * 1024 * 1024,
                used_bytes: total.saturating_sub(free) * 1024 * 1024,
                uuid: format!("{}:{index}", backend),
                ..Gpu::default()
            });
        }
    }
    // A CUDA device is also listed by Vulkan: keep CUDA (the standard prefers it).
    if gpus.iter().any(|gpu| gpu.backend == "cuda") {
        gpus.retain(|gpu| gpu.backend == "cuda");
    }
    gpus
}

/// The `llama-server` of the installed llama engine (its current version).
fn find_llama_server(home: &Path) -> Option<PathBuf> {
    let plugin = home.join("plugins").join("model-llama");
    let version = std::fs::read_to_string(plugin.join(".current")).ok()?;
    let root = plugin.join(version.trim());
    let mut stack = vec![(root, 0)];
    while let Some((dir, depth)) = stack.pop() {
        for entry in std::fs::read_dir(&dir).ok()?.filter_map(Result::ok) {
            let path = entry.path();
            if path.file_name().is_some_and(|name| name == "llama-server") && path.is_file() {
                return Some(path);
            }
            if path.is_dir() && depth < 6 {
                stack.push((path, depth + 1));
            }
        }
    }
    None
}

/// What the engines are told so a lane runs on its GPU only.
pub fn lane_env(gpu: &Gpu) -> Vec<(String, String)> {
    match gpu.backend.as_str() {
        "cuda" => vec![("CUDA_VISIBLE_DEVICES".into(), gpu.index.to_string())],
        "vulkan" => vec![("GGML_VK_VISIBLE_DEVICES".into(), gpu.index.to_string())],
        _ => Vec::new(),
    }
}

/// Memory models may use on a GPU: its memory less the reserve (on a Mac the working set is
/// already the GPU's share).
pub fn usable(gpu: &Gpu, reserve: u64) -> u64 {
    gpu.memory_bytes.saturating_sub(reserve)
}

#[allow(dead_code)]
pub fn label(gpu: &Gpu) -> String {
    format!("GPU {} · {} · {}", gpu.index, gpu.name, util::gb(gpu.memory_bytes))
}
