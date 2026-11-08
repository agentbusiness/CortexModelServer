//! `cc-model-embed`: CortexModelServer's embedding engine plugin (`model@1`, kind `embed`).
//! Texts to vectors (`model.embed`) with the knowledge app's embedder, copied as it is into `embedding/`:
//! bekko-embedding-v1-a8m (ModernBERT, 384 dims, mean-pooled and L2-normalized, batches of 32
//! ordered by length) on candle, the embedder every Cortex app uses.
//!
//! Without a frame, `cc-model-embed call <method> [<params json>]` serves one call (the
//! configuration from `CORTEX_PLUGIN_CONFIG`, JSON).

mod config;
// The knowledge app's embedding module as it is: parts of it (the injection guard, the work gate, Hugging
// Face downloads) are not called here.
#[allow(dead_code, unused_imports)]
mod embedding;
mod engine;
mod metrics;
mod shared_models;
mod store;

use cc_protocol::contracts::lifecycle::{Hello, ProbeCheck, ProbeReport, SwapPolicy};
use cc_protocol::contracts::model::methods;
use cc_protocol::{BoxFuture, Call, Ctx, Plugin, RpcError};
use config::Config;
use engine::Engine;
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Instant;
use tokio_util::sync::CancellationToken;

const PROVIDES: &[&str] = &[methods::INFO, methods::SLOTS, methods::METRICS, methods::EMBED];

#[derive(Default)]
struct EmbedPlugin {
    engine: Arc<tokio::sync::OnceCell<Arc<Engine>>>,
}

impl EmbedPlugin {
    /// The engine, created on the first call (the frame's configuration is known by then).
    fn engine(&self, ctx: &Ctx) -> impl std::future::Future<Output = Arc<Engine>> + Send + 'static {
        let cell = self.engine.clone();
        let ctx = ctx.clone();
        async move { cell.get_or_init(|| async move { Engine::new(Config::from_value(&ctx.config())) }).await.clone() }
    }
}

fn check(name: &str, started: Instant, outcome: Result<String, String>) -> ProbeCheck {
    let ms = started.elapsed().as_millis() as u64;
    match outcome {
        Ok(detail) => ProbeCheck { name: name.into(), ok: true, detail, ms },
        Err(detail) => ProbeCheck { name: name.into(), ok: false, detail, ms },
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The real self-test: the model is installed (or downloading, which is reported, not failed),
/// loads, gives unit vectors of its width, and puts a paraphrase nearer than an unrelated text.
async fn probe(engine: Arc<Engine>) -> ProbeReport {
    let started = Instant::now();
    if !engine.installed() {
        engine.install_in_background();
        let (stage, error) = engine.install_state();
        if !error.is_empty() {
            return ProbeReport::from_checks(vec![check("model installs", started, Err(error))]);
        }
        return ProbeReport::from_checks(vec![check("model installing", started, Ok(if stage.is_empty() { "installing".into() } else { stage }))]);
    }
    // CortexModelServer patch: a lane loads its model on the first real call, not for the self-test.
    if std::env::var_os("CORTEX_LANE").is_some() {
        return ProbeReport::from_checks(vec![check("model installed", started, Ok("loads on its first call".to_owned()))]);
    }
    let params = json!({"texts": ["The cat sits on the mat.", "A cat is sitting on a rug.", "Quarterly revenue rose by four percent."]});
    let outcome = match engine.dispatch(methods::EMBED, params, CancellationToken::new()).await {
        Ok(answer) => {
            let vectors: Vec<Vec<f32>> = serde_json::from_value(answer["vectors"].clone()).unwrap_or_default();
            let dims = answer["dims"].as_u64().unwrap_or(0) as usize;
            if vectors.len() != 3 || dims == 0 || vectors.iter().any(|v| v.len() != dims || v.iter().any(|x| !x.is_finite())) {
                Err(format!("expected 3 vectors of {dims} finite values"))
            } else {
                let (near, far) = (cosine(&vectors[0], &vectors[1]), cosine(&vectors[0], &vectors[2]));
                if near > far {
                    Ok(format!("{dims} dims, paraphrase {near:.3} > unrelated {far:.3}"))
                } else {
                    Err(format!("paraphrase {near:.3} not nearer than unrelated {far:.3}"))
                }
            }
        }
        Err(error) => Err(error.message),
    };
    ProbeReport::from_checks(vec![check("embeds texts", started, outcome)])
}

impl Plugin for EmbedPlugin {
    fn handle(&self, ctx: Ctx, call: Call) -> BoxFuture<'static, Result<Value, RpcError>> {
        let engine = self.engine(&ctx);
        Box::pin(async move {
            let engine = engine.await;
            let Call { method, params, cancel, .. } = call;
            engine.dispatch(&method, params, cancel).await
        })
    }

    fn probe(&self, ctx: Ctx) -> BoxFuture<'static, Result<ProbeReport, RpcError>> {
        let engine = self.engine(&ctx);
        Box::pin(async move { Ok(probe(engine.await).await) })
    }

    fn activate(&self, ctx: Ctx) -> BoxFuture<'static, Result<(), RpcError>> {
        let engine = self.engine(&ctx);
        Box::pin(async move {
            engine.await;
            Ok(())
        })
    }

    fn shutdown(&self, _ctx: Ctx) -> BoxFuture<'static, ()> {
        let engine = self.engine.get().cloned();
        Box::pin(async move {
            if let Some(engine) = engine {
                engine.shutdown().await;
            }
        })
    }
}

/// `call <method> [<params>] ...`: the calls in order on one engine.
async fn call_once(args: &[String]) -> i32 {
    let config = std::env::var("CORTEX_PLUGIN_CONFIG").ok().and_then(|text| serde_json::from_str(&text).ok()).unwrap_or(Value::Null);
    let engine = Engine::new(Config::from_value(&config));
    let mut rest = args.iter().peekable();
    let mut code = 0;
    while let Some(method) = rest.next() {
        let params = match rest.peek() {
            Some(text) if text.trim_start().starts_with('{') => match serde_json::from_str(rest.next().map(String::as_str).unwrap_or("{}")) {
                Ok(value) => value,
                Err(error) => {
                    eprintln!("params of {method}: {error}");
                    return 2;
                }
            },
            _ => json!({}),
        };
        let started = Instant::now();
        let outcome = engine.dispatch(method, params, CancellationToken::new()).await;
        eprintln!("{method} took {:.0} ms", started.elapsed().as_secs_f64() * 1000.0);
        match outcome {
            Ok(result) => println!("{}", serde_json::to_string_pretty(&result).unwrap_or_default()),
            Err(error) => {
                eprintln!("{method} error: {error}");
                code = 1;
            }
        }
    }
    engine.shutdown().await;
    code
}

/// CortexModelServer patch: the folder of NVIDIA's libraries, in the installed model-llama package
/// (`plugins/model-llama/<version>/backend/llama`) that has every library the CUDA build links.
fn cuda_libraries() -> Option<std::path::PathBuf> {
    const NEEDED: [&str; 4] = ["libcudart.so", "libcublas.so", "libcublasLt.so", "libcurand.so"];
    let exe = std::env::current_exe().ok()?;
    let plugins = exe.ancestors().nth(4)?;
    let holds_all = |dir: &std::path::PathBuf| {
        let names: Vec<String> = std::fs::read_dir(dir).into_iter().flatten().flatten().map(|file| file.file_name().to_string_lossy().into_owned()).collect();
        NEEDED.iter().all(|lib| names.iter().any(|name| name.starts_with(&format!("{lib}."))))
    };
    std::fs::read_dir(plugins.join("model-llama")).ok()?.flatten().map(|version| version.path().join("backend/llama")).filter(holds_all).max()
}

/// CortexModelServer patch: become the package's CUDA build (`cuda/` beside this executable, Linux
/// only, see build.sh) when NVIDIA's driver and libraries are here. Otherwise this build runs, on
/// the CPU: the CUDA build would not even start.
fn become_cuda_build() {
    use std::os::unix::process::CommandExt;
    unsafe extern "C" {
        fn dlopen(file: *const std::ffi::c_char, mode: std::ffi::c_int) -> *mut std::ffi::c_void;
    }
    const RTLD_LAZY: std::ffi::c_int = 1;
    if !cfg!(target_os = "linux") || cfg!(feature = "cuda") {
        return;
    }
    // SAFETY: a constant library name; the handle is kept for the life of the process.
    if unsafe { dlopen(c"libcuda.so.1".as_ptr(), RTLD_LAZY) }.is_null() {
        return;
    }
    let Some(build) = std::env::current_exe().ok().map(|exe| exe.with_file_name("cuda").join(env!("CARGO_BIN_NAME"))).filter(|path| path.is_file()) else { return };
    let Some(libraries) = cuda_libraries() else { return };
    let path = std::env::join_paths(std::iter::once(libraries).chain(std::env::var_os("LD_LIBRARY_PATH").iter().flat_map(std::env::split_paths))).unwrap_or_default();
    let error = std::process::Command::new(&build).args(std::env::args_os().skip(1)).env("LD_LIBRARY_PATH", path).exec();
    eprintln!("cannot start {}: {error}; running on the CPU", build.display());
}

fn main() {
    become_cuda_build();
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_env("CORTEX_LOG").unwrap_or_else(|_| "info".into()))
        .init();
    let runtime = match tokio::runtime::Builder::new_multi_thread().enable_all().build() {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("cannot start the async runtime: {error}");
            std::process::exit(1);
        }
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = runtime.block_on(async move {
        if args.first().map(String::as_str) == Some("call") {
            return call_once(&args[1..]).await;
        }
        let mut hello = Hello::new("model-embed", env!("CARGO_PKG_VERSION"), "model", "model@1");
        hello.provides = PROVIDES.iter().map(|m| (*m).to_owned()).collect();
        // Two instances of a lane cannot both hold the model: drain the old one first.
        hello.swap = SwapPolicy::DrainFirst;
        match cc_protocol::run_plugin(hello, Arc::new(EmbedPlugin::default())).await {
            Ok(()) => 0,
            Err(error) => {
                eprintln!("cc-model-embed: {error}");
                1
            }
        }
    });
    std::process::exit(code);
}
