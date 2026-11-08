//! `cc-model-relex`: CortexModelServer's extractor engine plugin (`model@1`, kind `extract`).
//! GLiNER-relex large (`knowledgator/gliner-relex-large-v1.0@4aedc92`) on candle: one encoder
//! pass gives a passage's entities and the relations between them (`model.extract`).
//!
//! `relex/` is the knowledge app's relex lane (`model-worker/src/relex`), copied as is; this file gives it
//! the three items of the knowledge app's worker it uses (`Args`, `emit`, `model_name`). The plugin process
//! runs it as a child, `cc-model-relex worker --model <dir>` (JSON lines on stdin/stdout, as
//! the knowledge app's pool drives it), so unloading the model is ending that process (see `engine`).
//!
//! Without a frame, `cc-model-relex call <method> [<params json>]` serves one call (the
//! configuration from `CORTEX_PLUGIN_CONFIG`, JSON).

mod config;
mod engine;
mod metrics;
mod relex;
mod store;

use cc_protocol::contracts::lifecycle::{Hello, ProbeCheck, ProbeReport, SwapPolicy};
use cc_protocol::contracts::model::methods;
use cc_protocol::{BoxFuture, Call, Ctx, Plugin, RpcError};
use config::Config;
use engine::Engine;
use serde_json::{Value, json};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tokio_util::sync::CancellationToken;

const PROVIDES: &[&str] = &[methods::INFO, methods::SLOTS, methods::METRICS, methods::EXTRACT];

/// What the knowledge app's worker hands its relex lane (only `model` is read there).
struct Args {
    model: PathBuf,
}

// Copied from the knowledge app's model-worker/src/main.rs (the relex lane calls them).
fn model_name(dir: &Path) -> String {
    dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

fn emit(value: &Value) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{value}");
    let _ = out.flush();
}

#[derive(Default)]
struct RelexPlugin {
    engine: Arc<tokio::sync::OnceCell<Arc<Engine>>>,
}

impl RelexPlugin {
    /// The engine, created on the first call (the frame's configuration is known by then).
    fn engine(&self, ctx: &Ctx) -> impl std::future::Future<Output = Arc<Engine>> + Send + 'static {
        let cell = self.engine.clone();
        let config = ctx.clone();
        async move { cell.get_or_init(|| async move { Engine::new(Config::from_value(&config.config())) }).await.clone() }
    }
}

fn check(name: &str, started: Instant, outcome: Result<String, String>) -> ProbeCheck {
    let ms = started.elapsed().as_millis() as u64;
    match outcome {
        Ok(detail) => ProbeCheck { name: name.into(), ok: true, detail, ms },
        Err(detail) => ProbeCheck { name: name.into(), ok: false, detail, ms },
    }
}

/// The real self-test: the model is installed (or downloading, which is reported, not failed),
/// loads, and finds the people and the relation in a known sentence.
async fn probe(engine: Arc<Engine>) -> ProbeReport {
    let started = Instant::now();
    if !engine.installed() {
        engine.install_in_background();
        let detail = match engine.install_progress() {
            Some(p) if !p.error.is_empty() => return ProbeReport::from_checks(vec![check("model installs", started, Err(p.error))]),
            Some(p) => format!("{} {:.2} of {:.2} GB", p.phase, p.bytes as f64 / 1e9, p.total as f64 / 1e9),
            None => "installing".into(),
        };
        return ProbeReport::from_checks(vec![check("model installing", started, Ok(detail))]);
    }
    let params = json!({"relex": {
        "text": "Ada Lovelace worked with Charles Babbage in London.",
        "kinds": ["person", "location"],
        "relations": ["worked with"],
    }});
    let outcome = match engine.dispatch(methods::EXTRACT, params, CancellationToken::new()).await {
        Ok(answer) => {
            let parsed: Value = answer["text"].as_str().and_then(|t| serde_json::from_str(t).ok()).unwrap_or(Value::Null);
            let people = parsed["entities"].as_array().map_or(0, |e| e.iter().filter(|e| e["label"] == "person").count());
            let relations = parsed["relations"].as_array().map_or(0, Vec::len);
            if people >= 2 {
                Ok(format!("{people} people, {relations} relations in {} ms", answer["stats"]["total_ms"]))
            } else {
                Err(format!("expected two people, got {}", answer["text"]))
            }
        }
        Err(error) => Err(error.message),
    };
    ProbeReport::from_checks(vec![check("extracts entities and relations", started, outcome)])
}

impl Plugin for RelexPlugin {
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

/// `worker --model <dir>`: the knowledge app's relex lane, reading requests on stdin until it closes.
fn worker(args: &[String]) -> i32 {
    let model = args.iter().position(|a| a == "--model").and_then(|i| args.get(i + 1)).map(PathBuf::from);
    let Some(model) = model else {
        eprintln!("usage: cc-model-relex worker --model <dir>");
        return 2;
    };
    match relex::serve(Args { model }) {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("{error:#}");
            1
        }
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_env("CORTEX_LOG").unwrap_or_else(|_| "info".into()))
        .init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("worker") {
        std::process::exit(worker(&args[1..]));
    }
    let runtime = match tokio::runtime::Builder::new_multi_thread().enable_all().build() {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("cannot start the async runtime: {error}");
            std::process::exit(1);
        }
    };
    let code = runtime.block_on(async move {
        if args.first().map(String::as_str) == Some("call") {
            return call_once(&args[1..]).await;
        }
        let mut hello = Hello::new("model-relex", env!("CARGO_PKG_VERSION"), "model", "model@1");
        hello.provides = PROVIDES.iter().map(|m| (*m).to_owned()).collect();
        // Two instances of a lane cannot both hold the model: drain the old one first.
        hello.swap = SwapPolicy::DrainFirst;
        match cc_protocol::run_plugin(hello, Arc::new(RelexPlugin::default())).await {
            Ok(()) => 0,
            Err(error) => {
                eprintln!("cc-model-relex: {error}");
                1
            }
        }
    });
    // The worker child is killed on drop; one left loading reads end-of-input and exits.
    std::process::exit(code);
}
