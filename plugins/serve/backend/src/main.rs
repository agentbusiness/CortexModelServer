//! `cc-serve` executable: serve `serve@1` to the frame.

mod fit;
mod gpus;
// The ordered outbox CortexTerm's plugins write the graph through, copied as it is: some of
// its helpers are not used here.
#[allow(dead_code)]
mod graph;
mod plan;
mod recipes;
mod server;
mod stats;
mod util;

use cc_protocol::contracts::lifecycle::{Hello, ProbeReport};
use cc_protocol::{BoxFuture, Call, Ctx, Plugin, PluginEnv, RpcError, run_plugin};
use serde_json::Value;
use server::Server;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

const PROVIDES: &[&str] = &[
    "serve.status.get",
    "serve.gpus.list",
    "serve.recipes.list",
    "serve.lanes.list",
    "serve.lanes.create",
    "serve.lanes.edit",
    "serve.lanes.delete",
    "serve.fit.get",
    "serve.plan.get",
    "serve.plan.run",
    "serve.calibrate.run",
    "serve.cluster.get",
    "serve.history.read",
];

struct ServePlugin {
    server: Arc<Server>,
    stop: CancellationToken,
}

impl Plugin for ServePlugin {
    fn handle(&self, ctx: Ctx, call: Call) -> BoxFuture<'static, Result<Value, RpcError>> {
        self.server.attach(&ctx);
        let server = self.server.clone();
        Box::pin(async move { server.handle(&call.method, call.params).await })
    }

    fn probe(&self, ctx: Ctx) -> BoxFuture<'static, Result<ProbeReport, RpcError>> {
        self.server.attach(&ctx);
        let report = self.server.probe();
        Box::pin(async move { Ok(report) })
    }

    fn activate(&self, ctx: Ctx) -> BoxFuture<'static, Result<(), RpcError>> {
        self.server.attach(&ctx);
        self.server.start(self.stop.clone());
        Box::pin(async { Ok(()) })
    }

    fn event(&self, ctx: Ctx, topic: String, data: Value) -> BoxFuture<'static, ()> {
        self.server.attach(&ctx);
        self.server.event(&topic, data);
        Box::pin(async {})
    }

    fn shutdown(&self, _ctx: Ctx) -> BoxFuture<'static, ()> {
        self.stop.cancel();
        Box::pin(async {
            graph::flush(std::time::Duration::from_secs(3)).await;
        })
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_env("CORTEX_LOG").unwrap_or_else(|_| "info".into()))
        .with_writer(std::io::stderr)
        .init();
    let runtime = match tokio::runtime::Builder::new_multi_thread().enable_all().build() {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("cc-serve: cannot start the async runtime: {error}");
            std::process::exit(1);
        }
    };
    let code = runtime.block_on(async {
        let env = match PluginEnv::from_env() {
            Ok(env) => env,
            Err(error) => {
                eprintln!("cc-serve: {error}");
                return 2;
            }
        };
        // Recipes are shipped data: the live folder's copy when it has one, else the package's.
        let packaged = std::env::current_exe().ok().and_then(|exe| package_root(&exe)).map(|root| root.join("data").join("recipes")).unwrap_or_default();
        let recipes = env.live_or("data/recipes", packaged);
        let mut hello = Hello::new("serve", env!("CARGO_PKG_VERSION"), "serve", "serve@1");
        hello.provides = PROVIDES.iter().map(|m| (*m).to_owned()).collect();
        hello.subscribes = vec!["model.progress".into(), "frame.lanes.changed".into(), "mesh.changed".into()];
        let plugin = Arc::new(ServePlugin { server: Server::new(env.home.clone(), recipes), stop: CancellationToken::new() });
        match run_plugin(hello, plugin).await {
            Ok(()) => 0,
            Err(error) => {
                eprintln!("cc-serve: {error}");
                1
            }
        }
    });
    std::process::exit(code);
}

/// The installed package's folder: the executable is `<package>/backend/target/release/cc-serve`.
fn package_root(exe: &std::path::Path) -> Option<std::path::PathBuf> {
    exe.ancestors().find(|dir| dir.join("plugin.toml").is_file()).map(std::path::Path::to_path_buf)
}
