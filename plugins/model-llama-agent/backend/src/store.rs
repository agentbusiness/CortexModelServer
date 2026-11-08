//! What this plugin keeps, in the app's graph (`memory@1`): the model chosen for a lane
//! (`model.select`), as a global setting.
//!
//! CortexModelServer patch: ct's plugin kept one `model.variant` setting and a quantized-KV check
//! per model. Several instances of this plugin run at once (one per lane), so the setting is per
//! lane (`model-llama-agent.<lane>.variant`; `model-llama-agent.variant` outside a lane), and the KV
//! check is gone (CortexAgent's fit chooses the KV type). Nothing is kept in files.

use cc_protocol::contracts::memory::methods;
use cc_protocol::{Ctx, RpcError, codes};
use serde_json::{Value, json};
use std::time::{Duration, Instant};

const RETRY: Duration = Duration::from_secs(1);

fn setting(lane: &str) -> String {
    if lane.is_empty() { "model-llama-agent.variant".into() } else { format!("model-llama-agent.{lane}.variant") }
}

fn waiting(error: &RpcError) -> bool {
    matches!(error.code.as_str(), codes::UNAVAILABLE | codes::CONNECTION_CLOSED | codes::DRAINING | codes::TIMEOUT)
}

async fn call(ctx: &Ctx, method: &str, params: Value, bound: Option<Duration>) -> Result<Value, RpcError> {
    let started = Instant::now();
    loop {
        match ctx.call(method, params.clone()).await {
            Err(error) if waiting(&error) && bound.is_none_or(|bound| started.elapsed() < bound) => tokio::time::sleep(RETRY).await,
            other => return other,
        }
    }
}

/// The lane's chosen model; `Err` when the memory plugin did not answer within `bound`.
pub async fn variant(ctx: &Ctx, lane: &str, bound: Duration) -> Result<Option<String>, ()> {
    match call(ctx, methods::SETTINGS_GET, json!({ "path": setting(lane) }), Some(bound)).await {
        Ok(value) => Ok(value.get("value").and_then(Value::as_str).map(str::to_owned).filter(|id| !id.is_empty())),
        Err(error) if error.code == codes::NOT_FOUND => Ok(None),
        Err(error) => {
            tracing::info!(error = %error.message, "the chosen model was not read from the settings yet");
            Err(())
        }
    }
}

/// Keep the lane's chosen model (in the background, until the memory plugin takes it).
pub fn save_variant(ctx: &Ctx, lane: &str, id: &str) {
    let (ctx, path, id) = (ctx.clone(), setting(lane), id.to_owned());
    tokio::spawn(async move {
        if let Err(error) = call(&ctx, methods::SETTINGS_SET, json!({ "path": path, "value": id, "by": "model" }), None).await {
            tracing::warn!(error = %error.message, "the chosen model was not kept");
        }
    });
}
