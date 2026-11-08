//! Model lanes (`frame.lanes.set`, `frame.lanes.list`).
//!
//! Every model this computer serves is a lane: one instance of an engine plugin (category
//! `model`), run as its own slot keyed `model/<lane id>`, with the lane's config merged over the
//! package's `[backend.config]`, its own environment (`CORTEX_LANE`, GPU selection) and data
//! folder. Many lanes run at once, of one engine or of several, and each is reached as the model
//! named by its id (`model::PLUGIN_PARAM` in a call's params; the hub routes by it).
//!
//! The frame keeps the lanes only while it runs: the serve plugin owns them (they are records in
//! its graph) and sets them again whenever it starts. A set replaces the whole list: new lanes
//! start, lanes no longer named drain and stop, changed lanes drain and restart, and unchanged
//! lanes keep serving. A lane whose engine is not installed waits for it (the install starts it).

use crate::hub::Frame;
use crate::supervisor::{self, Slot, lane_key, lane_of, valid_lane_id};
use cc_protocol::contracts::frame::{LaneSpec, LaneState, LanesResult, LanesSetParams, topics};
use cc_protocol::{RpcError, to_value};
use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;

/// `frame.lanes.set`: run exactly these lanes.
pub async fn set(frame: &Arc<Frame>, params: LanesSetParams) -> Result<LanesResult, RpcError> {
    let mut seen = BTreeSet::new();
    for lane in &params.lanes {
        if !valid_lane_id(&lane.id) {
            return Err(RpcError::invalid_params(format!("lane id `{}` must be 1 to 64 of [a-z0-9-]", lane.id)));
        }
        if !seen.insert(lane.id.clone()) {
            return Err(RpcError::invalid_params(format!("lane `{}` is named twice", lane.id)));
        }
        if let Some(unit) = frame.units.get(&lane.plugin)
            && unit.category().as_deref() != Some(supervisor::LANES_CATEGORY)
        {
            return Err(RpcError::invalid_params(format!("`{}` is not an engine (category `{}`)", lane.plugin, supervisor::LANES_CATEGORY)));
        }
        if !lane.config.is_null() && !lane.config.is_object() {
            return Err(RpcError::invalid_params(format!("lane `{}`: config must be a table", lane.id)));
        }
    }
    // Installs and lane changes take the same lock: an engine's swap never meets a half-set list.
    let _op = frame.units.op.lock().await;
    for slot in frame.sup.lane_slots() {
        let id = lane_of(&slot.category).unwrap_or_default().to_owned();
        if !seen.contains(&id) {
            supervisor::disable(frame, &slot).await;
            frame.sup.remove_slot(&slot.category);
            tracing::info!(lane = %id, "lane removed");
        }
    }
    for lane in params.lanes {
        let key = lane_key(&lane.id);
        match frame.sup.slot(&key) {
            Some(slot) if slot.lane().as_ref() == Some(&lane) => {
                // Unchanged; one that failed for good (its engine arrived since) starts again.
                if slot.live_current().is_none() && !slot.is_disabled() && engine_enabled(frame, &lane.plugin)
                    && let Some(target) = frame.units.current_spec(&lane.plugin)
                {
                    tokio::spawn(supervisor::ensure_running(frame.clone(), slot, target, Duration::ZERO));
                }
            }
            Some(slot) => {
                supervisor::disable(frame, &slot).await;
                if slot.plugin() != lane.plugin
                    && let Some(cfg) = frame.units.backend_config_of(&lane.plugin)
                {
                    slot.serve_with(crate::config::PluginConfig { category: key.clone(), ..cfg });
                }
                slot.set_lane(lane.clone());
                start(frame, &slot, &lane).await;
                tracing::info!(lane = %lane.id, plugin = %lane.plugin, "lane changed");
            }
            None => {
                let Some(cfg) = frame.units.backend_config_of(&lane.plugin) else {
                    // The engine is not installed: the lane waits for it.
                    let placeholder = crate::config::PluginConfig { plugin: lane.plugin.clone(), category: key.clone(), hello_timeout_ms: 10_000, probe_timeout_ms: 300_000 };
                    let slot = Slot::for_lane(&placeholder, lane.clone());
                    frame.sup.add_slot(slot.clone()).map_err(RpcError::invalid_params)?;
                    tracing::info!(lane = %lane.id, plugin = %lane.plugin, "lane waits for its engine to be installed");
                    continue;
                };
                let slot = Slot::for_lane(&cfg, lane.clone());
                frame.sup.add_slot(slot.clone()).map_err(RpcError::invalid_params)?;
                start(frame, &slot, &lane).await;
                tracing::info!(lane = %lane.id, plugin = %lane.plugin, "lane added");
            }
        }
    }
    let result = list(frame);
    publish(frame, &result);
    Ok(result)
}

/// Start a lane's slot when its engine is installed and enabled.
async fn start(frame: &Arc<Frame>, slot: &Arc<Slot>, lane: &LaneSpec) {
    if !engine_enabled(frame, &lane.plugin) {
        return;
    }
    if let Some(target) = frame.units.current_spec(&lane.plugin) {
        supervisor::enable(frame, slot, target).await;
    }
}

fn engine_enabled(frame: &Frame, plugin: &str) -> bool {
    frame.units.get(plugin).is_some_and(|unit| unit.enabled())
}

/// `frame.lanes.list`: every lane and its instance now.
pub fn list(frame: &Frame) -> LanesResult {
    let plugins = frame.sup.list(frame);
    let lanes = frame
        .sup
        .lane_slots()
        .into_iter()
        .filter_map(|slot| {
            let lane = slot.lane()?;
            let info = plugins.iter().find(|info| info.category == supervisor::LANES_CATEGORY && info.plugin_id == lane.id && !info.instance.is_empty());
            let state = match info {
                Some(info) => info.state.clone(),
                None => slot.describe_state(),
            };
            let state = if state == "disabled" && !engine_enabled(frame, &lane.plugin) {
                if frame.units.get(&lane.plugin).is_none() { format!("waiting for the engine `{}` to be installed", lane.plugin) } else { format!("the engine `{}` is disabled", lane.plugin) }
            } else {
                state
            };
            Some(LaneState {
                id: lane.id,
                plugin: lane.plugin,
                version: info.map(|info| info.build.clone()).unwrap_or_default(),
                instance: info.map(|info| info.instance.clone()).unwrap_or_default(),
                state,
                pid: info.map(|info| info.pid).unwrap_or(0),
                started_ms: info.map(|info| info.started_ms).unwrap_or(0),
                restarts: info.map(|info| info.restarts).unwrap_or(0),
            })
        })
        .collect();
    LanesResult { lanes }
}

/// Tell subscribers every lane's state (`frame.lanes.changed`).
pub fn publish(frame: &Frame, result: &LanesResult) {
    if let Ok(value) = to_value(result) {
        frame.publish(topics::LANES_CHANGED, &value);
    }
}
