//! Whether a lane fits on its GPU: weights + KV for its context + compute buffers, beside the
//! lanes already there.
//!
//! A lane's context is the KV pool its slots share (the engines' unified pool), so its KV is
//! `kv_bytes_per_token × ctx`, or each slot's window (`ctx_per_slot`: × slots); a recipe's `kv_bytes_per_token` comes from the engine's own sizing
//! (its fit code, the model's layer types: sliding-window and recurrent layers count at their
//! real size). Extractors have no KV.

use crate::util;
use cc_protocol::contracts::serve::{Fit, Gpu, LaneSettings, Recipe};

/// The memory of a lane with these settings.
pub fn bytes(recipe: &Recipe, settings: &LaneSettings) -> (u64, u64, u64) {
    let ctx = ctx_of(recipe, settings) as u64;
    let windows = if recipe.ctx_per_slot { slots_of(recipe, settings) as u64 } else { 1 };
    // uzu's pool holds memory only as far as its conversations reach; llama.cpp allocates all of it.
    let filled = if recipe.engine == "model-uzu" && recipe.ctx_min_per_slot > 0 && !recipe.ctx_per_slot { ctx.min(recipe.ctx_min_per_slot as u64 * slots_of(recipe, settings) as u64) } else { ctx };
    (recipe.weights_bytes, recipe.kv_bytes_per_token.saturating_mul(filled).saturating_mul(windows), recipe.compute_bytes)
}

pub fn total(recipe: &Recipe, settings: &LaneSettings) -> u64 {
    let (weights, kv, compute) = bytes(recipe, settings);
    weights + kv + compute
}

pub fn slots_of(recipe: &Recipe, settings: &LaneSettings) -> u32 {
    let slots = if settings.slots == 0 { recipe.slots_default.max(1) } else { settings.slots };
    slots.clamp(1, recipe.slots_max.max(1))
}

pub fn ctx_of(recipe: &Recipe, settings: &LaneSettings) -> u32 {
    if recipe.kv_bytes_per_token == 0 && recipe.ctx_default == 0 {
        return 0;
    }
    let ctx = if settings.ctx == 0 { recipe.ctx_default } else { settings.ctx }.max(recipe.ctx_min);
    if recipe.ctx_max > 0 { ctx.min(recipe.ctx_max) } else { ctx }
}

/// The context of a lane with `slots` slots: a shared pool grows with them, so each conversation
/// keeps the room it had. A window per slot stays as it is.
pub fn ctx_for_slots(recipe: &Recipe, settings: &LaneSettings, slots: u32) -> u32 {
    let ctx = ctx_of(recipe, settings);
    if ctx == 0 || recipe.ctx_per_slot || recipe.kv_bytes_per_token == 0 {
        return settings.ctx;
    }
    let scaled = (ctx as u64 * slots as u64 / slots_of(recipe, settings).max(1) as u64).max(recipe.ctx_min_per_slot as u64 * slots as u64);
    (if recipe.ctx_max > 0 { scaled.min(recipe.ctx_max as u64) } else { scaled }) as u32
}

/// Fit `settings` on `gpu`, where `others` bytes are taken by the other lanes.
pub fn check(recipe: &Recipe, settings: &LaneSettings, gpu: &Gpu, usable: u64, others: u64) -> Fit {
    let (weights, kv, compute) = bytes(recipe, settings);
    let total = weights + kv + compute;
    let room = usable.saturating_sub(others);
    let fits = total <= room && (recipe.min_gpu_bytes == 0 || gpu.memory_bytes >= recipe.min_gpu_bytes);
    // The most context that fits with the weights and buffers; slots share it.
    let max_ctx = if recipe.kv_bytes_per_token == 0 {
        0
    } else {
        let windows = if recipe.ctx_per_slot { slots_of(recipe, settings) as u64 } else { 1 };
        let left = room.saturating_sub(weights + compute) / (recipe.kv_bytes_per_token.max(1) * windows);
        let cap = if recipe.ctx_max > 0 { recipe.ctx_max as u64 } else { left };
        left.min(cap) as u32
    };
    let max_slots = if fits || recipe.kv_bytes_per_token == 0 { recipe.slots_max.max(1) } else { 0 };
    let slots = slots_of(recipe, settings);
    let ctx = ctx_of(recipe, settings);
    let note = if recipe.min_gpu_bytes > 0 && gpu.memory_bytes < recipe.min_gpu_bytes {
        format!("needs a GPU of {} or more", util::gb(recipe.min_gpu_bytes))
    } else if fits {
        if ctx > 0 { format!("fits {slots} slot{} at {}", if slots == 1 { "" } else { "s" }, util::ctx_label(ctx)) } else { format!("fits, {}", util::gb(total)) }
    } else {
        format!("needs {} more", util::gb(total - room))
    };
    Fit { fits, weights_bytes: weights, kv_bytes: kv, compute_bytes: compute, total_bytes: total, gpu_bytes: usable, others_bytes: others, max_slots, max_ctx, note }
}

/// The largest context of the recipe's usual sizes (or below) that fits in `room`.
pub fn best_ctx(recipe: &Recipe, room: u64) -> Option<u32> {
    if recipe.kv_bytes_per_token == 0 {
        return (recipe.weights_bytes + recipe.compute_bytes <= room).then_some(recipe.ctx_default);
    }
    let fixed = recipe.weights_bytes + recipe.compute_bytes;
    if fixed >= room {
        return None;
    }
    let most = (room - fixed) / recipe.kv_bytes_per_token;
    let wanted = if recipe.ctx_default > 0 { recipe.ctx_default as u64 } else { 32768 };
    if most >= wanted {
        return Some(wanted as u32);
    }
    // Shrink by halves to no less than 8k, or than the apps' minimum: smaller windows are not
    // worth a lane.
    let floor = 8192.max(recipe.ctx_min as u64);
    let mut ctx = wanted;
    while ctx > floor && ctx > most {
        ctx /= 2;
    }
    (ctx <= most && ctx >= floor).then_some(ctx as u32)
}
