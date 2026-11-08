//! Whether a lane fits on its GPU: weights + KV for its context + compute buffers, beside the
//! lanes already there.
//!
//! A lane's sizes are the app's own (its recipe), never resized to fit. A shared pool is counted
//! in shares (`ctx_share`, 32k for the office model): one share per slot, and a call that states
//! more books several (`ClaimParams.tokens`), so the biggest call any app states gets its shares
//! on top of the normal slots. A lane whose engine fits its own pool (`keys.ctx` empty: uzu) is
//! set no context and no slots; it is charged for what its conversations reach. A window per
//! slot (`ctx_per_slot`) is the engine's fixed window times the slots. Extractors have no KV.

use crate::util;
use cc_protocol::contracts::serve::{Fit, Gpu, LaneSettings, Recipe};

/// The engine sizes its own pool to the memory it is given: the lane sets no context and no slots.
pub fn engine_fits(recipe: &Recipe) -> bool {
    recipe.keys.ctx.is_empty() && recipe.kv_bytes_per_token > 0
}

/// The memory of a lane with these settings.
pub fn bytes(recipe: &Recipe, settings: &LaneSettings) -> (u64, u64, u64) {
    let ctx = ctx_of(recipe, settings) as u64;
    let windows = if recipe.ctx_per_slot { slots_of(recipe, settings) as u64 } else { 1 };
    // An engine that fits its own pool holds memory only as far as its conversations reach;
    // llama.cpp allocates all of it.
    let filled = if engine_fits(recipe) && recipe.ctx_min_per_slot > 0 { ctx.min(recipe.ctx_min_per_slot as u64 * slots_of(recipe, settings) as u64) } else { ctx };
    (recipe.weights_bytes, recipe.kv_bytes_per_token.saturating_mul(filled).saturating_mul(windows), recipe.compute_bytes)
}

pub fn total(recipe: &Recipe, settings: &LaneSettings) -> u64 {
    let (weights, kv, compute) = bytes(recipe, settings);
    weights + kv + compute
}

/// The shares of the biggest call an app states, when it is more than the one share every slot
/// has: a 131k report call on 32k shares is four, run on top of the normal slots.
pub fn big_call_shares(recipe: &Recipe) -> u32 {
    if recipe.ctx_share == 0 || recipe.ctx_min <= recipe.ctx_share { 0 } else { recipe.ctx_min.div_ceil(recipe.ctx_share) }
}

/// The slots a lane runs: as set, else the recipe's own plus the biggest call's shares.
pub fn slots_of(recipe: &Recipe, settings: &LaneSettings) -> u32 {
    let slots = if settings.slots > 0 { settings.slots } else { recipe.slots_default.max(1) + big_call_shares(recipe) };
    if recipe.slots_max > 0 { slots.min(recipe.slots_max).max(1) } else { slots.max(1) }
}

/// The context a lane runs: as set, else one share per slot, else the recipe's own (at least
/// what the apps state they need).
pub fn ctx_of(recipe: &Recipe, settings: &LaneSettings) -> u32 {
    if recipe.kv_bytes_per_token == 0 && recipe.ctx_default == 0 {
        return 0;
    }
    let ctx = if settings.ctx > 0 {
        settings.ctx
    } else if recipe.ctx_share > 0 {
        recipe.ctx_share.saturating_mul(slots_of(recipe, settings))
    } else {
        recipe.ctx_default.max(recipe.ctx_min)
    };
    if recipe.ctx_max > 0 { ctx.min(recipe.ctx_max) } else { ctx }
}

/// Fit `settings` on `gpu`, where `others` bytes are taken by the other lanes.
pub fn check(recipe: &Recipe, settings: &LaneSettings, gpu: &Gpu, usable: u64, others: u64) -> Fit {
    let (weights, kv, compute) = bytes(recipe, settings);
    let total = weights + kv + compute;
    let room = usable.saturating_sub(others);
    let fits = total <= room && (recipe.min_gpu_bytes == 0 || gpu.memory_bytes >= recipe.min_gpu_bytes);
    let slots = slots_of(recipe, settings);
    let ctx = ctx_of(recipe, settings);
    let left = room.saturating_sub(weights + compute);
    // The most context that fits with the weights and buffers, and the most slots at the lane's
    // share (a shared pool) or window (per slot). An engine that fits itself decides both.
    let (max_ctx, max_slots) = if recipe.kv_bytes_per_token == 0 {
        (0, if fits { recipe.slots_max.max(64) } else { 0 })
    } else if engine_fits(recipe) {
        (0, 0)
    } else {
        let windows = if recipe.ctx_per_slot { slots as u64 } else { 1 };
        let most_ctx = left / (recipe.kv_bytes_per_token.max(1) * windows);
        let most_ctx = if recipe.ctx_max > 0 { most_ctx.min(recipe.ctx_max as u64) } else { most_ctx };
        let per_slot = if recipe.ctx_share > 0 { recipe.ctx_share as u64 } else if recipe.ctx_per_slot { ctx.max(1) as u64 } else { 0 };
        let most_slots = if per_slot > 0 { left / (recipe.kv_bytes_per_token.max(1) * per_slot) } else if fits { recipe.slots_max.max(64) as u64 } else { 0 };
        let most_slots = if recipe.slots_max > 0 { most_slots.min(recipe.slots_max as u64) } else { most_slots };
        (most_ctx as u32, most_slots.min(u32::MAX as u64) as u32)
    };
    let note = if recipe.min_gpu_bytes > 0 && gpu.memory_bytes < recipe.min_gpu_bytes {
        format!("needs a GPU of {} or more", util::gb(recipe.min_gpu_bytes))
    } else if !fits {
        format!("needs {} more", util::gb(total - room))
    } else if engine_fits(recipe) {
        format!("fits, the engine sizes its pool in {}", util::gb(room))
    } else if ctx > 0 {
        format!("fits {slots} slot{} at {}", if slots == 1 { "" } else { "s" }, util::ctx_label(ctx))
    } else {
        format!("fits, {}", util::gb(total))
    };
    Fit { fits, weights_bytes: weights, kv_bytes: kv, compute_bytes: compute, total_bytes: total, gpu_bytes: usable, others_bytes: others, max_slots, max_ctx, note }
}
