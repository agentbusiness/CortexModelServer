//! The plan: which lanes this computer runs. It starts with none: only a lane set by hand or a
//! model a Cortex app on this computer has selected runs here.
//!
//! 1. Lanes set by hand stay as they are (their memory is taken first).
//! 2. What the Cortex apps on this computer have selected comes next: for each model an app's card
//!    lists, the recipe of the same model (same weights first, else the same family and tag). The
//!    selection is kept while the app is away. This app says the highest sharing priority on its
//!    card, so an app whose tags a lane serves stops its own model and uses the lane
//!    (`mesh::Card::sharing_priority`); what it has loaded now is not counted against the plan, and a lane
//!    waits to load until the app has let go of it.
//!
//! Every lane is fitted (weights + KV + buffers, beside the lanes already on its GPU) and its
//! context shrunk to fit when needed; one that does not fit is left out with a note. No GPU: no
//! lanes (the standard's GPU rule).
//!
//! A planned lane runs the slots and drafting the last full calibration found for it, measured
//! with the other lanes of its GPU busy, with its context sized for those slots; before one has
//! run, its recipe's defaults. A lane goes to the GPU calibration moved it to.

use crate::fit;
use cc_protocol::contracts::serve::{Gpu, LaneSettings, PlanChange, Recipe, Want};
use std::collections::{BTreeMap, BTreeSet};

/// What a full calibration found for a lane.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Tuned {
    pub slots: u32,
    pub speculative: Option<bool>,
    pub draft_max: Option<u32>,
}

pub struct Input<'a> {
    pub recipes: &'a [Recipe],
    pub gpus: &'a [Gpu],
    pub reserve: u64,
    pub wants: &'a [Want],
    pub manual: Vec<LaneSettings>,
    /// Memory other programs hold on each GPU (by index).
    pub taken: BTreeMap<u32, u64>,
    /// Planned lanes removed by hand (`<recipe>-g<gpu>`).
    pub declined: &'a BTreeSet<String>,
    /// What the last full calibration found, by lane id.
    pub calibrated: &'a BTreeMap<String, Tuned>,
    /// The GPU calibration moved a recipe to.
    pub moved: &'a BTreeMap<String, u32>,
}

pub struct Output {
    pub lanes: Vec<LaneSettings>,
    pub notes: Vec<String>,
}

struct Room<'a> {
    recipes: &'a [Recipe],
    gpus: &'a [Gpu],
    reserve: u64,
    used: BTreeMap<u32, u64>,
    lanes: Vec<LaneSettings>,
    notes: Vec<String>,
    declined: &'a BTreeSet<String>,
    calibrated: &'a BTreeMap<String, Tuned>,
}

/// A planned lane's id: the recipe and the GPU.
pub fn base_id(recipe: &str, gpu: u32) -> String {
    let base: String = format!("{recipe}-g{gpu}").chars().map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '-' }).collect();
    base.trim_matches('-').chars().take(60).collect()
}

impl Room<'_> {
    fn recipe(&self, id: &str) -> Option<&Recipe> {
        self.recipes.iter().find(|recipe| recipe.id == id)
    }

    fn free(&self, gpu: &Gpu) -> u64 {
        crate::gpus::usable(gpu, self.reserve).saturating_sub(self.used.get(&gpu.index).copied().unwrap_or(0))
    }

    /// Place `recipe` on `gpu` when it fits there (its context shrunk to fit). True when placed.
    fn place(&mut self, recipe: &Recipe, gpu: &Gpu) -> bool {
        if self.declined.contains(&format!("{}-g{}", recipe.id, gpu.index)) {
            return false;
        }
        let on_device = runs_on(recipe, gpu);
        if on_device && recipe.min_gpu_bytes > 0 && gpu.memory_bytes < recipe.min_gpu_bytes {
            return false;
        }
        if !recipe.backends.is_empty() && !recipe.backends.iter().any(|backend| *backend == gpu.backend || backend == "cpu") {
            return false;
        }
        let free = if on_device { self.free(gpu) } else { u64::MAX };
        let id = self.lane_id(&recipe.id, gpu.index);
        let found = self.calibrated;
        let tuned = found.get(&id).filter(|tuned| tuned.slots > 0);
        let calibrated = tuned.map(|tuned| tuned.slots);
        let slots = calibrated.unwrap_or(recipe.slots_default.max(1)).min(recipe.slots_max.max(1));
        let windows = if recipe.ctx_per_slot { slots as u64 } else { 1 };
        let wanted = fit::ctx_for_slots(recipe, &LaneSettings { slots: recipe.slots_default, ..LaneSettings::default() }, slots);
        let per_window = Recipe { kv_bytes_per_token: recipe.kv_bytes_per_token * windows, ctx_default: if wanted > 0 { wanted } else { recipe.ctx_default }, ..recipe.clone() };
        let Some(ctx) = fit::best_ctx(&per_window, free) else {
            return false;
        };
        // A shared pool gives every slot at least the share the engine needs: fewer slots
        // when the pool came out small.
        let slots = if recipe.ctx_min_per_slot > 0 && !recipe.ctx_per_slot { slots.min(ctx / recipe.ctx_min_per_slot) } else { slots };
        if slots == 0 {
            return false;
        }
        let slots = if recipe.slots_default == 0 && calibrated.is_none() { 0 } else { slots };
        let settings = LaneSettings { id, recipe: recipe.id.clone(), gpu: gpu.index, slots, ctx, lend: true, speculative: tuned.and_then(|tuned| tuned.speculative), draft_max: tuned.and_then(|tuned| tuned.draft_max), ..LaneSettings::default() };
        let bytes = fit::total(recipe, &settings);
        if bytes > free {
            return false;
        }
        if on_device {
            *self.used.entry(gpu.index).or_default() += bytes;
        }
        self.lanes.push(settings);
        true
    }

    fn lane_id(&self, recipe: &str, gpu: u32) -> String {
        let base = base_id(recipe, gpu);
        let mut id = base.clone();
        let mut n = 2;
        while self.lanes.iter().any(|lane| lane.id == id) {
            id = format!("{base}-{n}");
            n += 1;
        }
        id
    }
}

/// Whether a lane of `recipe` runs on `gpu` itself. One whose engine has no path to this kind of
/// GPU (the embedder and extractor on Vulkan) runs on the CPU and takes none of its memory.
fn runs_on(recipe: &Recipe, gpu: &Gpu) -> bool {
    recipe.backends.is_empty() || recipe.backends.iter().any(|backend| *backend == gpu.backend)
}

pub fn plan(input: Input) -> Output {
    let mut gpus: Vec<&Gpu> = input.gpus.iter().collect();
    gpus.sort_by(|a, b| b.memory_bytes.cmp(&a.memory_bytes).then(a.index.cmp(&b.index)));
    let mut room = Room { recipes: input.recipes, gpus: input.gpus, reserve: input.reserve, used: input.taken.clone(), lanes: Vec::new(), notes: Vec::new(), declined: input.declined, calibrated: input.calibrated };
    if gpus.is_empty() {
        room.notes.push("No GPU: this computer serves no model (models run on Metal, CUDA or Vulkan only).".into());
        return Output { lanes: input.manual, notes: room.notes };
    }
    // 1. Lanes set by hand.
    for lane in input.manual {
        if let Some(recipe) = room.recipe(&lane.recipe).cloned()
            && room.gpus.iter().any(|gpu| gpu.index == lane.gpu && runs_on(&recipe, gpu))
        {
            *room.used.entry(lane.gpu).or_default() += fit::total(&recipe, &lane);
        }
        room.lanes.push(lane);
    }
    // 2. What the apps on this computer have selected.
    let mut served_wants = BTreeSet::new();
    for want in input.wants {
        if want.recipe.is_empty() || served_wants.contains(&want.recipe) || room.lanes.iter().any(|lane| lane.recipe == want.recipe) {
            continue;
        }
        let Some(recipe) = room.recipe(&want.recipe).cloned() else { continue };
        let mut order = gpus.clone();
        if let Some(moved) = input.moved.get(&recipe.id) {
            order.sort_by_key(|gpu| gpu.index != *moved);
        }
        let placed = order.iter().any(|gpu| room.place(&recipe, gpu));
        if placed {
            served_wants.insert(want.recipe.clone());
        } else {
            room.notes.push(format!("{} uses {}, which does not fit beside the other lanes here.", want.app, recipe.name));
        }
    }
    if room.lanes.is_empty() {
        room.notes.push("No Cortex app on this computer has selected a model yet, and no lane was set by hand.".into());
    }
    Output { lanes: room.lanes, notes: room.notes }
}

/// What applying `planned` changes in `current`.
pub fn changes(current: &[LaneSettings], planned: &[LaneSettings]) -> Vec<PlanChange> {
    let mut out = Vec::new();
    for lane in planned {
        match current.iter().find(|now| now.id == lane.id) {
            None => out.push(PlanChange { op: "add".into(), lane: lane.clone(), why: "the plan runs it".into() }),
            Some(now) if now != lane => out.push(PlanChange { op: "change".into(), lane: lane.clone(), why: "the plan sizes it differently".into() }),
            Some(_) => {}
        }
    }
    for lane in current {
        if !planned.iter().any(|planned| planned.id == lane.id) {
            out.push(PlanChange { op: "remove".into(), lane: lane.clone(), why: "the plan does not run it".into() });
        }
    }
    out
}
