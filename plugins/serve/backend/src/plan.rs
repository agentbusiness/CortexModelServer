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
//! A planned lane runs the app's own sizes (`fit::slots_of`, `fit::ctx_of`: the recipe's slots,
//! plus the shares of the biggest call any app states); the plan never shrinks or grows a lane. A
//! lane runs where it fits the GPU alone: on unified memory (a Mac) the lanes load on a call and
//! those that do not fit loaded together take turns, giving memory back after five idle minutes;
//! on a discrete GPU, whose memory is hard, a lane that does not fit beside the others is left out
//! with a note. No GPU: no lanes (the standard's GPU rule).
//!
//! Calibration measures a lane and tunes its drafting (`speculative`, `draft_max`); a planned lane
//! runs the drafting the last full calibration found for it.

use crate::fit;
use cc_protocol::contracts::serve::{Gpu, LaneSettings, PlanChange, Recipe, Want};
use std::collections::{BTreeMap, BTreeSet};

/// How long a model of a GPU whose lanes take turns stays loaded without a call.
const TAKING_TURNS_IDLE_UNLOAD_S: u32 = 300;

/// What a full calibration found for a lane: its drafting.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Tuned {
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

    /// Place `recipe` on `gpu` at the app's own sizes when it fits there. True when placed;
    /// `taking_turns` when it fits the GPU alone but not beside the lanes already on it.
    fn place(&mut self, recipe: &Recipe, gpu: &Gpu) -> Option<bool> {
        if self.declined.contains(&format!("{}-g{}", recipe.id, gpu.index)) {
            return None;
        }
        let on_device = runs_on(recipe, gpu);
        if on_device && recipe.min_gpu_bytes > 0 && gpu.memory_bytes < recipe.min_gpu_bytes {
            return None;
        }
        if !recipe.backends.is_empty() && !recipe.backends.iter().any(|backend| *backend == gpu.backend || backend == "cpu") {
            return None;
        }
        let id = self.lane_id(&recipe.id, gpu.index);
        let tuned = self.calibrated.get(&id).cloned().unwrap_or_default();
        let base = LaneSettings { id, recipe: recipe.id.clone(), gpu: gpu.index, lend: true, speculative: tuned.speculative, draft_max: tuned.draft_max, ..LaneSettings::default() };
        // The sizes are written out so a change in what the apps need changes the lane. An engine
        // that fits itself is set none.
        let settings = if fit::engine_fits(recipe) { base } else { LaneSettings { slots: if recipe.slots_default > 0 { fit::slots_of(recipe, &base) } else { 0 }, ctx: fit::ctx_of(recipe, &base), ..base } };
        let mut taking_turns = false;
        if on_device {
            let bytes = fit::total(recipe, &settings);
            if bytes > crate::gpus::usable(gpu, self.reserve) {
                return None;
            }
            taking_turns = bytes > self.free(gpu);
            // A discrete GPU's memory is hard: a lane that does not fit beside the others is left out.
            if taking_turns && !gpu.unified {
                return None;
            }
            *self.used.entry(gpu.index).or_default() += bytes;
        }
        self.lanes.push(settings);
        Some(taking_turns)
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
/// GPU (the extractor on Vulkan) runs on the CPU and takes none of its memory.
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
    // 2. What the apps on this computer have selected: each model once, on the first GPU with
    // room beside the other lanes, else the first where it fits alone.
    let mut served = BTreeSet::new();
    for want in input.wants {
        if want.recipe.is_empty() || served.contains(&want.recipe) || room.lanes.iter().any(|lane| lane.recipe == want.recipe) {
            continue;
        }
        let Some(recipe) = room.recipe(&want.recipe).cloned() else { continue };
        let mut placed = None;
        for gpu in &gpus {
            if gpu.unified || room.free(gpu) >= fit::total(&recipe, &LaneSettings::default()) {
                placed = room.place(&recipe, gpu);
                if placed.is_some() {
                    break;
                }
            }
        }
        if placed.is_none() {
            for gpu in &gpus {
                placed = room.place(&recipe, gpu);
                if placed.is_some() {
                    break;
                }
            }
        }
        match placed {
            Some(taking_turns) => {
                served.insert(want.recipe.clone());
                if taking_turns {
                    room.notes.push(format!("{} takes turns with the other lanes: not all of them fit loaded at once.", recipe.name));
                }
            }
            None => room.notes.push(format!("{} uses {}, which does not fit on this computer.", want.app, recipe.name)),
        }
    }
    // Lanes that take turns give their memory back soon.
    let crowded: BTreeSet<u32> = room.gpus.iter().filter(|gpu| room.used.get(&gpu.index).copied().unwrap_or(0) > crate::gpus::usable(gpu, room.reserve)).map(|gpu| gpu.index).collect();
    for lane in room.lanes.iter_mut().filter(|lane| !lane.manual && crowded.contains(&lane.gpu)) {
        lane.idle_unload_s = Some(TAKING_TURNS_IDLE_UNLOAD_S);
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
