//! The automatic plan: which lanes this computer runs.
//!
//! 1. Lanes set by hand stay as they are (their memory is taken first).
//! 2. What the Cortex apps on this computer use comes next: for each model an app's card lists,
//!    the recipe of the same model (same weights first, else the same family and tag), so the
//!    app finds it loaded here and loads no copy of its own (the standard's "a model already in
//!    memory on this computer is used before one is loaded"). A model an app has loaded itself
//!    stays the app's: no second copy, and its memory is taken before anything is planned.
//! 3. Then each GPU, largest first, by its size: a coding model on a GPU of 24 GB or more (the
//!    largest recipe that fits), the office model on one of 9 GB or more (Gemma 4 E4B, else E2B
//!    on a small one), each once per GPU.
//! 4. Then once per computer, on the GPU with the most room left: the rapid model, an embedder
//!    and the extractor.
//!
//! Every lane is fitted (weights + KV + buffers, beside the lanes already on its GPU) and its
//! context shrunk to fit when needed; one that does not fit is left out with a note. No GPU: no
//! lanes (the standard's GPU rule).

use crate::fit;
use cc_protocol::contracts::serve::{Gpu, LaneSettings, PlanChange, Recipe, Want, kinds};
use std::collections::{BTreeMap, BTreeSet};

const GIB: u64 = 1 << 30;

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
        if recipe.min_gpu_bytes > 0 && gpu.memory_bytes < recipe.min_gpu_bytes {
            return false;
        }
        if !recipe.backends.is_empty() && !recipe.backends.iter().any(|backend| *backend == gpu.backend || backend == "cpu") {
            return false;
        }
        let free = self.free(gpu);
        let slots = recipe.slots_default.max(1);
        let windows = if recipe.ctx_per_slot { slots as u64 } else { 1 };
        let per_window = Recipe { kv_bytes_per_token: recipe.kv_bytes_per_token * windows, ..recipe.clone() };
        let Some(ctx) = fit::best_ctx(&per_window, free) else {
            return false;
        };
        // A shared pool gives every slot at least the share the engine needs: fewer slots
        // when the pool came out small.
        let slots = if recipe.ctx_min_per_slot > 0 && !recipe.ctx_per_slot { slots.min(ctx / recipe.ctx_min_per_slot) } else { slots };
        if slots == 0 {
            return false;
        }
        let slots = if recipe.slots_default == 0 { 0 } else { slots };
        let mut settings = LaneSettings { id: String::new(), recipe: recipe.id.clone(), gpu: gpu.index, slots, ctx, lend: true, ..LaneSettings::default() };
        settings.id = self.lane_id(&recipe.id, gpu.index);
        let bytes = fit::total(recipe, &settings);
        if bytes > free {
            return false;
        }
        *self.used.entry(gpu.index).or_default() += bytes;
        self.lanes.push(settings);
        true
    }

    fn lane_id(&self, recipe: &str, gpu: u32) -> String {
        let base: String = format!("{recipe}-g{gpu}").chars().map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '-' }).collect();
        let base = base.trim_matches('-').chars().take(60).collect::<String>();
        let mut id = base.clone();
        let mut n = 2;
        while self.lanes.iter().any(|lane| lane.id == id) {
            id = format!("{base}-{n}");
            n += 1;
        }
        id
    }

    fn has(&self, test: impl Fn(&Recipe) -> bool) -> bool {
        self.lanes.iter().any(|lane| self.recipe(&lane.recipe).is_some_and(&test))
    }

    fn on_gpu(&self, gpu: u32, test: impl Fn(&Recipe) -> bool) -> bool {
        self.lanes.iter().any(|lane| lane.gpu == gpu && self.recipe(&lane.recipe).is_some_and(&test))
    }
}

fn tagged(recipe: &Recipe, tag: &str) -> bool {
    recipe.tags.iter().any(|t| t == tag)
}

/// The candidates for a role, best first: available, picked automatically, then the engine
/// order of the platform (on a Mac uzu first, the engine CortexCode runs its model on), then
/// larger weights.
fn candidates<'a>(recipes: &'a [Recipe], test: impl Fn(&Recipe) -> bool) -> Vec<&'a Recipe> {
    let mut out: Vec<&Recipe> = recipes.iter().filter(|recipe| recipe.available && recipe.auto && test(recipe)).collect();
    let engine_rank = |engine: &str| match engine {
        "model-uzu" => 0,
        "model-llama" => 1,
        "model-llama-agent" => 2,
        "model-bonsai" => 3,
        _ => 4,
    };
    out.sort_by(|a, b| engine_rank(&a.engine).cmp(&engine_rank(&b.engine)).then(b.weights_bytes.cmp(&a.weights_bytes)));
    out
}

pub fn plan(input: Input) -> Output {
    let mut gpus: Vec<&Gpu> = input.gpus.iter().collect();
    gpus.sort_by(|a, b| b.memory_bytes.cmp(&a.memory_bytes).then(a.index.cmp(&b.index)));
    let mut room = Room { recipes: input.recipes, gpus: input.gpus, reserve: input.reserve, used: input.taken.clone(), lanes: Vec::new(), notes: Vec::new(), declined: input.declined };
    if gpus.is_empty() {
        room.notes.push("No GPU: this computer serves no model (models run on Metal, CUDA or Vulkan only).".into());
        return Output { lanes: input.manual, notes: room.notes };
    }
    // 1. Lanes set by hand.
    for lane in input.manual {
        if let Some(recipe) = room.recipe(&lane.recipe).cloned() {
            *room.used.entry(lane.gpu).or_default() += fit::total(&recipe, &lane);
        }
        room.lanes.push(lane);
    }
    // What the apps here already serve from memory: those tags need no lane (one copy per computer).
    let covered: BTreeSet<String> = input.wants.iter().filter(|want| want.loaded).flat_map(|want| want.tags.iter().cloned()).collect();
    let covered_recipes: BTreeSet<String> = input.wants.iter().filter(|want| want.loaded && !want.recipe.is_empty()).map(|want| want.recipe.clone()).collect();
    // 2. What the apps on this computer use.
    let mut served_wants = BTreeSet::new();
    for want in input.wants {
        // An app that has the model in memory itself keeps serving it: no second copy.
        if want.loaded || want.recipe.is_empty() || covered_recipes.contains(&want.recipe) || want.tags.iter().any(|tag| covered.contains(tag)) || served_wants.contains(&want.recipe) || room.lanes.iter().any(|lane| lane.recipe == want.recipe) {
            continue;
        }
        let Some(recipe) = room.recipe(&want.recipe).cloned() else { continue };
        let placed = gpus.iter().any(|gpu| room.place(&recipe, gpu));
        if placed {
            served_wants.insert(want.recipe.clone());
        } else {
            room.notes.push(format!("{} uses {}, which does not fit beside the other lanes here.", want.app, recipe.name));
        }
    }
    // 3. Per GPU: a coding model on a large one, the office model on every one that holds it.
    for gpu in &gpus {
        let gpu = *gpu;
        if gpu.memory_bytes >= 24 * GIB && !covered.contains("coding") && !room.on_gpu(gpu.index, |r| tagged(r, "coding")) {
            for recipe in candidates(input.recipes, |r| r.kind == kinds::CHAT && tagged(r, "coding")) {
                if room.place(recipe, gpu) {
                    break;
                }
            }
        }
        if !covered.contains("office") && !room.on_gpu(gpu.index, |r| tagged(r, "office")) {
            let office = candidates(input.recipes, |r| r.kind == kinds::CHAT && tagged(r, "office"));
            // The office model answers the same whichever computer lends it: the E4B first,
            // the smaller E2B only where the E4B does not fit.
            let mut sorted = office.clone();
            sorted.sort_by_key(|recipe| (!recipe.id.contains("e4b"), !recipe.id.contains("e2b")));
            for recipe in sorted {
                if room.place(recipe, gpu) {
                    break;
                }
            }
        }
    }
    // 4. Once per computer, where there is most room.
    let once: [(&str, Box<dyn Fn(&Recipe) -> bool>); 3] = [
        ("rapid", Box::new(|r: &Recipe| r.kind == kinds::CHAT && tagged(r, "rapid"))),
        ("embed", Box::new(|r: &Recipe| r.kind == kinds::EMBED)),
        ("extract", Box::new(|r: &Recipe| r.kind == kinds::EXTRACT)),
    ];
    for (role, test) in once.iter() {
        if room.has(|r| test(r)) {
            continue;
        }
        // An app here keeps this kind of model loaded itself (its tags are a recipe's of the role).
        if input.recipes.iter().any(|r| test(r) && r.tags.iter().any(|tag| covered.contains(tag))) {
            continue;
        }
        let mut by_room: Vec<&Gpu> = gpus.clone();
        by_room.sort_by_key(|gpu| std::cmp::Reverse(room.free(gpu)));
        let mut placed = false;
        for recipe in candidates(input.recipes, |r| test(r)) {
            if by_room.iter().any(|gpu| room.place(recipe, gpu)) {
                placed = true;
                break;
            }
        }
        if !placed && candidates(input.recipes, |r| test(r)).is_empty() {
            room.notes.push(format!("No {role} recipe runs on this computer."));
        }
    }
    let _ = room.gpus;
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
