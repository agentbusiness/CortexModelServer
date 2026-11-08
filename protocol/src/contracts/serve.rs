//! `serve@1`: the model server of this computer and the cluster it sees.
//!
//! The serve plugin owns this computer's GPUs and the lanes that run on them: each lane is one
//! recipe (a model on one of our patched engines, with the exact settings of the app it comes
//! from) on one GPU, with its own slots, context and cache. It starts them through the frame
//! (`frame.lanes.set`), records them in the graph, plans them from the hardware and from what the
//! Cortex apps on this computer use, and adds up the whole mesh for the cluster view.
//!
//! The window and the command line use these methods on any cortexmodelserver of the mesh (a
//! remote one through `mesh.link`); the methods are the same everywhere.

use serde::{Deserialize, Serialize};

pub mod methods {
    /// `{}` -> [`super::ServeStatus`].
    pub const STATUS_GET: &str = "serve.status.get";
    /// `{}` -> [`super::GpusResult`].
    pub const GPUS_LIST: &str = "serve.gpus.list";
    /// `{}` -> [`super::RecipesResult`]: every recipe, with whether it can run here.
    pub const RECIPES_LIST: &str = "serve.recipes.list";
    /// `{}` -> [`super::LanesView`].
    pub const LANES_LIST: &str = "serve.lanes.list";
    /// [`super::LaneSettings`] -> [`super::LaneView`]. The lane stays as set (the plan keeps it).
    pub const LANES_CREATE: &str = "serve.lanes.create";
    /// [`super::LaneEditParams`] -> [`super::LaneView`]. The lane stays as set (the plan keeps it).
    pub const LANES_EDIT: &str = "serve.lanes.edit";
    /// [`super::LaneIdParams`] -> `{}`. The plan does not bring the lane back.
    pub const LANES_DELETE: &str = "serve.lanes.delete";
    /// [`super::FitParams`] -> [`super::Fit`]: whether a lane fits, and its memory.
    pub const FIT_GET: &str = "serve.fit.get";
    /// `{}` -> [`super::Plan`]: the automatic plan for this computer and how it differs from now.
    pub const PLAN_GET: &str = "serve.plan.get";
    /// `{}` -> [`super::LanesView`]: back to the plan's lanes (lanes set by hand become the plan's
    /// again). The plan otherwise runs by itself: at start and whenever the hardware or the apps on
    /// this computer change.
    pub const PLAN_RUN: &str = "serve.plan.run";
    /// [`super::CalibrateParams`] -> [`super::CalibrationResult`].
    pub const CALIBRATE_RUN: &str = "serve.calibrate.run";
    /// `{}` -> [`super::Cluster`]: the whole mesh, added up.
    pub const CLUSTER_GET: &str = "serve.cluster.get";
    /// [`super::HistoryParams`] -> [`super::History`].
    pub const HISTORY_READ: &str = "serve.history.read";
}

pub mod topics {
    /// About every 2 s while anyone subscribes: [`super::ServeSample`].
    pub const SAMPLE: &str = "serve.sample";
    /// Lanes, plan or recipes changed: [`super::LanesView`].
    pub const CHANGED: &str = "serve.changed";
}

/// Lane kinds.
pub mod kinds {
    pub const CHAT: &str = "chat";
    pub const EXTRACT: &str = "extract";
}

/// A GPU of this computer (on a Mac: its unified memory, one entry).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct Gpu {
    /// The device index the engines use (`--device`, `CUDA_VISIBLE_DEVICES`); 0 on a Mac.
    pub index: u32,
    pub name: String,
    /// `metal` | `cuda` | `vulkan`.
    pub backend: String,
    /// Memory models may use: the GPU's memory, or on a Mac the GPU's working-set limit.
    pub memory_bytes: u64,
    /// In use now (every process), when the hardware reports it.
    #[serde(default)]
    pub used_bytes: u64,
    /// Busy share 0..1 now, when reported.
    #[serde(default)]
    pub util: f64,
    #[serde(default)]
    pub power_w: f64,
    #[serde(default)]
    pub temp_c: f64,
    /// Unified memory shared with the CPU (Apple silicon).
    #[serde(default)]
    pub unified: bool,
    /// Stable identity (CUDA UUID, PCI bus id or the Mac's chip), for the graph.
    #[serde(default)]
    pub uuid: String,
    /// GPU cores, when known.
    #[serde(default)]
    pub cores: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct GpusResult {
    pub gpus: Vec<Gpu>,
}

/// One model on one engine with the settings of the app it comes from (shipped data,
/// `plugins/serve/data/recipes/*.toml`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct Recipe {
    pub id: String,
    pub name: String,
    pub engine: String,
    pub origin: String,
    pub kind: String,
    #[serde(default)]
    pub family: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub platforms: Vec<String>,
    #[serde(default)]
    pub backends: Vec<String>,
    #[serde(default)]
    pub repo: String,
    #[serde(default)]
    pub revision: String,
    #[serde(default)]
    pub weights_bytes: u64,
    #[serde(default)]
    pub compute_bytes: u64,
    #[serde(default)]
    pub kv_bytes_per_token: u64,
    #[serde(default)]
    pub min_gpu_bytes: u64,
    #[serde(default)]
    pub ctx_default: u32,
    /// The most context a lane may be set to; 0 = no bound (memory bounds it).
    #[serde(default)]
    pub ctx_max: u32,
    /// A shared pool in shares: the context one call books by default (`ClaimParams.tokens`
    /// books more). The pool is one share per slot, and a call bigger than a share gets its
    /// shares on top of the normal slots. 0 = the pool is not counted in shares.
    #[serde(default)]
    pub ctx_share: u32,
    /// `ctx` is each slot's window (the engine sizes its KV as ctx × slots); otherwise it is the
    /// one pool the slots share.
    #[serde(default)]
    pub ctx_per_slot: bool,
    /// The smallest share of a shared pool one slot needs (the engine refuses a smaller pool:
    /// a conversation's window plus its fixed state); 0 = no bound.
    #[serde(default)]
    pub ctx_min_per_slot: u32,
    /// The smallest context a lane of this recipe may run: the most any app that selected the
    /// model says it needs on its card. Set by the serve plugin, not in the recipe's file. A lane
    /// that cannot have it is not planned, and the app keeps its own model. 0 = no bound.
    #[serde(default)]
    pub ctx_min: u32,
    /// Memory the engine sets aside in its own budget but does not hold: added to the lane's
    /// memory limit, not charged against the GPU.
    #[serde(default)]
    pub reserve_bytes: u64,
    #[serde(default)]
    pub slots_default: u32,
    /// The most slots a lane may be set to; 0 = no bound (memory bounds them).
    #[serde(default)]
    pub slots_max: u32,
    #[serde(default)]
    pub vision: bool,
    #[serde(default)]
    pub speculative: String,
    /// The engine's config for this recipe.
    #[serde(default)]
    pub config: serde_json::Value,
    /// The engine config key that carries each lane setting ("" = not settable).
    #[serde(default)]
    pub keys: RecipeKeys,
    /// The automatic plan may pick it (a recipe for special work, such as a long report lane,
    /// is set by hand).
    #[serde(default = "yes")]
    pub auto: bool,
    /// It can run on this computer (platform, backend, an installed engine).
    #[serde(default)]
    pub available: bool,
    /// Why not, when it cannot.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub reason: String,
    /// Its weights are in `~/.cortex-models` here.
    #[serde(default)]
    pub downloaded: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct RecipeKeys {
    #[serde(default)]
    pub slots: String,
    #[serde(default)]
    pub ctx: String,
    #[serde(default)]
    pub cache_ram: String,
    #[serde(default)]
    pub checkpoints: String,
    #[serde(default)]
    pub kv_type: String,
    #[serde(default)]
    pub speculative: String,
    /// How many tokens the drafter proposes at most ("" = not settable).
    #[serde(default)]
    pub draft_max: String,
    #[serde(default)]
    pub idle_unload_s: String,
    #[serde(default)]
    pub memory_limit: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct RecipesResult {
    pub recipes: Vec<Recipe>,
}

/// What a lane runs and how (what the window and the command line set).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct LaneSettings {
    /// Empty on create: made from the recipe and the GPU.
    #[serde(default)]
    pub id: String,
    pub recipe: String,
    /// The GPU's index (0 on a Mac).
    #[serde(default)]
    pub gpu: u32,
    /// 0 = the recipe's default.
    #[serde(default)]
    pub slots: u32,
    /// Context tokens per lane; 0 = the recipe's default.
    #[serde(default)]
    pub ctx: u32,
    /// Prompt cache in host RAM, MiB; `None` = the recipe's default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_ram_mib: Option<u32>,
    /// Context checkpoints per slot; `None` = the recipe's default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoints: Option<u32>,
    /// `f16` | `q8_0` | `int8` …; `None` = the recipe's default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_type: Option<String>,
    /// Speculative decoding on or off; `None` = the recipe's default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speculative: Option<bool>,
    /// Tokens the drafter proposes at most; `None` = the recipe's default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draft_max: Option<u32>,
    /// Unload after this long idle (0 = never); `None` = the recipe's default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idle_unload_s: Option<u32>,
    /// Keep it loaded: never unloaded when idle, loaded at start.
    #[serde(default)]
    pub pinned: bool,
    /// Lent to the mesh (other computers may claim it).
    #[serde(default = "yes")]
    pub lend: bool,
    /// Extra mesh tags beside the recipe's.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Set by hand: the plan keeps it as it is and plans the rest around it.
    #[serde(default)]
    pub manual: bool,
}

fn yes() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct LaneEditParams {
    pub id: String,
    pub settings: LaneSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct LaneIdParams {
    pub id: String,
}

/// A lane as it is now.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct LaneView {
    pub settings: LaneSettings,
    /// The recipe's name, engine, origin and kind.
    pub name: String,
    pub engine: String,
    #[serde(default)]
    pub origin: String,
    pub kind: String,
    #[serde(default)]
    pub family: String,
    #[serde(default)]
    pub quant: String,
    /// The lane's instance: starting | probing | active | draining | stopped | crashed | failed …
    pub state: String,
    /// The model is in memory.
    #[serde(default)]
    pub loaded: bool,
    #[serde(default)]
    pub slots: u32,
    #[serde(default)]
    pub slots_busy: u32,
    #[serde(default)]
    pub queued: u32,
    #[serde(default)]
    pub ctx: u32,
    /// Generated tokens per second now, all slots.
    #[serde(default)]
    pub tok_s: f64,
    /// Tokens served (prompt + generated) in the last hour.
    #[serde(default)]
    pub tokens_hour: u64,
    #[serde(default)]
    pub requests_hour: u64,
    /// Planned memory of the lane (weights + KV + buffers).
    #[serde(default)]
    pub memory_bytes: u64,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub error: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct LanesView {
    pub lanes: Vec<LaneView>,
}

/// [`methods::FIT_GET`]: would this lane fit, beside the lanes already on its GPU?
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct FitParams {
    pub settings: LaneSettings,
    /// The lane it replaces (edits: its memory is free for the new settings).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub replaces: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct Fit {
    pub fits: bool,
    pub weights_bytes: u64,
    pub kv_bytes: u64,
    pub compute_bytes: u64,
    pub total_bytes: u64,
    /// The GPU's memory for models, and what the other lanes on it take.
    pub gpu_bytes: u64,
    pub others_bytes: u64,
    /// The most slots and context that fit with the rest as asked.
    pub max_slots: u32,
    pub max_ctx: u32,
    /// One sentence: "fits 4 slots at 32k", "needs 4.2 GB more".
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct PlanChange {
    /// add | remove | change
    pub op: String,
    pub lane: LaneSettings,
    pub why: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct Plan {
    pub lanes: Vec<LaneSettings>,
    pub changes: Vec<PlanChange>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct CalibrateParams {
    /// One lane, or every chat lane when empty.
    #[serde(default)]
    pub lane: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct Calibration {
    pub lane: String,
    pub recipe: String,
    /// Single-stream decode and prefill, tokens per second.
    pub decode_tok_s: f64,
    pub prefill_tok_s: f64,
    /// All slots busy at once: generated tokens per second in total.
    pub aggregate_tok_s: f64,
    pub ttft_ms: f64,
    pub at_ms: u64,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub error: String,
    /// Total output of the lane reloaded at each slot count tried, fewest first.
    #[serde(default)]
    pub sweep: Vec<SlotLoad>,
    /// The slots to run with: the fewest that give close to the highest total output, and when
    /// the lane shares its GPU with other lanes, measured with them busy too.
    #[serde(default)]
    pub best_slots: u32,
    /// The lane's total output at `best_slots` with the GPU's other lanes busy at once
    /// (0: not measured).
    #[serde(default)]
    pub together_tok_s: f64,
    /// The slots the lane runs with.
    #[serde(default)]
    pub current_slots: u32,
    /// `chat`, `embed` or `extract`. For the last two `aggregate_tok_s` is what the lane gets
    /// through (tokens read; words for an extractor), and decode, prefill and first token are 0.
    #[serde(default)]
    pub kind: String,
    /// Total output at the best slots with each drafting setting tried ("off", "on", or a depth).
    #[serde(default)]
    pub drafts: Vec<DraftLoad>,
    /// The drafting setting to run with ("": the lane has none that can be set).
    #[serde(default)]
    pub best_draft: String,
    /// Context tokens at `best_slots`, so each conversation keeps the room it had.
    #[serde(default)]
    pub ctx: u32,
    /// The GPU this lane was moved to, because it got more done there.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moved_to: Option<u32>,
    /// Why no higher slot count was tried, when one was refused or failed.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub sweep_note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct DraftLoad {
    pub label: String,
    pub tok_s: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct SlotLoad {
    pub slots: u32,
    /// Generated tokens per second in total, wall clock.
    pub tok_s: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct CalibrationResult {
    pub results: Vec<Calibration>,
    /// GPUs whose lanes were measured together.
    #[serde(default)]
    pub gpus: Vec<GpuShare>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct GpuShare {
    pub gpu: u32,
    /// With its lanes busy at once: the sum of each lane's output over its best alone.
    /// 1.0 is a GPU that splits its time with nothing lost; below, the lanes get in each other's
    /// way; above, they run side by side.
    pub kept: f64,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub error: String,
}

/// This computer's server now.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ServeStatus {
    pub node: String,
    pub name: String,
    pub platform: String,
    pub gpus: Vec<Gpu>,
    pub lanes: Vec<LaneView>,
    /// Installed engines and their versions.
    pub engines: Vec<Engine>,
    /// What the Cortex apps on this computer use, served here first.
    pub wants: Vec<Want>,
    /// This computer's throughput and load now.
    #[serde(default)]
    pub totals: Totals,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct Engine {
    pub id: String,
    pub version: String,
    pub enabled: bool,
}

/// A model a Cortex app on this computer uses (read from its card).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct Want {
    pub app: String,
    pub model_id: String,
    pub family: String,
    pub tags: Vec<String>,
    /// The recipe that serves it here ("" when none does).
    #[serde(default)]
    pub recipe: String,
    /// The context the app says it needs (`ModelCard::ctx_needed`).
    #[serde(default)]
    pub ctx_needed: u32,
    /// The app has it in memory itself: it is left to the app (no second copy), and its memory
    /// is not planned for lanes.
    #[serde(default)]
    pub loaded: bool,
}

/// Live readings of one computer's server (`serve.sample`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ServeSample {
    pub at_ms: u64,
    pub gpus: Vec<Gpu>,
    pub lanes: Vec<LaneView>,
    pub totals: Totals,
}

/// Throughput and load added up (one computer or the whole mesh).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct Totals {
    /// Busy share of GPU power 0..1, each GPU weighted by its memory.
    pub gpu_util: f64,
    pub gpu_used_bytes: u64,
    pub gpu_bytes: u64,
    pub tok_s: f64,
    pub prefill_tok_s: f64,
    pub tokens_hour: u64,
    pub tokens_day: u64,
    pub requests_min: f64,
    /// Median time to first token over the last minutes, ms.
    pub ttft_ms: f64,
    pub slots_busy: u32,
    pub slots: u32,
    pub queued: u32,
    pub lanes_serving: u32,
    pub lanes_loading: u32,
    pub lanes_failed: u32,
    pub power_w: f64,
}

/// The whole mesh, added up from every computer's card.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct Cluster {
    pub totals: Totals,
    pub servers_online: u32,
    pub servers: u32,
    /// Computers whose GPUs report readings (the others count only by their models).
    pub reporting: u32,
    pub gpus: u32,
    pub updated_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct HistoryParams {
    /// `cluster` | `node` | `lane`
    pub scope: String,
    /// The lane id for `lane`.
    #[serde(default)]
    pub id: String,
    /// How far back, seconds (3600, 86400, 604800).
    pub range_s: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct HistoryPoint {
    pub at_ms: u64,
    pub totals: Totals,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct History {
    pub points: Vec<HistoryPoint>,
}
