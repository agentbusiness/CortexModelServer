//! The model server: lanes, their records, their live readings, the plan, and the cluster.
//!
//! Everything kept is in the graph (the memory plugin), written through `graph`'s ordered
//! outbox exactly as CortexTerm's plugins do:
//!
//! - `(:Gpu {key: uuid})-[:ON]->(:Machine)`: name, backend, index; memory is temporal.
//! - `(:Engine {key: plugin})-[:ON]->(:Machine)`: version is temporal.
//! - `(:Recipe {key: id})-[:ON]->(:Machine)`: name, engine, origin, kind (the shipped recipe it is).
//! - `(:Lane {key: id})-[:ON]->(:Machine)`, `-[:ON_GPU]->(:Gpu)`, `-[:RUNS]->(:Recipe)`,
//!   `-[:IN]->(:Engine)`: every setting is temporal, so each change keeps its history; a removed
//!   lane is retired.
//! - `(:Calibration {key: lane})-[:OF]->(:Lane)`: measured speeds, temporal.
//! - Streams: `model.served` (one record per served call, 30 days), `model.lanes.events` (lane
//!   added, changed, removed, by whom; 365 days), `serve.minutes` (this computer each minute,
//!   90 days), `serve.cluster.minutes` (the mesh each minute, 90 days).
//! - `(:Declined {key: <recipe>-g<gpu>})-[:ON]->(:Machine)`: a planned lane removed by hand; the plan
//!   does not bring it back (until the plan is reset).
//!
//! Live readings (`serve.sample`, every `sample_s`) are not stored per sample, as CortexTerm's
//! `system.sample`; what is kept is the minute summary.

use crate::graph::{self, Batch, Props, r};
use crate::{fit, gpus, plan, recipes, stats, util};
use cc_protocol::contracts::frame::{LaneSpec, LanesResult, LanesSetParams};
use cc_protocol::contracts::mesh::{Card, PeersResult};
use cc_protocol::contracts::model::{LiveMetrics, ModelInfo, Progress, SlotsResult};
use cc_protocol::contracts::serve::*;
use cc_protocol::contracts::lifecycle::{ProbeCheck, ProbeReport};
use cc_protocol::{Ctx, RpcError, codes};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

const SOFTWARE: &str = "cortexmodelserver";

/// What a lane is doing now (read from its engine and the frame).
#[derive(Debug, Clone, Default)]
struct Live {
    state: String,
    loaded: bool,
    slots: u32,
    busy: u32,
    queued: u32,
    ctx: u32,
    tok_s: f64,
    prefill_tok_s: f64,
    family: String,
    quant: String,
}

#[derive(Default)]
struct State {
    recipes: Vec<Recipe>,
    hardware: gpus::Hardware,
    gpus: Vec<Gpu>,
    lanes: Vec<LaneSettings>,
    /// Planned lanes removed by hand (`<recipe>-g<gpu>`): the plan leaves them out.
    declined: BTreeSet<String>,
    live: HashMap<String, Live>,
    wants: Vec<Want>,
    engines: Vec<Engine>,
    plan_notes: Vec<String>,
    cluster: Cluster,
    loaded: bool,
    /// Settings a calibration is trying now, by lane. Never recorded: the lanes' settings stay as
    /// they are, and `apply` runs these until they are cleared.
    trial: BTreeMap<String, Trial>,
    /// What the last full calibration found, by lane; the plan runs it.
    calibrated: BTreeMap<String, plan::Tuned>,
}

pub struct Server {
    ctx: OnceLock<Ctx>,
    home: PathBuf,
    recipes_dir: PathBuf,
    sample: Duration,
    reserve: u64,
    state: Mutex<State>,
    stats: Mutex<stats::Stats>,
    started: std::sync::atomic::AtomicBool,
    /// Lanes being warmed now, and when each was last tried (ms).
    warming: Mutex<HashMap<String, u64>>,
    calibrating: std::sync::atomic::AtomicBool,
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn invalid(message: impl Into<String>) -> RpcError {
    RpcError::invalid_params(message.into())
}

fn parse<T: serde::de::DeserializeOwned>(params: Value) -> Result<T, RpcError> {
    let params = if params.is_null() { json!({}) } else { params };
    serde_json::from_value(params).map_err(|error| invalid(error.to_string()))
}

fn to_value<T: serde::Serialize>(value: &T) -> Result<Value, RpcError> {
    serde_json::to_value(value).map_err(|error| RpcError::internal(error.to_string()))
}

impl Server {
    pub fn new(home: PathBuf, recipes_dir: PathBuf) -> Arc<Server> {
        Arc::new(Server {
            ctx: OnceLock::new(),
            home,
            recipes_dir,
            sample: Duration::from_secs(2),
            reserve: 2 << 30,
            state: Mutex::new(State::default()),
            stats: Mutex::new(stats::Stats::default()),
            started: std::sync::atomic::AtomicBool::new(false),
            warming: Mutex::new(HashMap::new()),
            calibrating: std::sync::atomic::AtomicBool::new(false),
        })
    }

    pub fn attach(&self, ctx: &Ctx) {
        let _ = self.ctx.set(ctx.clone());
    }

    fn ctx(&self) -> Result<&Ctx, RpcError> {
        self.ctx.get().ok_or_else(|| RpcError::new(codes::UNAVAILABLE, "the server is starting").retryable())
    }

    fn config(&self) -> Value {
        self.ctx.get().map(Ctx::config).unwrap_or(Value::Null)
    }

    fn reserve(&self) -> u64 {
        self.config().get("reserve_bytes").and_then(Value::as_u64).unwrap_or(self.reserve)
    }

    fn sample_every(&self) -> Duration {
        self.config().get("sample_s").and_then(Value::as_u64).map(|s| Duration::from_secs(s.clamp(1, 60))).unwrap_or(self.sample)
    }

    pub fn probe(&self) -> ProbeReport {
        let recipes = recipes::load(&self.recipes_dir);
        let ok = !recipes.is_empty();
        ProbeReport {
            ok,
            checks: vec![ProbeCheck {
                name: "recipes".into(),
                ok,
                detail: if ok { format!("{} recipes", recipes.len()) } else { format!("no recipe in {}", self.recipes_dir.display()) },
                ms: 0,
            }],
        }
    }

    // ---------------------------------------------------------------------------------------
    // Start

    /// Load what is kept, measure the hardware, run the lanes, then keep sampling.
    pub fn start(self: &Arc<Self>, stop: CancellationToken) {
        if self.started.swap(true, std::sync::atomic::Ordering::SeqCst) {
            return;
        }
        let server = self.clone();
        tokio::spawn(async move {
            let Some(ctx) = server.ctx.get().cloned() else { return };
            graph::connect(ctx.clone()).await;
            server.load().await;
            graph::set_ready();
            server.refresh_hardware().await;
            server.refresh_wants().await;
            server.follow_plan("start").await;
            if let Err(error) = server.apply().await {
                tracing::warn!(error = %error.message, "the lanes could not be set at start; trying again with the next minute");
            }
            server.run(stop).await;
        });
    }

    async fn load(&self) {
        let mut recipes = recipes::load(&self.recipes_dir);
        let declined: BTreeSet<String> = graph::query_all("MATCH (d:Declined) WHERE d.ended_ms IS NULL AND (d.state IS NULL OR d.state <> 'gone') RETURN d.key", json!({}))
            .await
            .map(|rows| rows.iter().filter_map(|row| row.first().and_then(Value::as_str).map(str::to_owned)).collect())
            .unwrap_or_default();
        let wants: Vec<Want> = graph::query_all("MATCH (s:Selection) WHERE s.ended_ms IS NULL AND (s.state IS NULL OR s.state <> 'gone') RETURN s.app, s.model_id, s.family, s.tags, s.recipe, s.ctx_needed", json!({}))
            .await
            .map(|rows| {
                rows.iter()
                    .map(|row| {
                        let text = |column: usize| row.get(column).map(graph::text).unwrap_or_default();
                        Want { app: text(0), model_id: text(1), family: text(2), tags: row.get(3).map(graph::texts).unwrap_or_default(), recipe: text(4), ctx_needed: row.get(5).and_then(Value::as_u64).unwrap_or(0) as u32, loaded: false }
                    })
                    .collect()
            })
            .unwrap_or_default();
        apply_needs(&mut recipes, &wants);
        let lanes = match graph::query_all(
            "MATCH (l:Lane) WHERE l.ended_ms IS NULL AND (l.state IS NULL OR l.state <> 'gone') RETURN l.key, l.recipe, l.gpu, l.slots, l.ctx, l.cache_ram_mib, l.checkpoints, l.kv_type, l.speculative, l.idle_unload_s, l.pinned, l.lend, l.tags, l.manual, l.draft_max ORDER BY l.key",
            json!({}),
        )
        .await
        {
            Ok(rows) => rows.iter().filter_map(|row| lane_of_row(row)).collect(),
            Err(error) => {
                tracing::info!(%error, "no lanes kept yet");
                Vec::new()
            }
        };
        recipes.sort_by(|a, b| a.engine.cmp(&b.engine).then(a.id.cmp(&b.id)));
        let found = graph::query_all("MATCH (c:Calibration) WHERE c.ended_ms IS NULL AND (c.state IS NULL OR c.state <> 'gone') RETURN c.key, c.plan_speculative, c.plan_draft", json!({})).await.unwrap_or_default();
        let mut calibrated = BTreeMap::new();
        for row in &found {
            let key = row.first().and_then(Value::as_str).unwrap_or_default();
            let draft = row.get(2).and_then(Value::as_u64).unwrap_or(0) as u32;
            if !key.is_empty() {
                calibrated.insert(key.to_owned(), plan::Tuned { speculative: (row.get(1).and_then(Value::as_str) == Some("off")).then_some(false), draft_max: (draft > 0).then_some(draft) });
            }
        }
        {
            let mut state = lock(&self.state);
            state.calibrated = calibrated;
            state.recipes = recipes;
            state.lanes = lanes;
            state.declined = declined;
            state.wants = wants;
            state.loaded = true;
        }
        // What the lanes served in the last day.
        let since = util::now_ms().saturating_sub(86_400_000);
        if let Ok(entries) = graph::stream_read("model.served", since).await {
            let mut stats = lock(&self.stats);
            for entry in entries {
                stats.restore(entry.at_ms, &entry.value);
            }
        }
        self.record_recipes();
    }

    async fn refresh_hardware(&self) {
        let Ok(ctx) = self.ctx() else { return };
        let hardware = gpus::measure(ctx, &self.home).await;
        let engines = engines(ctx).await;
        {
            let mut state = lock(&self.state);
            let backends: BTreeSet<String> = hardware.gpus.iter().map(|gpu| gpu.backend.clone()).collect();
            let installed: BTreeSet<String> = engines.iter().filter(|engine| engine.enabled).map(|engine| engine.id.clone()).collect();
            recipes::mark(&mut state.recipes, &backends, &installed);
            state.gpus = hardware.gpus.clone();
            state.hardware = hardware.clone();
            state.engines = engines.clone();
        }
        let Some(machine) = graph::machine() else { return };
        let mut batch = Batch::new();
        for gpu in &hardware.gpus {
            let unit = batch.unit();
            unit.node("Gpu", gpu.uuid.clone()).p("name", gpu.name.clone()).p("backend", gpu.backend.clone()).p("index", gpu.index as u64).p("unified", gpu.unified).t("memory_bytes", gpu.memory_bytes);
            unit.edge(r("Gpu", gpu.uuid.clone()), "ON", machine.clone());
        }
        for engine in &engines {
            let unit = batch.unit();
            unit.node("Engine", engine.id.clone()).t("version", engine.version.clone()).t("enabled", engine.enabled);
            unit.edge(r("Engine", engine.id.clone()), "ON", machine.clone());
        }
        graph::put(batch);
    }

    fn record_recipes(&self) {
        let Some(machine) = graph::machine() else { return };
        let recipes = lock(&self.state).recipes.clone();
        let mut batch = Batch::new();
        for recipe in &recipes {
            let unit = batch.unit();
            unit.node("Recipe", recipe.id.clone()).p("name", recipe.name.clone()).p("engine", recipe.engine.clone()).p("origin", recipe.origin.clone()).p("kind", recipe.kind.clone()).ps("repo", &recipe.repo).ps("revision", &recipe.revision);
            unit.edge(r("Recipe", recipe.id.clone()), "ON", machine.clone());
        }
        graph::put(batch);
    }

    /// What the Cortex apps on this computer have selected: the models their cards list (the cards
    /// of local members), each matched to the recipe that serves the same model. An app's card
    /// replaces what it selected before; an app that is away keeps its selection, so its lanes
    /// stay ready for when it comes back.
    async fn refresh_wants(&self) {
        let Ok(ctx) = self.ctx() else { return };
        let Some(peers) = peers(ctx).await else { return };
        let recipes = lock(&self.state).recipes.clone();
        let mut wants = Vec::new();
        let mut named = BTreeSet::new();
        for peer in &peers.peers {
            if !peer.member.local {
                continue;
            }
            let Some(card) = &peer.card else { continue };
            if card.software == SOFTWARE {
                continue;
            }
            // A card with no model yet (the app is still starting) changes nothing.
            if card.models.is_empty() && card.uses.is_empty() {
                continue;
            }
            named.insert(card.software.clone());
            // What it lends and what it uses from a lane here (a model it handed over is on its
            // card as used, and the lane must stay).
            let lent = card.models.iter().map(|model| (&model.model_id, &model.family, &model.tags, model.loaded, model.ctx_needed));
            let used = card.uses.iter().map(|model| (&model.model_id, &model.family, &model.tags, model.loaded, model.ctx_needed));
            for (model_id, family, tags, loaded, ctx_needed) in lent.chain(used) {
                let recipe = match_recipe(&recipes, model_id, family, tags).map(|recipe| recipe.id.clone()).unwrap_or_default();
                wants.push(Want { app: card.software.clone(), model_id: model_id.clone(), family: family.clone(), tags: tags.clone(), recipe, ctx_needed, loaded });
            }
        }
        let known = lock(&self.state).wants.clone();
        wants.extend(known.iter().filter(|want| !named.contains(&want.app)).map(|want| Want { loaded: false, ..want.clone() }));
        wants.sort_by(|a, b| (&a.app, &a.model_id).cmp(&(&b.app, &b.model_id)));
        wants.dedup();
        record_selections(&known, &wants);
        let mut state = lock(&self.state);
        apply_needs(&mut state.recipes, &wants);
        state.wants = wants;
    }

    // ---------------------------------------------------------------------------------------
    // Lanes

    fn recipe(&self, id: &str) -> Option<Recipe> {
        lock(&self.state).recipes.iter().find(|recipe| recipe.id == id).cloned()
    }

    fn gpu(&self, index: u32) -> Option<Gpu> {
        lock(&self.state).gpus.iter().find(|gpu| gpu.index == index).cloned()
    }

    /// The engine config and environment of a lane: the recipe's config with the lane's settings
    /// written through the recipe's keys, and the GPU selection.
    fn spec(&self, lane: &LaneSettings, recipe: &Recipe, gpu: Option<&Gpu>) -> LaneSpec {
        let mut config = recipe.config.clone();
        if !config.is_object() {
            config = json!({});
        }
        let keys = &recipe.keys;
        let mut set = |key: &str, value: Value| {
            if !key.is_empty()
                && let Value::Object(fields) = &mut config
            {
                fields.insert(key.to_owned(), value);
            }
        };
        // Slots left at 0 by a recipe whose engine fits them itself stay the engine's choice.
        if lane.slots > 0 || recipe.slots_default > 0 {
            set(&keys.slots, json!(fit::slots_of(recipe, lane)));
        }
        let ctx = fit::ctx_of(recipe, lane);
        if ctx > 0 {
            set(&keys.ctx, json!(ctx));
        }
        if let Some(mib) = lane.cache_ram_mib {
            set(&keys.cache_ram, json!(mib));
        }
        if let Some(checkpoints) = lane.checkpoints {
            set(&keys.checkpoints, json!(checkpoints));
        }
        if let Some(kv) = &lane.kv_type {
            set(&keys.kv_type, json!(kv));
        }
        if let Some(depth) = lane.draft_max {
            set(&keys.draft_max, json!(depth));
        }
        if lane.speculative == Some(false) && !keys.speculative.is_empty() {
            let off = match recipe.config.get(&keys.speculative) {
                Some(Value::Bool(_)) => json!(false),
                Some(Value::Number(_)) => json!(0),
                _ => json!("off"),
            };
            set(&keys.speculative, off);
        }
        // A lane loads its model on the first call and unloads it after an hour without one, unless
        // it is pinned, set to never (0), or set to another time.
        let idle = match lane.idle_unload_s {
            _ if lane.pinned => 315_360_000,
            Some(0) => 315_360_000,
            Some(seconds) => u64::from(seconds),
            None => 3600,
        };
        set(&keys.idle_unload_s, json!(idle));
        // Every lane gets a memory ceiling. An engine that sizes its own pool gets the GPU less
        // the other lanes on it, as the app alone would give it less what shares the GPU; a lane
        // whose context bounds what it allocates gets room for what the engine keeps aside beyond
        // the recipe's estimate (its own reserves, snapshots), a quarter and 1 GiB more.
        let limit = if fit::engine_fits(recipe) {
            gpu.map(|gpu| gpus::usable(gpu, self.reserve()).saturating_sub(self.others_bytes(lane, gpu))).unwrap_or(u64::MAX)
        } else {
            let planned = fit::total(recipe, lane);
            planned + planned / 4 + (1 << 30) + recipe.reserve_bytes
        };
        if limit < u64::MAX {
            set(&keys.memory_limit, json!(limit));
        }
        let env = gpu.map(gpus::lane_env).unwrap_or_default().into_iter().collect();
        LaneSpec { id: lane.id.clone(), plugin: recipe.engine.clone(), config, env, gpu: gpu.map(|gpu| gpu.index) }
    }

    /// Tell the frame which lanes run (all of them; unchanged ones keep serving).
    async fn apply(&self) -> Result<LanesResult, RpcError> {
        let ctx = self.ctx()?.clone();
        let (lanes, trial) = {
            let state = lock(&self.state);
            (state.lanes.clone(), state.trial.clone())
        };
        let mut specs = Vec::new();
        for lane in &lanes {
            let Some(recipe) = self.recipe(&lane.recipe) else {
                tracing::warn!(lane = %lane.id, recipe = %lane.recipe, "a lane names a recipe this version does not ship; it does not run");
                continue;
            };
            let mut lane = lane.clone();
            if let Some(trial) = trial.get(&lane.id) {
                trial.on(&mut lane);
            }
            let gpu = self.gpu(lane.gpu);
            specs.push(self.spec(&lane, &recipe, gpu.as_ref()));
        }
        let answer = ctx.call("frame.lanes.set", to_value(&LanesSetParams { lanes: specs })?).await?;
        let result: LanesResult = serde_json::from_value(answer).map_err(|error| RpcError::internal(error.to_string()))?;
        self.frame_states(&result);
        self.publish_changed();
        Ok(result)
    }

    fn frame_states(&self, result: &LanesResult) {
        let mut state = lock(&self.state);
        for lane in &result.lanes {
            state.live.entry(lane.id.clone()).or_default().state = lane.state.clone();
        }
        let known: BTreeSet<String> = result.lanes.iter().map(|lane| lane.id.clone()).collect();
        state.live.retain(|id, _| known.contains(id));
    }

    /// Record a lane's settings (every setting temporal: its history stays).
    fn record_lane(&self, lane: &LaneSettings, recipe: &Recipe, by: &str, op: &str) {
        let gpu_uuid = self.gpu(lane.gpu).map(|gpu| gpu.uuid);
        if let Some(machine) = graph::machine() {
            let mut batch = Batch::new();
            let unit = batch.unit();
            let node = unit.node("Lane", lane.id.clone());
            node.p("recipe", lane.recipe.clone())
                .p("engine", recipe.engine.clone())
                .t("gpu", lane.gpu as u64)
                .t("slots", lane.slots as u64)
                .t("ctx", lane.ctx as u64)
                .t("pinned", lane.pinned)
                .t("lend", lane.lend)
                .t("manual", lane.manual)
                .pl("tags", lane.tags.clone());
            node.t("cache_ram_mib", lane.cache_ram_mib.map(|v| json!(v)).unwrap_or(json!(-1)));
            node.t("checkpoints", lane.checkpoints.map(|v| json!(v)).unwrap_or(json!(-1)));
            node.t("kv_type", lane.kv_type.clone().unwrap_or_default());
            node.t("speculative", match lane.speculative {
                Some(true) => "on",
                Some(false) => "off",
                None => "",
            });
            node.t("draft_max", lane.draft_max.map(|v| json!(v)).unwrap_or(json!(-1)));
            node.t("idle_unload_s", lane.idle_unload_s.map(|v| json!(v)).unwrap_or(json!(-1)));
            unit.edge(r("Lane", lane.id.clone()), "ON", machine.clone());
            unit.edge(r("Lane", lane.id.clone()), "RUNS", r("Recipe", recipe.id.clone()));
            unit.edge(r("Lane", lane.id.clone()), "IN", r("Engine", recipe.engine.clone()));
            if let Some(uuid) = gpu_uuid {
                unit.edge(r("Lane", lane.id.clone()), "ON_GPU", r("Gpu", uuid));
            }
            graph::put(batch);
        }
        graph::append("model.lanes.events", vec![json!({"op": op, "lane": lane.id, "recipe": lane.recipe, "gpu": lane.gpu, "slots": lane.slots, "ctx": lane.ctx, "by": by})], 365);
    }

    fn retire_lane(&self, lane: &LaneSettings, by: &str) {
        graph::retire(vec![r("Lane", lane.id.clone())], "removed");
        graph::append("model.lanes.events", vec![json!({"op": "remove", "lane": lane.id, "recipe": lane.recipe, "by": by})], 365);
        lock(&self.stats).forget(&lane.id);
    }

    /// A planned lane removed by hand: the plan leaves it out from now on.
    fn decline(&self, lane: &LaneSettings) {
        let key = format!("{}-g{}", lane.recipe, lane.gpu);
        lock(&self.state).declined.insert(key.clone());
        if let Some(machine) = graph::machine() {
            let mut batch = Batch::new();
            let unit = batch.unit();
            unit.node("Declined", key.clone()).p("recipe", lane.recipe.clone()).p("gpu", lane.gpu as u64);
            unit.edge(r("Declined", key), "ON", machine);
            graph::put(batch);
        }
    }

    /// Check `settings` and fill what is left to the recipe; the lane's id from the recipe and
    /// GPU when empty.
    fn settle(&self, mut settings: LaneSettings, replaces: &str) -> Result<(LaneSettings, Recipe, Fit), RpcError> {
        let recipe = self.recipe(&settings.recipe).ok_or_else(|| invalid(format!("no recipe `{}`", settings.recipe)))?;
        if !recipe.available {
            return Err(invalid(format!("{} cannot run here: {}", recipe.name, recipe.reason)));
        }
        let gpu = self.gpu(settings.gpu).ok_or_else(|| invalid(format!("this computer has no GPU {}", settings.gpu)))?;
        if settings.slots == 0 {
            settings.slots = recipe.slots_default;
        }
        if settings.slots > 0 && recipe.slots_max > 0 {
            settings.slots = settings.slots.min(recipe.slots_max);
        }
        if settings.ctx > 0 && recipe.ctx_max > 0 {
            settings.ctx = settings.ctx.min(recipe.ctx_max);
        }
        // An engine that sizes its own pool is set none.
        if fit::engine_fits(&recipe) {
            settings.slots = 0;
            settings.ctx = 0;
        }
        if settings.id.is_empty() {
            let lanes = lock(&self.state).lanes.clone();
            let base: String = format!("{}-g{}", recipe.id, gpu.index).chars().map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '-' }).collect();
            let base: String = base.trim_matches('-').chars().take(60).collect();
            let mut id = base.clone();
            let mut n = 2;
            while lanes.iter().any(|lane| lane.id == id) {
                id = format!("{base}-{n}");
                n += 1;
            }
            settings.id = id;
        }
        if !settings.id.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-') || settings.id.len() > 64 {
            return Err(invalid("a lane id is 1 to 64 of [a-z0-9-]"));
        }
        let fit = self.fit_of(&settings, &recipe, &gpu, replaces);
        Ok((settings, recipe, fit))
    }

    fn fit_of(&self, settings: &LaneSettings, recipe: &Recipe, gpu: &Gpu, replaces: &str) -> Fit {
        // Unified memory: the lanes load on a call and take turns, so a lane needs only the GPU.
        if gpu.unified {
            return fit::check(recipe, settings, gpu, gpus::usable(gpu, self.reserve()), 0);
        }
        let state = lock(&self.state);
        let others: u64 = state
            .lanes
            .iter()
            .filter(|lane| lane.gpu == gpu.index && lane.id != replaces && lane.id != settings.id)
            .filter_map(|lane| state.recipes.iter().find(|r| r.id == lane.recipe).map(|r| fit::total(r, lane)))
            .sum();
        fit::check(recipe, settings, gpu, gpus::usable(gpu, self.reserve()), others)
    }

    /// What the other lanes of `lane`'s GPU take.
    fn others_bytes(&self, lane: &LaneSettings, gpu: &Gpu) -> u64 {
        let state = lock(&self.state);
        state.lanes.iter().filter(|other| other.gpu == gpu.index && other.id != lane.id).filter_map(|other| state.recipes.iter().find(|r| r.id == other.recipe).map(|r| fit::total(r, other))).sum()
    }

    // ---------------------------------------------------------------------------------------
    // The plan

    fn planned(&self) -> plan::Output {
        let state = lock(&self.state);
        let manual: Vec<LaneSettings> = state.lanes.iter().filter(|lane| lane.manual).cloned().collect();
        // Memory other programs hold on each GPU: on a GPU card what it reports in use beyond this
        // server's loaded lanes and the models the apps here are about to let go of (they hand
        // over to this server); unified memory has no per-process reading.
        let mut taken = std::collections::BTreeMap::new();
        for gpu in &state.gpus {
            let bytes = if gpu.unified {
                0
            } else {
                let apps: u64 = state
                    .wants
                    .iter()
                    .filter(|want| want.loaded)
                    .filter_map(|want| state.recipes.iter().find(|r| r.id == want.recipe))
                    .map(|recipe| fit::total(recipe, &LaneSettings::default()))
                    .sum();
                let ours: u64 = state
                    .lanes
                    .iter()
                    .filter(|lane| lane.gpu == gpu.index && state.live.get(&lane.id).is_some_and(|live| live.loaded))
                    .filter_map(|lane| state.recipes.iter().find(|r| r.id == lane.recipe).map(|r| fit::total(r, lane)))
                    .sum();
                gpu.used_bytes.saturating_sub(ours).saturating_sub(apps)
            };
            taken.insert(gpu.index, bytes);
        }
        plan::plan(plan::Input { recipes: &state.recipes, gpus: &state.gpus, reserve: self.reserve(), wants: &state.wants, manual, taken, declined: &state.declined, calibrated: &state.calibrated })
    }

    /// Set the planned lanes when they differ from the running ones (manual lanes stay).
    async fn follow_plan(&self, by: &str) -> bool {
        let output = self.planned();
        let current = lock(&self.state).lanes.clone();
        let changes = plan::changes(&current, &output.lanes);
        lock(&self.state).plan_notes = output.notes.clone();
        if changes.is_empty() {
            return false;
        }
        for change in &changes {
            match change.op.as_str() {
                "remove" => self.retire_lane(&change.lane, by),
                op => {
                    if let Some(recipe) = self.recipe(&change.lane.recipe) {
                        self.record_lane(&change.lane, &recipe, by, op);
                    }
                }
            }
        }
        lock(&self.state).lanes = output.lanes;
        tracing::info!(changes = changes.len(), "the plan changed the lanes");
        true
    }

    // ---------------------------------------------------------------------------------------
    // Live readings

    async fn run(self: Arc<Self>, stop: CancellationToken) {
        let mut tick = tokio::time::interval(self.sample_every());
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let mut last_minute = util::now_ms() / 60_000;
        loop {
            tokio::select! {
                _ = stop.cancelled() => return,
                _ = tick.tick() => {}
            }
            self.poll().await;
            let minute = util::now_ms() / 60_000;
            if minute != last_minute {
                last_minute = minute;
                self.each_minute().await;
            }
        }
    }

    /// Read every lane's engine and the GPUs, and publish the sample.
    async fn poll(&self) {
        let Ok(ctx) = self.ctx().cloned() else { return };
        let hardware = lock(&self.state).hardware.clone();
        let gpus = gpus::read(&ctx, &hardware).await;
        let ids: Vec<String> = lock(&self.state).lanes.iter().map(|lane| lane.id.clone()).collect();
        let mut readings = Vec::new();
        for id in ids {
            let params = json!({ cc_protocol::contracts::model::PLUGIN_PARAM: id });
            let limit = Duration::from_secs(3);
            let slots: Option<SlotsResult> = tokio::time::timeout(limit, ctx.call("model.slots", params.clone())).await.ok().and_then(Result::ok).and_then(|v| serde_json::from_value(v).ok());
            let metrics: Option<LiveMetrics> = tokio::time::timeout(limit, ctx.call("model.metrics", params.clone())).await.ok().and_then(Result::ok).and_then(|v| serde_json::from_value(v).ok());
            let needs_info = lock(&self.state).live.get(&id).is_none_or(|live| live.family.is_empty() && live.quant.is_empty());
            let info: Option<ModelInfo> = if needs_info || slots.as_ref().is_some_and(|s| s.loaded) {
                tokio::time::timeout(limit, ctx.call("model.info", params)).await.ok().and_then(Result::ok).and_then(|v| serde_json::from_value(v).ok())
            } else {
                None
            };
            readings.push((id, slots, metrics, info));
        }
        {
            let mut state = lock(&self.state);
            for (id, slots, metrics, info) in readings {
                let live = state.live.entry(id).or_default();
                if let Some(slots) = slots {
                    live.loaded = slots.loaded;
                    live.slots = slots.slots.len() as u32;
                    live.busy = slots.busy;
                    live.queued = slots.queued;
                } else {
                    live.loaded = false;
                    live.busy = 0;
                    live.queued = 0;
                }
                match metrics {
                    Some(metrics) => {
                        live.tok_s = metrics.decode_tps_aggregate;
                        live.prefill_tok_s = metrics.prefill_tps;
                    }
                    None => {
                        live.tok_s = 0.0;
                        live.prefill_tok_s = 0.0;
                    }
                }
                if let Some(info) = info {
                    live.family = info.family;
                    live.quant = info.quant;
                    live.ctx = info.ctx_total;
                    if live.slots == 0 {
                        live.slots = info.slots;
                    }
                }
            }
            // On a Mac the GPU's memory in use is what the loaded lanes hold.
            let mut gpus = gpus;
            if let Some(gpu) = gpus.first_mut().filter(|gpu| gpu.unified) {
                gpu.used_bytes = state
                    .lanes
                    .iter()
                    .filter(|lane| state.live.get(&lane.id).is_some_and(|live| live.loaded))
                    .filter_map(|lane| state.recipes.iter().find(|r| r.id == lane.recipe).map(|r| fit::total(r, lane)))
                    .sum();
            }
            state.gpus = gpus;
        }
        let sample = self.sample_now();
        if let Ok(value) = serde_json::to_value(&sample) {
            ctx.publish(topics::SAMPLE, value);
        }
        self.warm(&ctx, &sample.lanes);
    }

    /// Load every pinned lane's model that is not in memory yet; every other lane loads on its
    /// first call. A chat lane gets a prewarm of a short system block (loads the weights, keeps no
    /// conversation), an extractor one small real call. A lane is tried again a
    /// minute after a warm-up that did not load it.
    fn warm(&self, ctx: &Ctx, lanes: &[LaneView]) {
        let now = util::now_ms();
        for lane in lanes {
            if !lane.settings.pinned || lane.loaded || lane.state != "active" {
                continue;
            }
            // An app here still holds a model of this lane's tags: it hands over first, so the
            // two are not in memory together. A call that comes before that loads the lane.
            {
                let state = lock(&self.state);
                let tags = state.recipes.iter().find(|recipe| recipe.id == lane.settings.recipe).map(|recipe| recipe.tags.clone()).unwrap_or_default();
                if state.wants.iter().any(|want| want.loaded && want.tags.iter().any(|tag| tags.contains(tag))) {
                    continue;
                }
            }
            let id = lane.settings.id.clone();
            {
                let mut warming = lock(&self.warming);
                if warming.get(&id).is_some_and(|at| now.saturating_sub(*at) < 60_000) {
                    continue;
                }
                warming.insert(id.clone(), now);
            }
            let (method, params) = match lane.kind.as_str() {
                kinds::EXTRACT => ("model.extract", json!({"plugin": id, "relex": {"text": "The server starts.", "kinds": ["thing"]}})),
                _ => ("model.prewarm", json!({"plugin": id, "cache_key": format!("serve-warm-{id}"), "messages": [{"role": "system", "content": "You are a helpful assistant."}], "shared_only": true, "idle_only": true})),
            };
            let ctx = ctx.clone();
            tokio::spawn(async move {
                match tokio::time::timeout(Duration::from_secs(900), ctx.call(method, params)).await {
                    Ok(Ok(_)) => tracing::info!(lane = %id, "the lane's model is loaded and warm"),
                    Ok(Err(error)) => tracing::warn!(lane = %id, error = %error.message, "the lane's model did not load; trying again in a minute"),
                    Err(_) => tracing::warn!(lane = %id, "the lane's warm-up did not finish in 15 minutes"),
                }
            });
        }
    }

    fn views(&self) -> Vec<LaneView> {
        let now = util::now_ms();
        let state = lock(&self.state);
        let stats = lock(&self.stats);
        state
            .lanes
            .iter()
            .map(|lane| {
                let recipe = state.recipes.iter().find(|r| r.id == lane.recipe);
                let live = state.live.get(&lane.id).cloned().unwrap_or_default();
                let (tokens_hour, requests_hour) = stats.tokens(&lane.id, now, 60);
                let error = if recipe.is_none() { format!("the recipe `{}` is not shipped with this version", lane.recipe) } else if live.state.starts_with("failed") || live.state.starts_with("waiting") || live.state.contains("disabled") { live.state.clone() } else { String::new() };
                LaneView {
                    settings: lane.clone(),
                    name: recipe.map(|r| r.name.clone()).unwrap_or_else(|| lane.recipe.clone()),
                    engine: recipe.map(|r| r.engine.clone()).unwrap_or_default(),
                    origin: recipe.map(|r| r.origin.clone()).unwrap_or_default(),
                    kind: recipe.map(|r| r.kind.clone()).unwrap_or_else(|| kinds::CHAT.into()),
                    family: if live.family.is_empty() { recipe.map(|r| r.family.clone()).unwrap_or_default() } else { live.family.clone() },
                    quant: live.quant.clone(),
                    state: if live.state.is_empty() { "starting".into() } else { live.state.clone() },
                    loaded: live.loaded,
                    slots: if live.slots > 0 { live.slots } else { recipe.map(|r| fit::slots_of(r, lane)).unwrap_or(lane.slots) },
                    slots_busy: live.busy,
                    queued: live.queued,
                    ctx: if live.ctx > 0 { live.ctx } else { recipe.map(|r| fit::ctx_of(r, lane)).unwrap_or(lane.ctx) },
                    tok_s: live.tok_s,
                    tokens_hour,
                    requests_hour,
                    memory_bytes: recipe.map(|r| fit::total(r, lane)).unwrap_or(0),
                    error,
                }
            })
            .collect()
    }

    fn totals(&self, views: &[LaneView], gpus: &[Gpu]) -> Totals {
        let now = util::now_ms();
        // Locks are taken one at a time here: `views` holds state then stats, so holding stats
        // while asking for state would deadlock with it.
        let prefill_tok_s: f64 = lock(&self.state).live.values().map(|live| live.prefill_tok_s).sum();
        let stats = lock(&self.stats);
        let gpu_bytes: u64 = gpus.iter().map(|gpu| gpu.memory_bytes).sum();
        let weighted: f64 = gpus.iter().map(|gpu| gpu.util * gpu.memory_bytes as f64).sum();
        let mut ttfts = Vec::new();
        let mut requests_5 = 0;
        let mut tokens_day = 0;
        for view in views {
            ttfts.extend(stats.ttfts(&view.settings.id, now, 5));
            requests_5 += stats.tokens(&view.settings.id, now, 5).1;
            tokens_day += stats.tokens(&view.settings.id, now, 24 * 60).0;
        }
        let state_of = |view: &LaneView| view.state.as_str().to_owned();
        Totals {
            gpu_util: if gpu_bytes > 0 { weighted / gpu_bytes as f64 } else { 0.0 },
            gpu_used_bytes: gpus.iter().map(|gpu| gpu.used_bytes).sum(),
            gpu_bytes,
            tok_s: views.iter().map(|view| view.tok_s).sum(),
            prefill_tok_s,
            tokens_hour: views.iter().map(|view| view.tokens_hour).sum(),
            tokens_day,
            requests_min: requests_5 as f64 / 5.0,
            ttft_ms: stats::median(ttfts),
            slots_busy: views.iter().map(|view| view.slots_busy).sum(),
            slots: views.iter().map(|view| view.slots).sum(),
            queued: views.iter().map(|view| view.queued).sum(),
            lanes_serving: views.iter().filter(|view| state_of(view) == "active").count() as u32,
            lanes_loading: views.iter().filter(|view| matches!(view.state.as_str(), "starting" | "probing" | "new")).count() as u32,
            lanes_failed: views.iter().filter(|view| !view.error.is_empty() || view.state.starts_with("failed") || view.state.starts_with("restarting") || view.state == "crashed").count() as u32,
            power_w: gpus.iter().map(|gpu| gpu.power_w).sum(),
        }
    }

    fn sample_now(&self) -> ServeSample {
        let views = self.views();
        let gpus = lock(&self.state).gpus.clone();
        let totals = self.totals(&views, &gpus);
        ServeSample { at_ms: util::now_ms(), gpus, lanes: views, totals }
    }

    async fn each_minute(&self) {
        // The minute's summary of this computer.
        let sample = self.sample_now();
        let lanes: Vec<Value> = sample
            .lanes
            .iter()
            .map(|lane| json!({"id": lane.settings.id, "recipe": lane.settings.recipe, "tok_s": lane.tok_s, "busy": lane.slots_busy, "slots": lane.slots, "queued": lane.queued, "tokens_hour": lane.tokens_hour, "loaded": lane.loaded}))
            .collect();
        graph::append("serve.minutes", vec![json!({"totals": sample.totals, "lanes": lanes, "gpus": sample.gpus.iter().map(|g| json!({"index": g.index, "util": g.util, "used_bytes": g.used_bytes, "power_w": g.power_w, "temp_c": g.temp_c})).collect::<Vec<_>>()})], 90);
        // The mesh's.
        self.refresh_wants().await;
        let cluster = self.compute_cluster(&sample).await;
        graph::append("serve.cluster.minutes", vec![json!({"totals": cluster.totals, "servers": cluster.servers, "servers_online": cluster.servers_online, "reporting": cluster.reporting, "gpus": cluster.gpus})], 90);
        lock(&self.state).cluster = cluster;
        // The plan follows the hardware and the apps here.
        if minute_hardware_check(util::now_ms()) {
            self.refresh_hardware().await;
        }
        let changed = self.follow_plan("plan").await;
        // A calibration that was dropped midway (its caller went away) leaves no trial behind.
        let trial_over = !self.calibrating.load(std::sync::atomic::Ordering::SeqCst) && !std::mem::take(&mut lock(&self.state).trial).is_empty();
        let needs_apply = changed || trial_over || self.lanes_missing();
        if needs_apply && let Err(error) = self.apply().await {
            tracing::warn!(error = %error.message, "the lanes could not be set");
        }
    }

    /// The frame runs fewer lanes than are kept (it restarted, or a set failed).
    fn lanes_missing(&self) -> bool {
        let state = lock(&self.state);
        state.lanes.iter().any(|lane| !state.live.contains_key(&lane.id))
    }

    /// Every model server of the mesh, added up from its card; this one from its own readings.
    async fn compute_cluster(&self, own: &ServeSample) -> Cluster {
        let Ok(ctx) = self.ctx() else { return Cluster::default() };
        let peers = peers(ctx).await.unwrap_or_default();
        let mut totals = own.totals.clone();
        let mut servers = 1;
        let mut online = 1;
        let mut reporting = u32::from(!own.gpus.is_empty());
        let mut gpu_count = own.gpus.len() as u32;
        let mut weighted = own.totals.gpu_util * own.totals.gpu_bytes as f64;
        // The cluster is the model servers: other apps (on this computer or another) are left out.
        for peer in &peers.peers {
            let Some(card) = &peer.card else { continue };
            if card.software != SOFTWARE || peer.member.local {
                continue;
            }
            let computer = true;
            if computer {
                servers += 1;
                if peer.reachable {
                    online += 1;
                }
            }
            if !peer.reachable {
                continue;
            }
            add_card(&mut totals, card, &mut weighted, &mut reporting, &mut gpu_count, computer);
        }
        totals.gpu_util = if totals.gpu_bytes > 0 { weighted / totals.gpu_bytes as f64 } else { 0.0 };
        Cluster { totals, servers_online: online, servers, reporting, gpus: gpu_count, updated_ms: util::now_ms() }
    }

    fn publish_changed(&self) {
        if let Some(ctx) = self.ctx.get()
            && let Ok(value) = serde_json::to_value(LanesView { lanes: self.views() })
        {
            ctx.publish(topics::CHANGED, value);
        }
    }

    // ---------------------------------------------------------------------------------------
    // Events

    pub fn event(&self, topic: &str, data: Value) {
        match topic {
            "model.progress" => {
                let lane = data.get("lane").and_then(Value::as_str).unwrap_or_default().to_owned();
                if lane.is_empty() {
                    return;
                }
                let Ok(progress) = serde_json::from_value::<Progress>(data) else { return };
                let record = lock(&self.stats).progress(&lane, &progress, util::now_ms());
                if let Some(record) = record {
                    graph::append("model.served", vec![record], 30);
                }
            }
            "frame.lanes.changed" => {
                if let Ok(result) = serde_json::from_value::<LanesResult>(data) {
                    self.frame_states(&result);
                    self.publish_changed();
                }
            }
            _ => {}
        }
    }

    // ---------------------------------------------------------------------------------------
    // Methods

    pub async fn handle(&self, method: &str, params: Value) -> Result<Value, RpcError> {
        if !lock(&self.state).loaded && method != methods::RECIPES_LIST {
            return Err(RpcError::new(codes::UNAVAILABLE, "the model server is starting").retryable());
        }
        match method {
            methods::STATUS_GET => to_value(&self.status()),
            methods::GPUS_LIST => to_value(&GpusResult { gpus: lock(&self.state).gpus.clone() }),
            methods::RECIPES_LIST => {
                let recipes = lock(&self.state).recipes.clone();
                let recipes = if recipes.is_empty() { recipes::load(&self.recipes_dir) } else { recipes };
                to_value(&RecipesResult { recipes })
            }
            methods::LANES_LIST => to_value(&LanesView { lanes: self.views() }),
            methods::LANES_CREATE => {
                let settings: LaneSettings = parse(params.clone())?;
                let by = caller(&params);
                let (mut settings, recipe, fit) = self.settle(settings, "")?;
                if !fit.fits {
                    return Err(invalid(format!("{} does not fit on GPU {}: {}", recipe.name, settings.gpu, fit.note)));
                }
                if lock(&self.state).lanes.iter().any(|lane| lane.id == settings.id) {
                    return Err(RpcError::new(codes::CONFLICT, format!("lane `{}` exists", settings.id)));
                }
                settings.manual = true;
                lock(&self.state).lanes.push(settings.clone());
                self.record_lane(&settings, &recipe, &by, "add");
                self.apply().await?;
                self.view(&settings.id)
            }
            methods::LANES_EDIT => {
                let params_value = params.clone();
                let edit: LaneEditParams = parse(params)?;
                let by = caller(&params_value);
                if !lock(&self.state).lanes.iter().any(|lane| lane.id == edit.id) {
                    return Err(RpcError::not_found(format!("no lane `{}`", edit.id)));
                }
                let mut settings = edit.settings;
                settings.id = edit.id.clone();
                let (mut settings, recipe, fit) = self.settle(settings, &edit.id)?;
                if !fit.fits {
                    return Err(invalid(format!("{} does not fit on GPU {}: {}", recipe.name, settings.gpu, fit.note)));
                }
                settings.manual = true;
                {
                    let mut state = lock(&self.state);
                    for lane in &mut state.lanes {
                        if lane.id == edit.id {
                            *lane = settings.clone();
                        }
                    }
                }
                self.record_lane(&settings, &recipe, &by, "change");
                self.apply().await?;
                self.view(&settings.id)
            }
            methods::LANES_DELETE => {
                let by = caller(&params);
                let LaneIdParams { id } = parse(params)?;
                let removed = {
                    let mut state = lock(&self.state);
                    let index = state.lanes.iter().position(|lane| lane.id == id).ok_or_else(|| RpcError::not_found(format!("no lane `{id}`")))?;
                    state.lanes.remove(index)
                };
                if !removed.manual {
                    self.decline(&removed);
                }
                self.retire_lane(&removed, &by);
                self.apply().await?;
                Ok(json!({}))
            }
            methods::FIT_GET => {
                let params: FitParams = parse(params)?;
                let replaces = params.replaces.clone();
                let mut settings = params.settings;
                if settings.id.is_empty() {
                    settings.id = replaces.clone();
                }
                let recipe = self.recipe(&settings.recipe).ok_or_else(|| invalid(format!("no recipe `{}`", settings.recipe)))?;
                let gpu = self.gpu(settings.gpu).ok_or_else(|| invalid(format!("this computer has no GPU {}", settings.gpu)))?;
                let mut fit = self.fit_of(&settings, &recipe, &gpu, &replaces);
                if !recipe.available {
                    fit.fits = false;
                    fit.note = recipe.reason.clone();
                }
                to_value(&fit)
            }
            methods::PLAN_GET => {
                let output = self.planned();
                let state = lock(&self.state);
                let changes = plan::changes(&state.lanes, &output.lanes);
                to_value(&Plan { lanes: output.lanes, changes, notes: output.notes })
            }
            methods::PLAN_RUN => {
                // Back to the plan: lanes set by hand become the plan's again, removed ones may
                // come back.
                let declined: Vec<String> = {
                    let mut state = lock(&self.state);
                    for lane in &mut state.lanes {
                        lane.manual = false;
                    }
                    std::mem::take(&mut state.declined).into_iter().collect()
                };
                if !declined.is_empty() {
                    graph::retire(declined.iter().map(|key| r("Declined", key.clone())).collect(), "plan reset");
                }
                self.follow_plan(&caller(&params)).await;
                self.apply().await?;
                to_value(&LanesView { lanes: self.views() })
            }
            methods::CALIBRATE_RUN => {
                let CalibrateParams { lane } = parse(params)?;
                to_value(&self.calibrate(&lane).await?)
            }
            methods::CLUSTER_GET => {
                let sample = self.sample_now();
                let cluster = self.compute_cluster(&sample).await;
                lock(&self.state).cluster = cluster.clone();
                to_value(&cluster)
            }
            methods::HISTORY_READ => {
                let params: HistoryParams = parse(params)?;
                to_value(&self.history(params).await?)
            }
            other => Err(RpcError::unknown_method(other)),
        }
    }

    fn view(&self, id: &str) -> Result<Value, RpcError> {
        let view = self.views().into_iter().find(|view| view.settings.id == id).ok_or_else(|| RpcError::not_found(format!("no lane `{id}`")))?;
        to_value(&view)
    }

    fn status(&self) -> ServeStatus {
        let views = self.views();
        let state = lock(&self.state);
        let gpus = state.gpus.clone();
        let engines = state.engines.clone();
        let wants = state.wants.clone();
        let chip = state.hardware.chip.clone();
        drop(state);
        let totals = self.totals(&views, &gpus);
        ServeStatus { node: String::new(), name: chip, platform: util::platform().into(), gpus, lanes: views, engines, wants, totals }
    }

    async fn history(&self, params: HistoryParams) -> Result<History, RpcError> {
        let topic = match params.scope.as_str() {
            "cluster" => "serve.cluster.minutes",
            "node" | "lane" => "serve.minutes",
            other => return Err(invalid(format!("scope `{other}`: cluster, node or lane"))),
        };
        let range = params.range_s.clamp(60, 90 * 86_400);
        let since = util::now_ms().saturating_sub(range * 1000);
        let entries = graph::stream_read(topic, since).await.map_err(|error| RpcError::new(codes::UNAVAILABLE, error).retryable())?;
        // At most about 360 points: minutes are averaged into buckets for long ranges.
        let bucket_ms = ((range * 1000) / 360).max(60_000);
        let mut points: Vec<HistoryPoint> = Vec::new();
        let mut current: Option<(u64, Vec<Totals>)> = None;
        for entry in entries {
            let totals: Option<Totals> = if params.scope == "lane" {
                entry.value.get("lanes").and_then(Value::as_array).and_then(|lanes| lanes.iter().find(|lane| lane.get("id").and_then(Value::as_str) == Some(params.id.as_str()))).map(|lane| Totals {
                    tok_s: lane.get("tok_s").and_then(Value::as_f64).unwrap_or(0.0),
                    slots_busy: lane.get("busy").and_then(Value::as_u64).unwrap_or(0) as u32,
                    slots: lane.get("slots").and_then(Value::as_u64).unwrap_or(0) as u32,
                    queued: lane.get("queued").and_then(Value::as_u64).unwrap_or(0) as u32,
                    tokens_hour: lane.get("tokens_hour").and_then(Value::as_u64).unwrap_or(0),
                    ..Totals::default()
                })
            } else {
                entry.value.get("totals").cloned().and_then(|value| serde_json::from_value(value).ok())
            };
            let Some(totals) = totals else { continue };
            let bucket = entry.at_ms / bucket_ms * bucket_ms;
            match &mut current {
                Some((at, list)) if *at == bucket => list.push(totals),
                _ => {
                    if let Some((at, list)) = current.take() {
                        points.push(HistoryPoint { at_ms: at, totals: average(&list) });
                    }
                    current = Some((bucket, vec![totals]));
                }
            }
        }
        if let Some((at, list)) = current {
            points.push(HistoryPoint { at_ms: at, totals: average(&list) });
        }
        Ok(History { points })
    }

    /// Measure every lane for real and, on a full run, apply what was found. A chat lane: one
    /// call alone (decode, prefill, first token), its total at every slot count worth trying,
    /// then each drafting setting. Then each GPU's lanes busy at once: memory says what can be
    /// loaded together, only this says what the GPU can compute together.
    async fn calibrate(&self, only: &str) -> Result<CalibrationResult, RpcError> {
        let ctx = self.ctx()?.clone();
        let lanes: Vec<LaneView> = self.views().into_iter().filter(|view| only.is_empty() || view.settings.id == only).collect();
        if lanes.is_empty() {
            return Err(RpcError::not_found(if only.is_empty() { "no lane to calibrate".to_owned() } else { format!("no lane `{only}`") }));
        }
        if self.calibrating.swap(true, std::sync::atomic::Ordering::SeqCst) {
            return Err(RpcError::new(codes::UNAVAILABLE, "a calibration is running on this computer").retryable());
        }
        let _running = CalibrationRunning(&self.calibrating);
        let mut results = Vec::new();
        let mut chosen: BTreeMap<String, Trial> = BTreeMap::new();
        for lane in &lanes {
            let id = lane.settings.id.clone();
            let mut calibration = Calibration { lane: id.clone(), recipe: lane.settings.recipe.clone(), at_ms: util::now_ms(), current_slots: lane.slots.max(1), kind: lane.kind.clone(), ..Calibration::default() };
            if lane.kind == kinds::CHAT {
                match calibrate_call(ctx.clone(), id.clone(), 0).await {
                    Ok(answer) => {
                        let usage = &answer["usage"];
                        let number = |name: &str| usage.get(name).and_then(Value::as_f64).unwrap_or(0.0);
                        if number("decode_ms") > 0.0 {
                            calibration.decode_tok_s = number("completion_tokens") / number("decode_ms") * 1000.0;
                        }
                        if number("prefill_ms") > 0.0 {
                            calibration.prefill_tok_s = (number("prompt_tokens") - number("cached_tokens")).max(0.0) / number("prefill_ms") * 1000.0;
                        }
                        calibration.ttft_ms = number("queue_ms") + number("prefill_ms");
                    }
                    Err(error) => calibration.error = error.message,
                }
                if calibration.error.is_empty()
                    && let Some(trial) = self.tune_chat(&ctx, lane, &mut calibration).await
                {
                    chosen.insert(id, trial);
                }
            } else if let Some(trial) = self.measure_other(&ctx, lane, &mut calibration).await {
                chosen.insert(id, trial);
            }
            results.push(calibration);
        }
        self.keep_calibration(only.is_empty(), &lanes, &results, &chosen).await;
        Ok(CalibrationResult { results, gpus: Vec::new() })
    }

    /// Record the results and, on a full run, apply the drafting found: to a lane set by hand
    /// directly, to a planned lane through the plan. Sizes are never changed.
    async fn keep_calibration(&self, full: bool, lanes: &[LaneView], results: &[Calibration], chosen: &BTreeMap<String, Trial>) {
        for (lane, calibration) in lanes.iter().zip(results) {
            let id = calibration.lane.clone();
            let trial = chosen.get(&id).filter(|_| full && calibration.error.is_empty());
            if let Some(trial) = trial {
                if lane.settings.manual {
                    let edited = {
                        let mut state = lock(&self.state);
                        state.lanes.iter_mut().find(|own| own.id == id).map(|own| {
                            trial.on(own);
                            own.clone()
                        })
                    };
                    if let (Some(edited), Some(recipe)) = (edited, self.recipe(&lane.settings.recipe)) {
                        self.record_lane(&edited, &recipe, "calibration", "change");
                    }
                } else {
                    lock(&self.state).calibrated.insert(id.clone(), plan::Tuned { speculative: trial.speculative, draft_max: trial.draft_max });
                }
            }
            let tuned = lock(&self.state).calibrated.get(&id).cloned().unwrap_or_default();
            if let Some(_machine) = graph::machine() {
                let mut batch = Batch::new();
                let unit = batch.unit();
                unit.node("Calibration", id.clone())
                    .t("decode_tok_s", calibration.decode_tok_s)
                    .t("prefill_tok_s", calibration.prefill_tok_s)
                    .t("aggregate_tok_s", calibration.aggregate_tok_s)
                    .t("ttft_ms", calibration.ttft_ms)
                    .t("best_slots", calibration.best_slots as u64)
                    .t("together_tok_s", calibration.together_tok_s)
                    .t("plan_slots", lane.slots as u64)
                    .t("plan_speculative", if tuned.speculative == Some(false) { "off" } else { "" })
                    .t("plan_draft", tuned.draft_max.unwrap_or(0) as u64)
                    .t("plan_gpu", json!(-1))
                    .p("recipe", calibration.recipe.clone())
                    .ps("error", &calibration.error);
                unit.edge(r("Calibration", id.clone()), "OF", r("Lane", id.clone()));
                graph::put(batch);
            }
        }
        if full {
            self.follow_plan("calibration").await;
            if let Err(error) = self.apply().await {
                tracing::warn!(error = %error.message, "the calibrated lanes could not be set");
            }
        }
    }

    /// A lane with no slots or drafting to try: what it gets through.
    async fn measure_other(&self, ctx: &Ctx, lane: &LaneView, calibration: &mut Calibration) -> Option<Trial> {
        let id = lane.settings.id.clone();
        // The first call may load the model; it is not measured.
        if let Err(error) = lane_work(ctx.clone(), id.clone(), lane.kind.clone(), 0).await {
            calibration.error = error.message;
            return None;
        }
        let started = std::time::Instant::now();
        match lane_work(ctx.clone(), id, lane.kind.clone(), 1).await {
            Ok(work) => {
                let tok_s = work / started.elapsed().as_secs_f64().max(f64::EPSILON);
                calibration.sweep.push(SlotLoad { slots: 1, tok_s });
                calibration.aggregate_tok_s = tok_s;
                calibration.best_slots = 1;
                Some(Trial::at(&lane.settings, 1))
            }
            Err(error) => {
                calibration.error = error.message;
                None
            }
        }
    }

    /// Measure the lane with its slots busy, then each drafting setting, and give it its own
    /// settings back. The drafting found.
    async fn tune_chat(&self, ctx: &Ctx, lane: &LaneView, calibration: &mut Calibration) -> Option<Trial> {
        let Some(recipe) = self.recipe(&lane.settings.recipe) else {
            calibration.error = format!("no recipe `{}`", lane.settings.recipe);
            return None;
        };
        // The lane's own slots, all busy at once: what it gets through. The sizes stay the app's.
        let current = calibration.current_slots;
        self.try_slots(ctx, lane, &recipe, current, calibration).await;
        let mut found = None;
        if !calibration.sweep.is_empty() {
            calibration.aggregate_tok_s = calibration.sweep.iter().find(|load| load.slots == current).map_or(0.0, |load| load.tok_s);
            calibration.best_slots = current;
            calibration.ctx = lane.ctx;
            let mut trial = Trial::at(&lane.settings, current);
            let choices = draft_choices(&recipe);
            for (label, speculative, draft_max) in &choices {
                let attempt = Trial { speculative: *speculative, draft_max: *draft_max, ..trial.clone() };
                match self.measure_trial(ctx, &lane.settings.id, &lane.kind, &attempt).await {
                    Ok(tok_s) => calibration.drafts.push(DraftLoad { label: label.clone(), tok_s }),
                    Err(error) => {
                        calibration.sweep_note = format!("drafting {label}: {}", error.message);
                        break;
                    }
                }
            }
            if let Some(best) = best_draft(&calibration.drafts) {
                let (label, speculative, draft_max) = &choices[best];
                calibration.best_draft = label.clone();
                trial.speculative = *speculative;
                trial.draft_max = *draft_max;
            }
            found = Some(trial);
        }
        lock(&self.state).trial.clear();
        if let Err(error) = self.apply().await {
            calibration.error = format!("the lane could not go back to its own settings: {}", error.message);
            return None;
        }
        if found.is_none() {
            calibration.error = std::mem::take(&mut calibration.sweep_note);
        }
        found
    }

    /// The lane's total with `slots` calls at once. False when it could not be measured
    /// (`sweep_note` says why).
    async fn try_slots(&self, ctx: &Ctx, lane: &LaneView, recipe: &Recipe, slots: u32, calibration: &mut Calibration) -> bool {
        let trial = Trial::at(&lane.settings, slots);
        let settings = lane.settings.clone();
        if let Some(gpu) = self.gpu(settings.gpu) {
            let fit = self.fit_of(&settings, recipe, &gpu, &settings.id);
            if !fit.fits {
                calibration.sweep_note = format!("{slots} slots: {}", fit.note);
                return false;
            }
        }
        match self.measure_trial(ctx, &settings.id, &lane.kind, &trial).await {
            Ok(tok_s) => {
                calibration.sweep.push(SlotLoad { slots, tok_s });
                calibration.sweep.sort_by_key(|load| load.slots);
                true
            }
            Err(error) => {
                calibration.sweep_note = format!("{slots} slots: {}", error.message);
                false
            }
        }
    }

    /// Reload the lane with `trial` and measure its total with every slot busy.
    async fn measure_trial(&self, ctx: &Ctx, id: &str, kind: &str, trial: &Trial) -> Result<f64, RpcError> {
        lock(&self.state).trial = BTreeMap::from([(id.to_owned(), trial.clone())]);
        self.apply().await?;
        lane_ready(ctx, id, kind).await?;
        let started = std::time::Instant::now();
        let work = lane_round(ctx, id, kind, trial.slots).await?;
        Ok(work / started.elapsed().as_secs_f64().max(f64::EPSILON))
    }
}

/// Drafting a calibration runs a lane with for a while. The lane's sizes (slots, context) are the
/// app's own and are never changed by a calibration; `slots` is how many calls a round sends.
#[derive(Debug, Clone, PartialEq)]
struct Trial {
    slots: u32,
    speculative: Option<bool>,
    draft_max: Option<u32>,
}

impl Trial {
    fn at(settings: &LaneSettings, slots: u32) -> Self {
        Self { slots: slots.max(1), speculative: settings.speculative, draft_max: settings.draft_max }
    }

    fn on(&self, lane: &mut LaneSettings) {
        lane.speculative = self.speculative;
        lane.draft_max = self.draft_max;
    }
}

/// Marks a calibration as over however it ends.
struct CalibrationRunning<'a>(&'a std::sync::atomic::AtomicBool);

impl Drop for CalibrationRunning<'_> {
    fn drop(&mut self) {
        self.0.store(false, std::sync::atomic::Ordering::SeqCst);
    }
}

/// Wait for a reloaded lane to come up and load its model (one call, not measured).
async fn lane_ready(ctx: &Ctx, id: &str, kind: &str) -> Result<(), RpcError> {
    let deadline = std::time::Instant::now() + Duration::from_secs(600);
    loop {
        match lane_work(ctx.clone(), id.to_owned(), kind.to_owned(), 0).await {
            Ok(_) => return Ok(()),
            Err(error) if error.retryable && std::time::Instant::now() < deadline => tokio::time::sleep(Duration::from_secs(2)).await,
            Err(error) => return Err(error),
        }
    }
}

/// One call per slot at once; the work done.
async fn lane_round(ctx: &Ctx, id: &str, kind: &str, slots: u32) -> Result<f64, RpcError> {
    let mut set = tokio::task::JoinSet::new();
    for n in 1..=slots as usize {
        set.spawn(lane_work(ctx.clone(), id.to_owned(), kind.to_owned(), n));
    }
    let mut work = 0.0;
    while let Some(done) = set.join_next().await {
        work += done.map_err(|error| RpcError::internal(error.to_string()))??;
    }
    Ok(work)
}

const WORK_TEXT: &str = "The river carved its valley over a million years, moving silt from the high plateau to the delta, where the town of Arden grew beside the old stone bridge built by the Marlow family.";

/// One real call on a lane; the work it did (tokens generated or read, words extracted from).
async fn lane_work(ctx: Ctx, id: String, kind: String, n: usize) -> Result<f64, RpcError> {
    match kind.as_str() {
        kinds::EXTRACT => {
            let text = format!("{WORK_TEXT} {WORK_TEXT}");
            let words = text.split_whitespace().count() as f64;
            ctx.call("model.extract", json!({"plugin": id, "relex": {"text": text, "kinds": ["person", "place", "organization", "thing"], "relations": ["located in", "built by"]}})).await?;
            Ok(words)
        }
        _ => Ok(calibrate_call(ctx, id, n).await?["usage"]["completion_tokens"].as_f64().unwrap_or(0.0)),
    }
}

/// One real chat call. `n` varies the prompt, long enough that every call runs to the limit.
async fn calibrate_call(ctx: Ctx, id: String, n: usize) -> Result<Value, RpcError> {
    let params = json!({
        "plugin": id,
        "messages": [{"role": "user", "content": format!("Write {} numbered sentences about how rivers shape valleys. Plain text.", 40 + n)}],
        "max_tokens": 192,
        "sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": 1},
        "thinking": {"enabled": false},
        "priority": "background",
        "cache_key": format!("calibrate-{id}-{n}"),
    });
    ctx.call("model.chat", params).await
}

/// The drafting settings a recipe's engine lets us set, cheapest first: (label, on/off, depth).
fn draft_choices(recipe: &Recipe) -> Vec<(String, Option<bool>, Option<u32>)> {
    if recipe.speculative.is_empty() || recipe.speculative == "none" {
        return Vec::new();
    }
    if !recipe.keys.speculative.is_empty() {
        return vec![("off".to_owned(), Some(false), None), ("on".to_owned(), None, None)];
    }
    if recipe.keys.draft_max.is_empty() {
        return Vec::new();
    }
    let own = recipe.config.get(&recipe.keys.draft_max).and_then(Value::as_u64).unwrap_or(4) as u32;
    let depths: BTreeSet<u32> = [1, 2, 4, 8, own].into_iter().collect();
    depths.into_iter().map(|depth| (depth.to_string(), None, Some(depth))).collect()
}

/// The cheapest drafting setting within 5% of the fastest.
fn best_draft(drafts: &[DraftLoad]) -> Option<usize> {
    let peak = drafts.iter().map(|load| load.tok_s).fold(0.0, f64::max);
    drafts.iter().position(|load| load.tok_s >= peak * 0.95)
}

/// Re-measure the hardware every 10 minutes (a GPU added or taken out, drivers updated).
fn minute_hardware_check(now_ms: u64) -> bool {
    (now_ms / 60_000) % 10 == 0
}

fn caller(params: &Value) -> String {
    let caller = params.get("caller");
    let name = caller.and_then(|c| c.get("name")).and_then(Value::as_str).unwrap_or_default();
    let app = caller.and_then(|c| c.get("app")).and_then(Value::as_str).unwrap_or_default();
    match (name.is_empty(), app.is_empty()) {
        (true, true) => "this computer".into(),
        (false, _) => name.to_owned(),
        (true, false) => app.to_owned(),
    }
}

fn lane_of_row(row: &[Value]) -> Option<LaneSettings> {
    let text = |i: usize| row.get(i).and_then(Value::as_str).unwrap_or_default().to_owned();
    let number = |i: usize| row.get(i).and_then(Value::as_i64);
    let flag = |i: usize, default: bool| row.get(i).and_then(Value::as_bool).unwrap_or(default);
    let optional = |i: usize| number(i).filter(|v| *v >= 0).map(|v| v as u32);
    let id = text(0);
    let recipe = text(1);
    if id.is_empty() || recipe.is_empty() {
        return None;
    }
    let kv = text(7);
    let speculative = match text(8).as_str() {
        "on" => Some(true),
        "off" => Some(false),
        _ => None,
    };
    Some(LaneSettings {
        id,
        recipe,
        gpu: number(2).unwrap_or(0).max(0) as u32,
        slots: number(3).unwrap_or(0).max(0) as u32,
        ctx: number(4).unwrap_or(0).max(0) as u32,
        cache_ram_mib: optional(5),
        checkpoints: optional(6),
        kv_type: (!kv.is_empty()).then_some(kv),
        speculative,
        idle_unload_s: optional(9),
        pinned: flag(10, false),
        lend: flag(11, true),
        tags: row.get(12).and_then(Value::as_array).map(|tags| tags.iter().filter_map(Value::as_str).map(str::to_owned).collect()).unwrap_or_default(),
        manual: flag(13, false),
        draft_max: optional(14),
    })
}

/// Each recipe's smallest context: the most any app that selected the model needs, so one lane
/// serves them all. A window per slot takes its share.
fn apply_needs(recipes: &mut [Recipe], wants: &[Want]) {
    for recipe in recipes {
        let needed = wants.iter().filter(|want| want.recipe == recipe.id).map(|want| want.ctx_needed).max().unwrap_or(0);
        let needed = if recipe.ctx_per_slot { needed / recipe.slots_default.max(1) } else { needed };
        recipe.ctx_min = if recipe.ctx_max > 0 { needed.min(recipe.ctx_max) } else { needed };
    }
}

/// The recipe that serves the model an app's card lists: the same weights (the model id names
/// the repository), else the same family and a shared tag; on a Mac the uzu recipe first.
fn match_recipe<'a>(recipes: &'a [Recipe], model_id: &str, family: &str, tags: &[String]) -> Option<&'a Recipe> {
    let id = model_id.to_lowercase();
    let usable: Vec<&Recipe> = recipes.iter().filter(|recipe| recipe.available).collect();
    let by_weights = usable.iter().find(|recipe| {
        let repo = recipe.repo.to_lowercase();
        let tail = repo.rsplit('/').next().unwrap_or_default().to_owned();
        !id.is_empty() && (id == repo || id == tail || id.contains(&tail) || tail.contains(&id))
    });
    if let Some(recipe) = by_weights {
        return Some(recipe);
    }
    let mut same: Vec<&&Recipe> = usable.iter().filter(|recipe| (family.is_empty() || recipe.family.eq_ignore_ascii_case(family)) && recipe.tags.iter().any(|t| tags.contains(t))).collect();
    same.sort_by_key(|recipe| (recipe.engine != "model-uzu", !recipe.auto));
    same.first().map(|recipe| **recipe)
}

/// Keep in the graph what changed in the apps' selections.
fn record_selections(known: &[Want], wants: &[Want]) {
    let key = |want: &Want| format!("{}|{}", want.app, want.model_id);
    let same = |a: &Want, b: &Want| key(a) == key(b) && a.family == b.family && a.tags == b.tags && a.recipe == b.recipe && a.ctx_needed == b.ctx_needed;
    let gone: Vec<_> = known.iter().filter(|want| !wants.iter().any(|now| key(now) == key(want))).map(|want| r("Selection", key(want))).collect();
    if !gone.is_empty() {
        graph::retire(gone, "no longer selected");
    }
    let Some(machine) = graph::machine() else { return };
    let mut batch = Batch::new();
    for want in wants.iter().filter(|want| !known.iter().any(|before| same(before, want))) {
        let unit = batch.unit();
        unit.node("Selection", key(want)).p("app", want.app.clone()).p("model_id", want.model_id.clone()).p("family", want.family.clone()).p("recipe", want.recipe.clone()).pl("tags", want.tags.clone()).t("ctx_needed", want.ctx_needed as u64);
        unit.edge(r("Selection", key(want)), "ON", machine.clone());
    }
    if !batch.is_empty() {
        graph::put(batch);
    }
}

async fn peers(ctx: &Ctx) -> Option<PeersResult> {
    let answer = tokio::time::timeout(Duration::from_secs(5), ctx.call("mesh.peers", json!({}))).await.ok()?.ok()?;
    serde_json::from_value(answer).ok()
}

async fn engines(ctx: &Ctx) -> Vec<Engine> {
    // The installed plugins of the `model` category (every unit, also those with no lane yet).
    let answer = ctx.call("frame.plugins.list", json!({})).await.unwrap_or(Value::Null);
    let mut out: Vec<Engine> = Vec::new();
    for unit in answer.get("units").and_then(Value::as_array).into_iter().flatten() {
        if unit.get("category").and_then(Value::as_str) != Some("model") {
            continue;
        }
        let id = unit.get("id").and_then(Value::as_str).unwrap_or_default().to_owned();
        if id.is_empty() || out.iter().any(|engine| engine.id == id) {
            continue;
        }
        out.push(Engine { id, version: unit.get("version").and_then(Value::as_str).unwrap_or_default().to_owned(), enabled: unit.get("enabled").and_then(Value::as_bool).unwrap_or(true) });
    }
    if out.is_empty() {
        for plugin in answer.get("plugins").and_then(Value::as_array).into_iter().flatten() {
            if plugin.get("category").and_then(Value::as_str) != Some("model") {
                continue;
            }
            let id = plugin.get("plugin").and_then(Value::as_str).unwrap_or_default().to_owned();
            if !id.is_empty() && !out.iter().any(|engine| engine.id == id) {
                out.push(Engine { id, version: plugin.get("build").and_then(Value::as_str).unwrap_or_default().to_owned(), enabled: plugin.get("state").and_then(Value::as_str) != Some("disabled") });
            }
        }
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

/// A member's card added to the totals. A computer's GPUs count once (an app on this computer
/// shares its GPU, which this computer's readings already hold).
fn add_card(totals: &mut Totals, card: &Card, weighted: &mut f64, reporting: &mut u32, gpu_count: &mut u32, computer: bool) {
    if computer && !card.gpus.is_empty() {
        *reporting += 1;
        *gpu_count += card.gpus.len() as u32;
        for gpu in &card.gpus {
            totals.gpu_bytes += gpu.memory_bytes;
            totals.gpu_used_bytes += gpu.used_bytes;
            *weighted += gpu.util * gpu.memory_bytes as f64;
            totals.power_w += gpu.power_w;
        }
    }
    match &card.serving {
        Some(serving) => {
            totals.tok_s += serving.tok_s;
            totals.prefill_tok_s += serving.prefill_tok_s;
            totals.tokens_hour += serving.tokens_hour;
            totals.tokens_day += serving.tokens_day;
            totals.requests_min += serving.requests_min;
            totals.lanes_serving += serving.lanes_serving;
            totals.lanes_loading += serving.lanes_loading;
            totals.lanes_failed += serving.lanes_failed;
            if serving.ttft_ms > 0.0 {
                totals.ttft_ms = if totals.ttft_ms > 0.0 { (totals.ttft_ms + serving.ttft_ms) / 2.0 } else { serving.ttft_ms };
            }
        }
        // An app that is not a model server: what its models run now.
        None => {
            totals.tok_s += card.models.iter().map(|model| model.slots_busy as f64 * model.tok_s).sum::<f64>();
            totals.tokens_hour += card.models.iter().map(|model| model.tokens_hour).sum::<u64>();
            totals.lanes_serving += card.models.iter().filter(|model| model.loaded).count() as u32;
        }
    }
    totals.slots += card.models.iter().map(|model| model.slots).sum::<u32>();
    totals.slots_busy += card.models.iter().map(|model| model.slots_busy).sum::<u32>();
    totals.queued += card.models.iter().map(|model| model.queued).sum::<u32>();
}

fn average(list: &[Totals]) -> Totals {
    let n = list.len().max(1) as f64;
    let f = |get: fn(&Totals) -> f64| list.iter().map(get).sum::<f64>() / n;
    let u = |get: fn(&Totals) -> u64| (list.iter().map(get).sum::<u64>() as f64 / n) as u64;
    let s = |get: fn(&Totals) -> u32| (list.iter().map(get).sum::<u32>() as f64 / n).round() as u32;
    Totals {
        gpu_util: f(|t| t.gpu_util),
        gpu_used_bytes: u(|t| t.gpu_used_bytes),
        gpu_bytes: u(|t| t.gpu_bytes),
        tok_s: f(|t| t.tok_s),
        prefill_tok_s: f(|t| t.prefill_tok_s),
        tokens_hour: u(|t| t.tokens_hour),
        tokens_day: u(|t| t.tokens_day),
        requests_min: f(|t| t.requests_min),
        ttft_ms: f(|t| t.ttft_ms),
        slots_busy: s(|t| t.slots_busy),
        slots: s(|t| t.slots),
        queued: s(|t| t.queued),
        lanes_serving: s(|t| t.lanes_serving),
        lanes_loading: s(|t| t.lanes_loading),
        lanes_failed: s(|t| t.lanes_failed),
        power_w: f(|t| t.power_w),
    }
}
