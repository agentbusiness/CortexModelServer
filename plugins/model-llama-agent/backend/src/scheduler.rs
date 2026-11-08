// CortexModelServer: from CortexAgent (crates/model/src/llama_server.rs `SlotScheduler`,
// `SchedulerState`, `ScheduledSlot`, `SavedSlot`, `choose_slot`, `saved_beyond_limit`,
// `slot_file_stem`, `conversation_key`, `generation_reserve`, `save_slot`, `spawn_slot_spill`,
// `prepare_slot_dir` and `LlamaServer::lease`), made async (tokio) for this plugin. The rules are
// CortexAgent's (from CortexCode's llama.cpp scheduler):
//
// - every request is pinned to a slot this scheduler chose (`id_slot`), which is what makes the
//   vendored patches in `get_available_slot` / `try_clear_idle_slots` reachable;
// - admission by priority class, then arrival; a request waits for a free slot and for room in the
//   shared pool (`--kv-unified`); nothing running is preempted;
// - affinity: a conversation returns to its own slot; a new one takes an empty slot, else the least
//   recently used one (llama.cpp moves the displaced one to its RAM prompt cache);
// - disk (`--slot-save-path`): a background conversation idle 30 s is saved and its slot erased; a
//   foreground conversation is saved when another is given its slot; either is restored into the
//   slot its next request runs on. Files are bounded to 32 GiB, oldest first.
//
// Not carried over: CortexAgent's `primed` history and `block_only` prewarmed host block (its own
// prompt layout, which model@1 callers do not share).

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

const SLOT_IDLE_SPILL: Duration = Duration::from_secs(30);
const SLOT_SPILL_LIMIT_BYTES: u64 = 32 * 1024 * 1024 * 1024;
const GENERATION_RESERVE_TOKENS: u64 = 8192;
pub const SLOT_FILE_SUFFIX: &str = ".slot";
const SLOT_ACTION_TIMEOUT: Duration = Duration::from_secs(300);

/// Who waits: a lower class is served first. CortexAgent has Foreground (0) and Background (2);
/// CortexModelServer patch: model@1's `agent` priority is class 1, between them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SlotLane {
    #[default]
    Foreground,
    Agent,
    Background,
}

impl SlotLane {
    const fn class(self) -> u8 {
        match self {
            Self::Foreground => 0,
            Self::Agent => 1,
            Self::Background => 2,
        }
    }

    pub fn of(priority: cc_protocol::contracts::model::Priority) -> Self {
        use cc_protocol::contracts::model::Priority;
        match priority {
            Priority::Foreground => Self::Foreground,
            Priority::Agent => Self::Agent,
            Priority::Background => Self::Background,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SavedSlot {
    pub file: String,
    pub bytes: u64,
    order: u64,
}

#[derive(Clone, Debug, Default)]
struct ScheduledSlot {
    key: Option<String>,
    busy: bool,
    lane: SlotLane,
    last_used: u64,
    idle_since: Option<Instant>,
    reserved: u64,
    filled: bool,
    held: bool,
}

#[derive(Default)]
struct State {
    slots: Vec<ScheduledSlot>,
    waiters: BTreeSet<(u8, u64)>,
    arrivals: u64,
    clock: u64,
    pool: u64,
    saved: HashMap<String, SavedSlot>,
    saves: u64,
}

impl State {
    fn fits(&self, tokens: u64) -> bool {
        self.pool == 0 || self.slots.iter().filter(|slot| slot.busy).map(|slot| slot.reserved).sum::<u64>().saturating_add(tokens) <= self.pool
    }
}

pub struct Scheduler {
    state: Mutex<State>,
    changed: tokio::sync::Notify,
    dir: PathBuf,
}

/// The right to run one request on one slot; dropping it frees the slot.
pub struct Lease {
    scheduler: Arc<Scheduler>,
    pub slot: u16,
    pub restore: Option<SavedSlot>,
    pub displace: Option<String>,
}

impl Drop for Lease {
    fn drop(&mut self) {
        self.scheduler.release(self.slot);
    }
}

impl Scheduler {
    /// `pool`: cells of the unified KV pool; `dir`: the `--slot-save-path` folder.
    pub fn new(count: u16, pool: u64, dir: PathBuf) -> Arc<Self> {
        let slots = vec![ScheduledSlot::default(); usize::from(count.max(1))];
        Arc::new(Self { state: Mutex::new(State { slots, pool, ..State::default() }), changed: tokio::sync::Notify::new(), dir })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Waits for a slot for conversation `key`. Only `cancel` ends the wait (`None` then).
    pub async fn acquire(self: &Arc<Self>, key: &str, lane: SlotLane, tokens: u64, cancel: &CancellationToken) -> Option<Lease> {
        let ticket = {
            let mut state = self.lock();
            state.arrivals += 1;
            let ticket = (lane.class(), state.arrivals);
            state.waiters.insert(ticket);
            ticket
        };
        loop {
            let notified = self.changed.notified();
            {
                let mut state = self.lock();
                if cancel.is_cancelled() {
                    state.waiters.remove(&ticket);
                    drop(state);
                    self.changed.notify_waiters();
                    return None;
                }
                let tokens = if state.pool > 0 { tokens.min(state.pool) } else { tokens };
                if state.waiters.first() == Some(&ticket)
                    && state.fits(tokens)
                    && let Some(index) = choose_slot(&state, key)
                {
                    state.waiters.remove(&ticket);
                    let lease = self.grant(&mut state, index, key, lane, tokens);
                    drop(state);
                    self.changed.notify_waiters();
                    return Some(lease);
                }
            }
            tokio::select! {
                _ = notified => {}
                _ = tokio::time::sleep(Duration::from_millis(20)) => {}
                _ = cancel.cancelled() => {}
            }
        }
    }

    fn grant(self: &Arc<Self>, state: &mut State, index: usize, key: &str, lane: SlotLane, tokens: u64) -> Lease {
        let own = state.slots[index].key.as_deref() == Some(key);
        let restore = if own { None } else { state.saved.remove(key) };
        let slot = &mut state.slots[index];
        let mut displace = None;
        if !own && slot.filled && slot.lane == SlotLane::Foreground && let Some(previous) = slot.key.clone() {
            displace = Some(previous);
        }
        slot.key = Some(key.to_owned());
        slot.busy = true;
        slot.lane = lane;
        slot.reserved = tokens;
        slot.idle_since = None;
        for (other_index, other) in state.slots.iter_mut().enumerate() {
            if other_index != index && !other.busy && other.key.as_deref() == Some(key) {
                other.key = None;
            }
        }
        Lease { scheduler: self.clone(), slot: u16::try_from(index).unwrap_or(u16::MAX), restore, displace }
    }

    fn release(&self, index: u16) {
        let mut state = self.lock();
        state.clock += 1;
        let clock = state.clock;
        if let Some(slot) = state.slots.get_mut(usize::from(index)) {
            slot.busy = false;
            slot.reserved = 0;
            slot.last_used = clock;
            slot.idle_since = Some(Instant::now());
            slot.filled = true;
        }
        drop(state);
        self.changed.notify_waiters();
    }

    fn take_idle(&self, now: Instant, idle: Duration) -> Vec<(u16, String)> {
        let mut state = self.lock();
        let mut due = Vec::new();
        for (index, slot) in state.slots.iter_mut().enumerate() {
            let idle_enough = slot.idle_since.is_some_and(|since| now.saturating_duration_since(since) >= idle);
            if !slot.busy && !slot.held && slot.filled && slot.lane == SlotLane::Background && idle_enough
                && let Some(key) = slot.key.clone()
            {
                slot.held = true;
                due.push((u16::try_from(index).unwrap_or(u16::MAX), key));
            }
        }
        due
    }

    fn finish_spill(&self, index: u16, key: &str, saved: Option<SavedSlot>) -> Vec<String> {
        let mut state = self.lock();
        let Some(slot) = state.slots.get_mut(usize::from(index)) else { return saved.map(|saved| vec![saved.file]).unwrap_or_default() };
        slot.held = false;
        let Some(saved) = saved else {
            slot.idle_since = None;
            drop(state);
            self.changed.notify_waiters();
            return Vec::new();
        };
        slot.filled = false;
        if slot.key.as_deref() == Some(key) {
            slot.key = None;
        }
        let stale = Self::record(&mut state, key, saved);
        drop(state);
        self.changed.notify_waiters();
        stale
    }

    pub fn record_saved(&self, key: &str, saved: SavedSlot) -> Vec<String> {
        Self::record(&mut self.lock(), key, saved)
    }

    fn record(state: &mut State, key: &str, mut saved: SavedSlot) -> Vec<String> {
        if state.slots.iter().any(|slot| slot.key.as_deref() == Some(key)) {
            return vec![saved.file];
        }
        state.saves += 1;
        saved.order = state.saves;
        let mut delete = Vec::new();
        if let Some(previous) = state.saved.insert(key.to_owned(), saved) {
            delete.push(previous.file);
        }
        for key in saved_beyond_limit(&state.saved, SLOT_SPILL_LIMIT_BYTES) {
            if let Some(record) = state.saved.remove(&key) {
                tracing::info!(bytes = record.bytes, "deleted the oldest saved slot (disk bound)");
                delete.push(record.file);
            }
        }
        delete
    }

    /// Conversations resident per slot (for `model.slots`).
    pub fn keys(&self) -> Vec<(Option<String>, SlotLane)> {
        self.lock().slots.iter().map(|slot| (slot.key.clone(), slot.lane)).collect()
    }

    pub fn queued(&self) -> u32 {
        self.lock().waiters.len() as u32
    }

    pub fn saved_keys(&self) -> Vec<String> {
        self.lock().saved.keys().cloned().collect()
    }

    pub fn remove_files(&self, files: &[String]) {
        for file in files {
            let _ = std::fs::remove_file(self.dir.join(file));
        }
    }
}

fn choose_slot(state: &State, key: &str) -> Option<usize> {
    let free = |slot: &ScheduledSlot| !slot.busy && !slot.held;
    if let Some(index) = state.slots.iter().position(|slot| slot.key.as_deref() == Some(key) && free(slot)) {
        return Some(index);
    }
    let least_recent = |accept: &dyn Fn(&ScheduledSlot) -> bool| {
        state.slots.iter().enumerate().filter(|(_, slot)| free(slot) && accept(slot)).min_by_key(|(_, slot)| slot.last_used).map(|(index, _)| index)
    };
    least_recent(&|slot: &ScheduledSlot| slot.key.is_none()).or_else(|| least_recent(&|_: &ScheduledSlot| true))
}

fn saved_beyond_limit(saved: &HashMap<String, SavedSlot>, limit: u64) -> Vec<String> {
    let mut records: Vec<(&String, &SavedSlot)> = saved.iter().collect();
    records.sort_by_key(|(_, record)| record.order);
    let mut total: u64 = records.iter().map(|(_, record)| record.bytes).sum();
    let mut delete = Vec::new();
    for (key, record) in records {
        if total <= limit {
            break;
        }
        total = total.saturating_sub(record.bytes);
        delete.push(key.clone());
    }
    delete
}

pub fn fnv_bytes(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn slot_file_stem(key: &str) -> String {
    let readable: String = key.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_').take(48).collect();
    format!("{readable}-{:016x}", fnv_bytes(key.as_bytes()))
}

/// CortexAgent's `conversation_key`: from the first non-system message, else all messages.
/// CortexModelServer patch: a model@1 call's `cache_key` is used first when it has one.
pub fn conversation_key(cache_key: Option<&str>, wire: &serde_json::Value) -> String {
    if let Some(key) = cache_key.filter(|key| !key.is_empty()) {
        return key.to_owned();
    }
    let first = wire["messages"]
        .as_array()
        .and_then(|messages| messages.iter().find(|message| message["role"] != "system" && message["role"] != "developer"))
        .map(serde_json::Value::to_string);
    match first {
        Some(text) => format!("conversation:{:016x}", fnv_bytes(text.as_bytes())),
        None => format!("prompt:{:016x}", fnv_bytes(wire["messages"].to_string().as_bytes())),
    }
}

pub fn generation_reserve(max_tokens: Option<u32>) -> u64 {
    match max_tokens {
        None | Some(0) => GENERATION_RESERVE_TOKENS,
        Some(max) => u64::from(max),
    }
}

/// Create the slot folder and remove the slot files an earlier server left.
pub fn prepare_slot_dir(dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("cannot create the slot directory {}: {e}", dir.display()))?;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && path.file_name().and_then(|name| name.to_str()).is_some_and(|name| name.ends_with(SLOT_FILE_SUFFIX)) {
                let _ = std::fs::remove_file(path);
            }
        }
    }
    Ok(())
}

/// One `/slots/{id}?action=...` call.
pub async fn slot_action(http: &reqwest::Client, base: &str, slot: u16, action: &str, body: &serde_json::Value) -> Result<serde_json::Value, String> {
    let path = format!("/slots/{slot}?action={action}");
    let response = http.post(format!("{base}{path}")).json(body).timeout(SLOT_ACTION_TIMEOUT).send().await.map_err(|e| format!("{path} failed: {e}"))?;
    let status = response.status();
    let text = response.text().await.map_err(|e| format!("reading {path} failed: {e}"))?;
    if !status.is_success() {
        return Err(format!("{path} returned status {status}: {}", text.chars().take(512).collect::<String>()));
    }
    serde_json::from_str(&text).map_err(|e| format!("{path} returned invalid JSON: {e}"))
}

/// Saves slot `slot`'s conversation `key` to disk, then erases the slot when `erase`.
pub async fn save_slot(http: &reqwest::Client, base: &str, dir: &Path, slot: u16, key: &str, erase: bool) -> Option<SavedSlot> {
    let unique = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |elapsed| elapsed.as_nanos());
    let file = format!("{}-{unique}{SLOT_FILE_SUFFIX}", slot_file_stem(key));
    let started = Instant::now();
    let saved = match slot_action(http, base, slot, "save", &serde_json::json!({"filename": file})).await {
        Ok(saved) => saved,
        Err(error) => {
            tracing::warn!(slot, %error, "saving a slot to disk failed; its conversation is processed again later");
            let _ = std::fs::remove_file(dir.join(&file));
            return None;
        }
    };
    let bytes = saved["n_written"].as_u64().unwrap_or(0);
    tracing::info!(slot, n_tokens = saved["n_saved"].as_u64().unwrap_or(0), bytes, save_ms = started.elapsed().as_secs_f64() * 1000.0, erase, "saved a slot's conversation to disk");
    if erase && let Err(error) = slot_action(http, base, slot, "erase", &serde_json::json!({})).await {
        tracing::warn!(slot, %error, "erasing a saved slot failed; it stays resident until reused");
    }
    Some(SavedSlot { file, bytes, order: 0 })
}

/// Every second, saves each background conversation idle for 30 s and erases its slot. Ends when
/// the scheduler is dropped.
pub fn spawn_slot_spill(scheduler: Weak<Scheduler>, http: reqwest::Client, base: String) {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(1)).await;
            let Some(scheduler) = scheduler.upgrade() else { return };
            for (slot, key) in scheduler.take_idle(Instant::now(), SLOT_IDLE_SPILL) {
                let saved = save_slot(&http, &base, scheduler.dir(), slot, &key, true).await;
                let stale = scheduler.finish_spill(slot, &key, saved);
                scheduler.remove_files(&stale);
            }
        }
    });
}
