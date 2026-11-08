//! What each lane served: from the engines' `model.progress` events (the frame stamps each with
//! its lane). A request's end (`done`) is one served call, recorded in stream `model.served`; its
//! first decoded token gives the time to first token. Counts are kept per minute for a day, so
//! the last hour and day are sums, and are rebuilt from the stream at start.

use cc_protocol::contracts::model::Progress;
use serde_json::{Value, json};
use std::collections::{HashMap, VecDeque};

const DAY_MINUTES: u64 = 24 * 60;

#[derive(Default, Clone, Debug)]
struct Minute {
    minute: u64,
    prompt: u64,
    generated: u64,
    requests: u64,
    ttft_ms: Vec<f64>,
}

#[derive(Default)]
pub struct Stats {
    lanes: HashMap<String, VecDeque<Minute>>,
    /// (lane, request) -> (first seen ms, first decode ms).
    open: HashMap<(String, String), (u64, Option<u64>)>,
}

/// One served call, as the stream keeps it.
pub fn record(lane: &str, progress: &Progress, ttft_ms: Option<f64>) -> Value {
    json!({
        "lane": lane,
        "work": progress.work,
        "priority": format!("{:?}", progress.priority).to_lowercase(),
        "prompt": progress.prompt_tokens,
        "cached": progress.cached_tokens,
        "generated": progress.generated,
        "tok_s": progress.tok_s,
        "ttft_ms": ttft_ms,
        "finish": progress.finish,
    })
}

impl Stats {
    /// A `model.progress` event of `lane` at `now_ms`. Returns the served-call record when the
    /// request ended.
    pub fn progress(&mut self, lane: &str, progress: &Progress, now_ms: u64) -> Option<Value> {
        let key = (lane.to_owned(), progress.request.clone());
        let entry = self.open.entry(key.clone()).or_insert((now_ms, None));
        if progress.phase == "decode" && entry.1.is_none() {
            entry.1 = Some(now_ms);
        }
        if progress.phase != "done" {
            // Requests whose end never came (an engine restarted) do not pile up.
            if self.open.len() > 4096 {
                let cutoff = now_ms.saturating_sub(3_600_000);
                self.open.retain(|_, (start, _)| *start >= cutoff);
            }
            return None;
        }
        let (start, first) = self.open.remove(&key).unwrap_or((now_ms, None));
        let ttft = first.map(|first| first.saturating_sub(start) as f64);
        if progress.work == "generate" {
            self.add(lane, now_ms, progress.prompt_tokens as u64, progress.generated as u64, ttft);
        }
        Some(record(lane, progress, ttft))
    }

    /// A served call read back from the stream (at start).
    pub fn restore(&mut self, at_ms: u64, value: &Value) {
        let lane = value.get("lane").and_then(Value::as_str).unwrap_or_default();
        if lane.is_empty() || value.get("work").and_then(Value::as_str) != Some("generate") {
            return;
        }
        let number = |name: &str| value.get(name).and_then(Value::as_u64).unwrap_or(0);
        self.add(lane, at_ms, number("prompt"), number("generated"), value.get("ttft_ms").and_then(Value::as_f64));
    }

    fn add(&mut self, lane: &str, at_ms: u64, prompt: u64, generated: u64, ttft: Option<f64>) {
        let minute = at_ms / 60_000;
        let minutes = self.lanes.entry(lane.to_owned()).or_default();
        if minutes.back().is_none_or(|last| last.minute != minute) {
            minutes.push_back(Minute { minute, ..Minute::default() });
        }
        if let Some(last) = minutes.back_mut() {
            last.prompt += prompt;
            last.generated += generated;
            last.requests += 1;
            if let Some(ttft) = ttft {
                last.ttft_ms.push(ttft);
            }
        }
        while minutes.front().is_some_and(|first| first.minute + DAY_MINUTES < minute) {
            minutes.pop_front();
        }
    }

    fn since(&self, lane: &str, now_ms: u64, minutes: u64) -> impl Iterator<Item = &Minute> {
        let from = (now_ms / 60_000).saturating_sub(minutes.saturating_sub(1));
        self.lanes.get(lane).into_iter().flat_map(move |list| list.iter().filter(move |m| m.minute >= from))
    }

    /// Tokens (prompt + generated) and requests in the last `minutes`.
    pub fn tokens(&self, lane: &str, now_ms: u64, minutes: u64) -> (u64, u64) {
        self.since(lane, now_ms, minutes).fold((0, 0), |(tokens, requests), m| (tokens + m.prompt + m.generated, requests + m.requests))
    }

    /// Times to first token in the last `minutes`.
    pub fn ttfts(&self, lane: &str, now_ms: u64, minutes: u64) -> Vec<f64> {
        self.since(lane, now_ms, minutes).flat_map(|m| m.ttft_ms.iter().copied()).collect()
    }

    pub fn forget(&mut self, lane: &str) {
        self.lanes.remove(lane);
        self.open.retain(|(open, _), _| open != lane);
    }
}

pub fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    values[values.len() / 2]
}
