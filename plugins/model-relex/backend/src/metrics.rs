//! Throughput over the requests of the last few minutes (`model.metrics`, `model.slots.metrics`).
//! An encoder generates nothing: its rate is tokens read per second (`prefill_tps`); the decode
//! rates stay 0.

use cc_protocol::contracts::model::LiveMetrics;
use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// The window the rates cover.
const WINDOW: Duration = Duration::from_secs(300);

struct Done {
    at: Instant,
    tokens: u64,
    work_ms: f64,
    queue_ms: f64,
}

#[derive(Default)]
pub struct Metrics {
    done: Mutex<VecDeque<Done>>,
}

impl Metrics {
    pub fn record(&self, tokens: u64, work_ms: f64, queue_ms: f64) {
        if let Ok(mut done) = self.done.lock() {
            done.push_back(Done { at: Instant::now(), tokens, work_ms, queue_ms });
            while done.front().is_some_and(|d| d.at.elapsed() > WINDOW) {
                done.pop_front();
            }
        }
    }

    pub fn live(&self) -> LiveMetrics {
        let Ok(mut done) = self.done.lock() else { return LiveMetrics::default() };
        while done.front().is_some_and(|d| d.at.elapsed() > WINDOW) {
            done.pop_front();
        }
        let requests = done.len() as u32;
        let tokens: u64 = done.iter().map(|d| d.tokens).sum();
        let work_ms: f64 = done.iter().map(|d| d.work_ms).sum();
        let queue_ms: f64 = done.iter().map(|d| d.queue_ms).sum();
        LiveMetrics {
            window_s: WINDOW.as_secs_f64(),
            requests,
            prefill_tps: if work_ms > 0.0 { tokens as f64 * 1000.0 / work_ms } else { 0.0 },
            mean_queue_ms: if requests > 0 { queue_ms / requests as f64 } else { 0.0 },
            ..LiveMetrics::default()
        }
    }
}
