//! The GLiNER-relex lane: one encoder pass over a text gives its entities
//! (spans typed from a list of kinds) and the relations between them (typed
//! from a list of relation names). No generation, no grammar: the model is
//! DeBERTa-v3 plus small heads, run on candle straight from the model
//! folder's `model.safetensors`.
//!
//! Requests on stdin, one JSON line each:
//!   {"id", "relex": {"text", "kinds": [..], "relations": [..],
//!                    "threshold": 0.3, "relation_threshold": 0.5}}
//! answered with
//!   {"id", "text": "<JSON: {entities, relations, words}>", "stats": {...}}
//! The answer's JSON mirrors the Python `GLiNER.inference` result: entities
//! with character offsets, label and score; relations with head and tail
//! entities, the relation name and score.

mod config;
mod deberta;
mod decode;
mod encode;
mod lstm;
mod mlp;
mod model;
mod windows;
mod words;

use crate::{Args, emit};
use anyhow::{Context, Result, anyhow};
use candle_core::Device;
use serde::Deserialize;
use serde_json::{Value, json};
use std::io::BufRead;
use std::time::Instant;

#[derive(Deserialize)]
struct Request {
    id: String,
    relex: Payload,
}

#[derive(Deserialize)]
pub(super) struct Payload {
    pub text: String,
    pub kinds: Vec<String>,
    #[serde(default)]
    pub relations: Vec<String>,
    #[serde(default = "default_threshold")]
    pub threshold: f32,
    #[serde(default)]
    pub relation_threshold: Option<f32>,
}

fn default_threshold() -> f32 {
    0.5
}

pub fn serve(args: Args) -> Result<()> {
    let started = Instant::now();
    let device = pick_device();
    let config = config::load(&args.model)?;
    let tokenizer = tokenizers::Tokenizer::from_file(args.model.join("tokenizer.json"))
        .map_err(|e| anyhow!("tokenizer.json: {e}"))?;
    let encoder = encode::Encoder::new(tokenizer, &config)?;
    let model = model::Relex::load(&args.model, &config, &device)?;
    // The first pass on a device compiles its kernels (8-13 s on Metal); done
    // here so the first request is as fast as every other.
    let warm = Payload { text: "One thing acts on another thing.".to_string(), kinds: vec!["thing".to_string()], relations: vec!["acts on".to_string()], threshold: 0.5, relation_threshold: None };
    extract(&encoder, &model, &config, warm).context("warming the lane")?;
    emit(&json!({
        "ready": true,
        "model": crate::model_name(&args.model),
        "lane": "relex",
        "device": device_name(&device),
        "load_ms": started.elapsed().as_millis() as u64,
        "pid": std::process::id(),
    }));
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let request: Request = match serde_json::from_str(&line) {
            Ok(request) => request,
            Err(e) => {
                emit(&json!({ "id": null, "error": format!("bad relex request: {e}") }));
                continue;
            }
        };
        let id = request.id.clone();
        let answer = extract(&encoder, &model, &config, request.relex);
        // candle keeps every device buffer a pass used for the next pass and
        // returns the unused ones only on a synchronize; without it the lane
        // keeps the largest passage's working memory for good.
        device.synchronize()?;
        match answer {
            Ok((text, stats)) => emit(&json!({ "id": id, "text": text, "stats": stats })),
            Err(e) => emit(&json!({ "id": id, "error": format!("{e:#}") })),
        }
    }
    Ok(())
}

/// Subtokens of text in one encoder pass at most; see `windows`.
const WINDOW_TOKENS: usize = 768;

/// One text through the model and the decoders, window by window, with the
/// step timings. Words keep their offsets in the whole text, so every
/// window's spans are already the text's.
fn extract(encoder: &encode::Encoder, model: &model::Relex, config: &config::RelexConfig, payload: Payload) -> Result<(String, Value)> {
    let total = Instant::now();
    let relation_threshold = payload.relation_threshold.unwrap_or(payload.threshold);
    let words = words::split(&payload.text);
    let windows = windows::windows(encoder, &payload.text, &words, WINDOW_TOKENS)?;
    let mut decoded = decode::Decoded { entities: Vec::new(), relations: Vec::new(), words: words.len() };
    let (mut encode_ms, mut forward_ms, mut tokens, mut selected_spans, mut pairs) = (0u64, 0u64, 0usize, 0usize, 0usize);
    for window in &windows {
        let step = Instant::now();
        let encoded = encoder.encode(&words[window.clone()], &payload.kinds, &payload.relations)?;
        encode_ms += step.elapsed().as_millis() as u64;
        let forward = Instant::now();
        let scored = model.forward(&encoded, config.max_width, payload.threshold)?;
        forward_ms += forward.elapsed().as_millis() as u64;
        let part = decode::decode(&payload.text, &words[window.clone()], &payload.kinds, &payload.relations, &scored, config.max_width, payload.threshold, relation_threshold);
        decoded.entities.extend(part.entities);
        decoded.relations.extend(part.relations);
        tokens += encoded.input_ids.len();
        selected_spans += scored.selected.len();
        pairs += scored.pair_scores.len();
    }
    let text = serde_json::to_string(&decoded).context("relex answer")?;
    let stats = json!({
        "total_ms": total.elapsed().as_millis() as u64,
        "encode_ms": encode_ms,
        "forward_ms": forward_ms,
        "words": words.len(),
        "windows": windows.len(),
        "tokens": tokens,
        "selected_spans": selected_spans,
        "pairs": pairs,
        "generated_tokens": 0,
        "prefilled_tokens": tokens,
    });
    Ok((text, stats))
}

/// Metal or CUDA when this build and this machine have one, else the CPU.
fn pick_device() -> Device {
    if candle_core::utils::metal_is_available() {
        if let Ok(device) = Device::new_metal(0) {
            return device;
        }
    }
    if candle_core::utils::cuda_is_available() {
        if let Ok(device) = Device::new_cuda(0) {
            return device;
        }
    }
    Device::Cpu
}

fn device_name(device: &Device) -> &'static str {
    match device {
        Device::Cpu => "cpu",
        Device::Metal(_) => "metal",
        Device::Cuda(_) => "cuda",
    }
}
