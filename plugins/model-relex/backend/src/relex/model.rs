//! GLiNER-relex forward pass on candle: DeBERTa-v3 token states, projected to
//! the head width; the kind and relation marker states as prompts; the first
//! subtoken of each word through the BiLSTM; every span up to `max_width`
//! words represented from its start and end words and scored against each
//! kind; the spans a kind claims above `threshold` paired every way and each
//! pair scored against each relation.

use super::config::RelexConfig;
use super::encode::Encoded;
use super::lstm::BiLstm;
use super::mlp::Mlp;
use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder, linear};
use super::deberta::DebertaV2Model;
use std::path::Path;

pub(super) struct Relex {
    device: Device,
    encoder: DebertaV2Model,
    projection: Linear,
    rnn: BiLstm,
    project_start: Mlp,
    project_end: Mlp,
    out_project: Mlp,
    prompt_rep: Mlp,
    pair_rep: Mlp,
}

/// What the decoders read: per span the probability of each kind, the spans
/// kept for pairing, and per pair the probability of each relation.
pub(super) struct Scored {
    /// (W * max_width) rows of kind probabilities; span (s, k) is row s * max_width + k.
    pub span_probs: Vec<Vec<f32>>,
    /// Kept spans as (start word, end word), in row order.
    pub selected: Vec<(usize, usize)>,
    /// Pairs as (head, tail) indices into `selected`, every ordered pair.
    pub pairs: Vec<(usize, usize)>,
    /// Per pair the probability of each relation.
    pub pair_scores: Vec<Vec<f32>>,
}

impl Relex {
    pub fn load(dir: &Path, config: &RelexConfig, device: &Device) -> Result<Self> {
        let weights = dir.join("model.safetensors");
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], DType::F32, device) }
            .with_context(|| weights.display().to_string())?;
        // The encoder holds 434 of the model's 470 million weights, in 16 bits
        // (half the memory; LAB_RELEX_DTYPE=f32 for 32). Measured against the
        // torch model on the reference pages: the same 148 entities and 52
        // relations, scores within 0.004. The heads stay in 32 bits.
        let encoder_dtype = match std::env::var("LAB_RELEX_DTYPE").as_deref() {
            Ok("f32") => DType::F32,
            _ => DType::F16,
        };
        let encoder_vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], encoder_dtype, device) }
            .with_context(|| weights.display().to_string())?;
        let width = config.hidden_size;
        let encoder = DebertaV2Model::load(encoder_vb.pp("token_rep_layer.bert_layer.model"), &config.encoder_config)
            .context("loading the encoder")?;
        let projection = linear(config.encoder_config.hidden_size, width, vb.pp("token_rep_layer.projection"))?;
        // The BiLSTM's weights on the CPU, where it runs (`lstm`).
        let cpu_vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], DType::F32, &Device::Cpu) }
            .with_context(|| weights.display().to_string())?;
        let rnn = BiLstm::load(cpu_vb.pp("rnn.lstm"), width)?;
        let spans = vb.pp("span_rep_layer.span_rep_layer");
        Ok(Self {
            device: device.clone(),
            encoder,
            projection,
            rnn,
            project_start: Mlp::load(spans.pp("project_start"), width, width * 4, width)?,
            project_end: Mlp::load(spans.pp("project_end"), width, width * 4, width)?,
            out_project: Mlp::load(spans.pp("out_project"), width * 2, width * 4, width)?,
            prompt_rep: Mlp::load(vb.pp("prompt_rep_layer"), width, width * 4, width)?,
            pair_rep: Mlp::load(vb.pp("pair_rep_layer"), width * 2, width * 4, width)?,
        })
    }

    pub fn forward(&self, encoded: &Encoded, max_width: usize, threshold: f32) -> Result<Scored> {
        let word_count = encoded.word_positions.len();
        if word_count == 0 {
            return Ok(Scored { span_probs: Vec::new(), selected: Vec::new(), pairs: Vec::new(), pair_scores: Vec::new() });
        }
        let input_ids = Tensor::new(encoded.input_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let tokens = self.encoder.forward(&input_ids, None, None).context("encoder")?.to_dtype(DType::F32)?;
        let tokens = self.projection.forward(&tokens)?.squeeze(0)?;
        let gather = |positions: &[u32]| -> Result<Tensor> {
            Ok(tokens.index_select(&Tensor::new(positions, &self.device)?, 0)?)
        };
        let kind_raw = gather(&encoded.kind_positions)?;
        let kind_prompts = self.prompt_rep.forward(&kind_raw)?;
        // No relation names: no prompts to gather (an empty index is refused on Metal).
        let relation_prompts = if encoded.relation_positions.is_empty() { None } else { Some(gather(&encoded.relation_positions)?) };
        let first_subtokens = gather(&encoded.word_positions)?;
        let words = self.rnn.forward(&first_subtokens, &self.device)?;
        if std::env::var_os("LAB_RELEX_DEBUG").is_some() {
            // The same statistics experiments/dedicated/relex_probe.py prints
            // from the torch model, to place a divergence.
            describe("tokens(projected)", &tokens)?;
            describe("token[1] (<<ENT>>)", &tokens.get(1)?)?;
            describe("ent prompts raw", &kind_raw)?;
            if let Some(prompts) = &relation_prompts {
                describe("rel prompts raw", prompts)?;
            }
            describe("words (first subtoken)", &first_subtokens)?;
            describe("lstm out", &words)?;
            describe("lstm out word0", &words.get(0)?)?;
            describe("kind prompts (mlp)", &kind_prompts)?;
            describe("start proj", &self.project_start.forward(&words)?)?;
        }

        // Every (start, width) span; a span past the last word is masked and
        // represented from word 0, as the Python side does, and never kept.
        let mut starts = Vec::with_capacity(word_count * max_width);
        let mut ends = Vec::with_capacity(word_count * max_width);
        let mut valid = Vec::with_capacity(word_count * max_width);
        for s in 0..word_count {
            for k in 0..max_width {
                let inside = s + k < word_count;
                starts.push(if inside { s as u32 } else { 0 });
                ends.push(if inside { (s + k) as u32 } else { 0 });
                valid.push(inside);
            }
        }
        let start_rep = self.project_start.forward(&words)?.index_select(&Tensor::new(starts.as_slice(), &self.device)?, 0)?;
        let end_rep = self.project_end.forward(&words)?.index_select(&Tensor::new(ends.as_slice(), &self.device)?, 0)?;
        let span_rep = self.out_project.forward(&Tensor::cat(&[start_rep, end_rep], 1)?.relu()?)?;
        let span_probs: Vec<Vec<f32>> = candle_nn::ops::sigmoid(&span_rep.matmul(&kind_prompts.t()?)?)?.to_vec2()?;

        let mut keep = Vec::new();
        let mut selected = Vec::new();
        for (row, probs) in span_probs.iter().enumerate() {
            if valid[row] && probs.iter().cloned().fold(f32::MIN, f32::max) > threshold {
                keep.push(row as u32);
                selected.push((starts[row] as usize, ends[row] as usize));
            }
        }
        let mut pairs = Vec::new();
        for head in 0..selected.len() {
            for tail in 0..selected.len() {
                if head != tail {
                    pairs.push((head, tail));
                }
            }
        }
        let pair_scores = if pairs.is_empty() || relation_prompts.is_none() {
            Vec::new()
        } else {
            let relation_prompts = relation_prompts.expect("checked above");
            let kept = span_rep.index_select(&Tensor::new(keep.as_slice(), &self.device)?, 0)?;
            let heads: Vec<u32> = pairs.iter().map(|(h, _)| *h as u32).collect();
            let tails: Vec<u32> = pairs.iter().map(|(_, t)| *t as u32).collect();
            let head_rep = kept.index_select(&Tensor::new(heads.as_slice(), &self.device)?, 0)?;
            let tail_rep = kept.index_select(&Tensor::new(tails.as_slice(), &self.device)?, 0)?;
            let pair_rep = self.pair_rep.forward(&Tensor::cat(&[head_rep, tail_rep], 1)?)?;
            candle_nn::ops::sigmoid(&pair_rep.matmul(&relation_prompts.t()?)?)?.to_vec2()?
        };
        Ok(Scored { span_probs, selected, pairs, pair_scores })
    }
}

/// Mean, standard deviation and first values of a tensor, on stderr.
fn describe(name: &str, tensor: &Tensor) -> Result<()> {
    let flat: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
    let n = flat.len() as f32;
    let mean = flat.iter().sum::<f32>() / n;
    let var = flat.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / (n - 1.0).max(1.0);
    eprintln!("{name:28} shape {:?} mean {mean:.6} std {:.6} first {:?}", tensor.dims(), var.sqrt(), &flat[..flat.len().min(4)]);
    Ok(())
}
