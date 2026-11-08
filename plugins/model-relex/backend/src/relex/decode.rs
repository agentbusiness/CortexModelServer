//! From span and pair probabilities to the answer: entities are the spans a
//! kind claims above the threshold, kept greedily by score so none overlap
//! (flat NER); relations are the pairs a relation claims above its threshold
//! whose head and tail both survived as entities, matched by word boundaries.

use super::model::Scored;
use super::words::Word;
use serde::Serialize;

#[derive(Serialize, Clone)]
pub(super) struct Entity {
    pub start: usize,
    pub end: usize,
    pub text: String,
    pub label: String,
    pub score: f32,
    #[serde(skip)]
    word_start: usize,
    #[serde(skip)]
    word_end: usize,
}

#[derive(Serialize)]
pub(super) struct Endpoint {
    pub start: usize,
    pub end: usize,
    pub text: String,
    #[serde(rename = "type")]
    pub kind: String,
}

#[derive(Serialize)]
pub(super) struct Relation {
    pub head: Endpoint,
    pub tail: Endpoint,
    pub relation: String,
    pub score: f32,
}

#[derive(Serialize)]
pub(super) struct Decoded {
    pub entities: Vec<Entity>,
    pub relations: Vec<Relation>,
    pub words: usize,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn decode(
    text: &str,
    words: &[Word],
    kinds: &[String],
    relations: &[String],
    scored: &Scored,
    max_width: usize,
    threshold: f32,
    relation_threshold: f32,
) -> Decoded {
    let word_count = words.len();
    // Candidates in the order torch.where lists them (start, width, kind).
    let mut candidates: Vec<(usize, usize, usize, f32)> = Vec::new();
    for (row, probs) in scored.span_probs.iter().enumerate() {
        let (start, width) = (row / max_width, row % max_width);
        if start + width + 1 > word_count {
            continue;
        }
        for (kind, score) in probs.iter().enumerate() {
            if *score > threshold {
                candidates.push((start, start + width, kind, *score));
            }
        }
    }
    // Greedy: best score first (ties keep candidate order), no overlap.
    candidates.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
    let mut chosen: Vec<(usize, usize, usize, f32)> = Vec::new();
    for candidate in candidates {
        let overlaps = chosen.iter().any(|kept| !(candidate.0 > kept.1 || kept.0 > candidate.1));
        if !overlaps {
            chosen.push(candidate);
        }
    }
    chosen.sort_by_key(|c| c.0);
    let entities: Vec<Entity> = chosen
        .iter()
        .map(|(start, end, kind, score)| {
            let (from, to) = (words[*start].start, words[*end].end);
            Entity { start: from, end: to, text: text[from..to].to_string(), label: kinds[*kind].clone(), score: *score, word_start: *start, word_end: *end }
        })
        .collect();
    let entity_at = |span: (usize, usize)| entities.iter().position(|e| (e.word_start, e.word_end) == span);
    let endpoint = |entity: &Entity| Endpoint { start: entity.start, end: entity.end, text: entity.text.clone(), kind: entity.label.clone() };
    let mut decoded_relations = Vec::new();
    for (pair, probs) in scored.pairs.iter().zip(&scored.pair_scores) {
        let (Some(head), Some(tail)) = (entity_at(scored.selected[pair.0]), entity_at(scored.selected[pair.1])) else { continue };
        for (relation, score) in probs.iter().enumerate() {
            if *score > relation_threshold {
                decoded_relations.push(Relation { head: endpoint(&entities[head]), tail: endpoint(&entities[tail]), relation: relations[relation].clone(), score: *score });
            }
        }
    }
    Decoded { entities, relations: decoded_relations, words: word_count }
}
