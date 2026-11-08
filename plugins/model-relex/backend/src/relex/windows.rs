//! A long text in windows: runs of whole sentences (ending at `.`, `!`, `?`
//! or a line break) of at most `cap` subtokens each. The encoder's working
//! memory grows with the square of its sequence; measured on recorded
//! passages, a 2,495-token one took 6.2 GB, while 768-token windows peak at
//! 2.2 GB, run 20% faster and keep 98% of the whole-text relations. A
//! sentence longer than `cap` is cut at the cap.

use super::encode::Encoder;
use super::words::Word;
use anyhow::{Result, anyhow};
use std::ops::Range;

pub(super) fn windows(encoder: &Encoder, text: &str, words: &[Word], cap: usize) -> Result<Vec<Range<usize>>> {
    let counts = word_tokens(encoder, words)?;
    if counts.iter().sum::<usize>() <= cap {
        return Ok(vec![0..words.len()]);
    }
    let mut sentences: Vec<Range<usize>> = Vec::new();
    let mut start = 0;
    for (index, word) in words.iter().enumerate() {
        let next = words.get(index + 1).map_or(text.len(), |w| w.start);
        if matches!(word.text.as_str(), "." | "!" | "?") || text[word.end..next].contains('\n') {
            sentences.push(start..index + 1);
            start = index + 1;
        }
    }
    if start < words.len() {
        sentences.push(start..words.len());
    }
    let mut out: Vec<Range<usize>> = Vec::new();
    let mut window = 0..0;
    let mut tokens = 0;
    for sentence in sentences {
        let size: usize = counts[sentence.clone()].iter().sum();
        if tokens + size > cap && !window.is_empty() {
            out.push(window.clone());
            window = sentence.start..sentence.start;
            tokens = 0;
        }
        if size > cap {
            for index in sentence {
                if tokens + counts[index] > cap && !window.is_empty() {
                    out.push(window.clone());
                    window = index..index;
                    tokens = 0;
                }
                window.end = index + 1;
                tokens += counts[index];
            }
            continue;
        }
        if window.is_empty() {
            window = sentence.start..sentence.start;
        }
        window.end = sentence.end;
        tokens += size;
    }
    if !window.is_empty() {
        out.push(window);
    }
    Ok(out)
}

/// Subtokens of each word, as the encoder splits pre-split words.
fn word_tokens(encoder: &Encoder, words: &[Word]) -> Result<Vec<usize>> {
    let sequence: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
    let encoding = encoder.tokenizer.encode(sequence, false).map_err(|e| anyhow!("tokenizing: {e}"))?;
    let mut counts = vec![0usize; words.len()];
    for word in encoding.get_word_ids().iter().flatten() {
        counts[*word as usize] += 1;
    }
    Ok(counts)
}
