//! GLiNER's whitespace word splitter: a word is a run of word characters,
//! joined by `-` or `_` when the run continues, or any single other
//! non-space character. Each word keeps its character offsets so spans map
//! back onto the text.

use regex::Regex;
use std::sync::OnceLock;

pub(super) struct Word {
    pub text: String,
    pub start: usize,
    pub end: usize,
}

pub(super) fn split(text: &str) -> Vec<Word> {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    let pattern = PATTERN.get_or_init(|| Regex::new(r"\w+(?:[-_]\w+)*|\S").expect("word pattern"));
    pattern
        .find_iter(text)
        .map(|m| Word { text: m.as_str().to_string(), start: m.start(), end: m.end() })
        .collect()
}
