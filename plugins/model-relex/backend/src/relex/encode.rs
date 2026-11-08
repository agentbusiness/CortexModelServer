//! The model's input: the prompt words (`<<ENT>> kind ... <<SEP>> <<REL>>
//! relation ... <<SEP>>`) followed by the text's words, run through the
//! encoder's tokenizer as pre-split words; then the positions the heads read:
//! every marker token, and the first subtoken of each text word.

use super::config::RelexConfig;
use super::words::Word;
use anyhow::{Result, anyhow, ensure};
use tokenizers::Tokenizer;

pub(super) struct Encoded {
    pub input_ids: Vec<u32>,
    /// Token positions of the `<<ENT>>` markers, one per kind, in order.
    pub kind_positions: Vec<u32>,
    /// Token positions of the `<<REL>>` markers, one per relation, in order.
    pub relation_positions: Vec<u32>,
    /// Token position of the first subtoken of each text word, in word order.
    pub word_positions: Vec<u32>,
}

pub(super) struct Encoder {
    pub(super) tokenizer: Tokenizer,
    ent_token: String,
    rel_token: String,
    sep_token: String,
    class_token_index: u32,
    rel_token_index: u32,
}

impl Encoder {
    pub fn new(tokenizer: Tokenizer, config: &RelexConfig) -> Result<Self> {
        for (token, id) in [(&config.ent_token, config.class_token_index), (&config.rel_token, config.rel_token_index)] {
            let found = tokenizer.token_to_id(token).ok_or_else(|| anyhow!("tokenizer has no {token}"))?;
            ensure!(found == id, "{token} is id {found} in the tokenizer, {id} in gliner_config.json");
        }
        Ok(Self {
            tokenizer,
            ent_token: config.ent_token.clone(),
            rel_token: config.rel_token.clone(),
            sep_token: config.sep_token.clone(),
            class_token_index: config.class_token_index,
            rel_token_index: config.rel_token_index,
        })
    }

    pub fn encode(&self, words: &[Word], kinds: &[String], relations: &[String]) -> Result<Encoded> {
        let mut prompt: Vec<String> = Vec::new();
        for kind in kinds {
            prompt.push(self.ent_token.clone());
            if !kind.is_empty() {
                prompt.push(kind.clone());
            }
        }
        prompt.push(self.sep_token.clone());
        for relation in relations {
            prompt.push(self.rel_token.clone());
            prompt.push(relation.clone());
        }
        prompt.push(self.sep_token.clone());
        let prompt_words = prompt.len();
        let mut sequence = prompt;
        sequence.extend(words.iter().map(|w| w.text.clone()));
        let encoding = self.tokenizer.encode(sequence, true).map_err(|e| anyhow!("tokenizing: {e}"))?;
        let input_ids = encoding.get_ids().to_vec();
        let mut kind_positions = Vec::new();
        let mut relation_positions = Vec::new();
        let mut word_positions = Vec::new();
        // The first subtoken of every word, counted in word order; the prompt's
        // words are skipped, the text's words are the heads' word sequence.
        let mut previous: Option<u32> = None;
        let mut seen_words = 0usize;
        for (position, word_id) in encoding.get_word_ids().iter().enumerate() {
            let id = input_ids[position];
            if id == self.class_token_index {
                kind_positions.push(position as u32);
            } else if id == self.rel_token_index {
                relation_positions.push(position as u32);
            }
            if let Some(word) = word_id {
                if previous != Some(*word) {
                    seen_words += 1;
                    if seen_words > prompt_words {
                        word_positions.push(position as u32);
                    }
                }
            }
            previous = *word_id;
        }
        ensure!(kind_positions.len() == kinds.len(), "{} kind markers for {} kinds", kind_positions.len(), kinds.len());
        ensure!(relation_positions.len() == relations.len(), "{} relation markers for {} relations", relation_positions.len(), relations.len());
        ensure!(word_positions.len() == words.len(), "{} word tokens for {} words", word_positions.len(), words.len());
        Ok(Encoded { input_ids, kind_positions, relation_positions, word_positions })
    }
}
