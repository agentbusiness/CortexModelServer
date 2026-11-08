//! The one-layer bidirectional LSTM over the word embeddings: the forward
//! half reads the words in order, the backward half reads them reversed, and
//! their hidden states are joined back to the head width.
//!
//! It runs on the CPU whatever device the rest of the model uses. A
//! recurrence is one small step per word, each waiting for the last: on Metal
//! every step is a dozen tiny kernels, each its own command buffer, and the
//! steps were nearly all of a page's extraction time with the GPU reading
//! busy. On the CPU a step is a few microseconds of arithmetic.

use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::rnn::Direction;
use candle_nn::{LSTM, LSTMConfig, RNN, VarBuilder, lstm};

pub(super) struct BiLstm {
    forward: LSTM,
    backward: LSTM,
}

impl BiLstm {
    /// `vb`: the weights on the CPU.
    pub fn load(vb: VarBuilder, width: usize) -> Result<Self> {
        let hidden = width / 2;
        let forward = lstm(width, hidden, LSTMConfig { direction: Direction::Forward, ..Default::default() }, vb.clone())?;
        let backward = lstm(width, hidden, LSTMConfig { direction: Direction::Backward, ..Default::default() }, vb)?;
        Ok(Self { forward, backward })
    }

    /// `words`: (W, D) -> (W, D).
    pub fn forward(&self, words: &Tensor, device: &Device) -> Result<Tensor> {
        let steps = words.dim(0)?;
        let input = words.to_device(&Device::Cpu)?.unsqueeze(0)?;
        let ahead = self.forward.states_to_tensor(&self.forward.seq(&input)?)?;
        let reversed: Vec<u32> = (0..steps as u32).rev().collect();
        let reversed = Tensor::new(reversed.as_slice(), &Device::Cpu)?;
        let backwards = input.index_select(&reversed, 1)?;
        let behind = self.backward.states_to_tensor(&self.backward.seq(&backwards)?)?.index_select(&reversed, 1)?;
        Ok(Tensor::cat(&[ahead, behind], 2)?.squeeze(0)?.to_device(device)?)
    }
}
