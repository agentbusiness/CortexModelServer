//! GLiNER's projection block: Linear -> ReLU -> (Dropout) -> Linear, stored
//! as a `Sequential` whose linear layers are entries 0 and 3.

use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{Linear, VarBuilder, linear};

pub(super) struct Mlp {
    first: Linear,
    last: Linear,
}

impl Mlp {
    pub fn load(vb: VarBuilder, in_dim: usize, hidden: usize, out_dim: usize) -> Result<Self> {
        Ok(Self { first: linear(in_dim, hidden, vb.pp("0"))?, last: linear(hidden, out_dim, vb.pp("3"))? })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.last.forward(&self.first.forward(x)?.relu()?)?)
    }
}
