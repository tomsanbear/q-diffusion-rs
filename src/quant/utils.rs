use candle_core::Tensor;
use candle_nn::Module;

pub struct StraightThrough {}

impl StraightThrough {
    pub fn load() -> Self {
        StraightThrough {}
    }
}

impl Module for StraightThrough {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        Ok(x.clone())
    }
}
