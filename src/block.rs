use anyhow::Result;
use candle_nn::VarBuilder;

use crate::quant::{
    uaq::{UniformAffineQuantizer, UniformAffineQuantizerCfg},
    utils::StraightThrough,
};

pub struct BaseQuantBlock {
    use_weight_quant: bool,
    use_act_quant: bool,
    act_quantizer: UniformAffineQuantizer,
    activation_function: StraightThrough,
    ignore_reconstruction: bool,
}

pub struct BaseQuantBlockCfg {
    act_quant_params: UniformAffineQuantizerCfg,
}

impl BaseQuantBlock {
    pub fn load(cfg: BaseQuantBlockCfg, vb: &VarBuilder) -> Result<Self> {
        let act_quantizer = UniformAffineQuantizer::load(cfg.act_quant_params, &vb.clone())?;
        let activation_function = StraightThrough::load();
        return Ok({
            BaseQuantBlock {
                use_weight_quant: false,
                use_act_quant: false,
                act_quantizer,
                activation_function,
                ignore_reconstruction: false,
            }
        });
    }
}
