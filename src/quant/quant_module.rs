use anyhow::Result;
use candle_core::{IndexOp, Tensor};

use candle_nn::{Conv1d, Conv2d, Linear, Module, VarBuilder};

use super::{
    uaq::{UniformAffineQuantizer, UniformAffineQuantizerCfg},
    utils::StraightThrough,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActQuantMode {
    QDiff,
    None,
}

#[derive(Debug, Clone)]
pub struct QuantCfg {
    pub weight_quant_params: UniformAffineQuantizerCfg,
    pub act_quant_params: UniformAffineQuantizerCfg,
    pub act_quant_mode: ActQuantMode,
    pub disable_act_quant: bool,
}

struct LinearCfg {
    pub out_features: usize,
    pub in_features: usize,
    pub bias: bool,
}

#[derive(Debug, Clone)]
pub enum QuantForwardFunction {
    Linear(Linear),
    Conv1d(Conv1d),
    Conv2d(Conv2d),
}

pub struct QuantModule {
    weight_quantizer: UniformAffineQuantizer,
    act_quantizer: Option<UniformAffineQuantizer>,
    weight: Tensor,
    bias: Option<Tensor>,
    activation_function: StraightThrough,
    forward_function: QuantForwardFunction,
    split: usize,
    act_quant_mode: ActQuantMode,
    disable_act_quant: bool,
    use_act_quant: bool,
    use_weight_quant: bool,
}

pub struct QuantModuleCfg {
    pub forward_function: QuantForwardFunction,
    pub weight_quant_params: UniformAffineQuantizerCfg,
    pub act_quant_params: UniformAffineQuantizerCfg,
    pub act_quant_mode: ActQuantMode,
    pub disable_act_quant: bool,
}

impl QuantModule {
    pub fn load(cfg: QuantModuleCfg, vb: &VarBuilder) -> Result<Self> {
        let forward_function = cfg.forward_function.clone();

        let (weight, bias) = match forward_function.clone() {
            QuantForwardFunction::Linear(l) => (l.weight().clone(), l.bias().cloned()),
            QuantForwardFunction::Conv1d(c) => (c.weight().clone(), c.bias().cloned()),
            QuantForwardFunction::Conv2d(c) => (c.weight().clone(), c.bias().cloned()),
        };

        let weight_quantizer = UniformAffineQuantizer::load(cfg.weight_quant_params, &vb.clone())?;
        let act_quantizer = if cfg.act_quant_mode == ActQuantMode::QDiff {
            Some(UniformAffineQuantizer::load(
                cfg.act_quant_params,
                &vb.clone(),
            )?)
        } else {
            None
        };
        let activation_function = StraightThrough::load();
        return Ok(QuantModule {
            forward_function,
            weight,
            bias,
            weight_quantizer,
            act_quantizer,
            activation_function,
            split: 0,
            act_quant_mode: cfg.act_quant_mode,
            disable_act_quant: cfg.disable_act_quant,
            use_act_quant: false,
            use_weight_quant: false,
        });
    }

    pub fn forward(&mut self, input: Tensor, split: usize) -> Result<Tensor> {
        if split != 0 && self.split != 0 {
            assert!(split == self.split);
        } else if split != 0 {
            self.split = split;
        };

        let input = if !self.disable_act_quant && self.use_act_quant {
            if self.split != 0 {
                if self.act_quant_mode == ActQuantMode::QDiff {
                    let mut act_quantizer = self.act_quantizer.clone().unwrap();
                    let input_0 = act_quantizer.forward(input.i((.., ..self.split, .., ..))?)?;
                    let input_1 = act_quantizer.forward(input.i((.., self.split.., .., ..))?)?;
                    Tensor::cat(&[input_0, input_1], 1)?
                } else {
                    input
                }
            } else {
                if self.act_quant_mode == ActQuantMode::QDiff {
                    let mut act_quantizer = self.act_quantizer.clone().unwrap();
                    act_quantizer.forward(input)?
                } else {
                    input
                }
            }
        } else {
            input
        };

        let weight = if self.use_weight_quant {
            if self.split != 0 {
                let mut weight_quantizer = self.weight_quantizer.clone();
                let weight_0 =
                    weight_quantizer.forward(self.weight.i((.., ..self.split, .., ..))?)?;
                let weight_1 =
                    weight_quantizer.forward(self.weight.i((.., self.split.., .., ..))?)?;
                Tensor::cat(&[weight_0, weight_1], 1)?
            } else {
                self.weight.clone()
            }
        } else {
            self.weight.clone()
        };
        let bias = self.bias.clone();

        let output = match self.forward_function.clone() {
            QuantForwardFunction::Linear(_) => Linear::new(weight, bias).forward(&input)?,
            QuantForwardFunction::Conv1d(c) => {
                let config = c.config().clone();
                Conv1d::new(weight, bias, config).forward(&input)?
            }
            QuantForwardFunction::Conv2d(c) => {
                let config = c.config().clone();
                Conv2d::new(weight, bias, config).forward(&input)?
            }
        };

        let output = self.activation_function.forward(&output)?;

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use candle_nn::{conv1d, conv2d, linear, Conv1dConfig, Conv2dConfig};

    use super::*;

    #[test]
    fn test_linear() -> Result<()> {
        use candle_core::DType;

        use crate::quant::uaq::ScaleMethod;

        let device = candle_core::Device::Cpu;
        let dtype = DType::F32;
        let vb = VarBuilder::zeros(dtype, &device);
        let weight_quant_params = UniformAffineQuantizerCfg {
            symmetric: false,
            num_bits: 4,
            leaf_param: false,
            channel_wise: false,
            scale_method: ScaleMethod::Max,
            always_zero: false,
        };
        let act_quant_params = UniformAffineQuantizerCfg {
            symmetric: false,
            num_bits: 4,
            leaf_param: false,
            channel_wise: false,
            scale_method: ScaleMethod::Max,
            always_zero: false,
        };
        let act_quant_mode = ActQuantMode::QDiff;
        let disable_act_quant = false;
        let forward_function = linear(10, 10, vb.clone())?;
        let cfg = QuantModuleCfg {
            forward_function: QuantForwardFunction::Linear(forward_function),
            weight_quant_params,
            act_quant_params,
            act_quant_mode,
            disable_act_quant,
        };
        let mut quant_module = QuantModule::load(cfg, &vb).unwrap();
        let input = Tensor::ones((1, 1, 10, 10), DType::F32, &device)?;
        let output = quant_module.forward(input, 0)?;

        assert_eq!(output.flatten_all()?.to_vec1::<f32>()?, vec![0.0; 100]);

        Ok(())
    }

    #[test]
    fn test_linear_quant() -> Result<()> {
        use candle_core::DType;

        use crate::quant::uaq::ScaleMethod;

        let device = candle_core::Device::Cpu;
        let dtype = DType::F32;
        let vb = VarBuilder::zeros(dtype, &device);
        let weight_quant_params = UniformAffineQuantizerCfg {
            symmetric: false,
            num_bits: 4,
            leaf_param: false,
            channel_wise: false,
            scale_method: ScaleMethod::Max,
            always_zero: false,
        };
        let act_quant_params = UniformAffineQuantizerCfg {
            symmetric: false,
            num_bits: 4,
            leaf_param: false,
            channel_wise: false,
            scale_method: ScaleMethod::Max,
            always_zero: false,
        };
        let act_quant_mode = ActQuantMode::QDiff;
        let disable_act_quant = false;
        let forward_function = linear(10, 10, vb.clone())?;
        let cfg = QuantModuleCfg {
            forward_function: QuantForwardFunction::Linear(forward_function),
            weight_quant_params,
            act_quant_params,
            act_quant_mode,
            disable_act_quant,
        };
        let mut quant_module = QuantModule::load(cfg, &vb).unwrap();
        let input = Tensor::ones((1, 1, 10, 10), DType::F32, &device)?;
        let output = quant_module.forward(input, 0)?;

        assert_eq!(output.flatten_all()?.to_vec1::<f32>()?, vec![0.0; 100]);

        Ok(())
    }

    #[test]
    fn test_conv1d() -> Result<()> {
        use candle_core::DType;

        use crate::quant::uaq::ScaleMethod;

        let device = candle_core::Device::Cpu;
        let dtype = DType::F32;
        let vb = VarBuilder::zeros(dtype, &device);
        let weight_quant_params = UniformAffineQuantizerCfg {
            symmetric: false,
            num_bits: 4,
            leaf_param: false,
            channel_wise: false,
            scale_method: ScaleMethod::Max,
            always_zero: false,
        };
        let act_quant_params = UniformAffineQuantizerCfg {
            symmetric: false,
            num_bits: 4,
            leaf_param: false,
            channel_wise: false,
            scale_method: ScaleMethod::Max,
            always_zero: false,
        };
        let act_quant_mode = ActQuantMode::QDiff;
        let disable_act_quant = false;
        let forward_function = conv1d(
            10,
            10,
            2,
            Conv1dConfig {
                ..Default::default()
            },
            vb.clone(),
        )?;
        let cfg = QuantModuleCfg {
            forward_function: QuantForwardFunction::Conv1d(forward_function),
            weight_quant_params,
            act_quant_params,
            act_quant_mode,
            disable_act_quant,
        };
        let mut quant_module = QuantModule::load(cfg, &vb).unwrap();
        let input = Tensor::ones((1, 10, 10), DType::F32, &device)?;
        let output = quant_module.forward(input, 0)?;

        assert_eq!(output.dims(), &[1, 10, 9]);

        Ok(())
    }

    #[test]
    fn test_conv2d() -> Result<()> {
        use candle_core::DType;

        use crate::quant::uaq::ScaleMethod;

        let device = candle_core::Device::Cpu;
        let dtype = DType::F32;
        let vb = VarBuilder::zeros(dtype, &device);
        let weight_quant_params = UniformAffineQuantizerCfg {
            symmetric: false,
            num_bits: 4,
            leaf_param: false,
            channel_wise: false,
            scale_method: ScaleMethod::Max,
            always_zero: false,
        };
        let act_quant_params = UniformAffineQuantizerCfg {
            symmetric: false,
            num_bits: 4,
            leaf_param: false,
            channel_wise: false,
            scale_method: ScaleMethod::Max,
            always_zero: false,
        };
        let act_quant_mode = ActQuantMode::QDiff;
        let disable_act_quant = false;
        let forward_function = conv2d(
            10,
            10,
            2,
            Conv2dConfig {
                ..Default::default()
            },
            vb.clone(),
        )?;
        let cfg = QuantModuleCfg {
            forward_function: QuantForwardFunction::Conv2d(forward_function),
            weight_quant_params,
            act_quant_params,
            act_quant_mode,
            disable_act_quant,
        };
        let mut quant_module = QuantModule::load(cfg, &vb).unwrap();
        let input = Tensor::ones((1, 10, 10, 10), DType::F32, &device)?;
        let output = quant_module.forward(input, 0)?;

        assert_eq!(output.dims(), &[1, 10, 9, 9]);

        Ok(())
    }
}
