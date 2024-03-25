use core::panic;

use anyhow::Result;
use candle_core::{IndexOp, Tensor, D};
use candle_nn::{linear, Init, Linear, Module, VarBuilder};
use tracing::warn;

fn round_ste(x: &Tensor) -> Result<Tensor> {
    panic!("Not implemented");
}

enum Reduction {
    None,
    All,
}

fn lp_loss(pred: Tensor, target: Tensor, p: f64, reduction: Reduction) -> Result<Tensor> {
    panic!("Not implemented");
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScaleMethod {
    Max,
}

#[derive(Debug, Clone)]
pub struct UniformAffineQuantizerCfg {
    pub symmetric: bool,
    pub num_bits: usize,
    pub leaf_param: bool,
    pub channel_wise: bool,
    pub scale_method: ScaleMethod,
    pub always_zero: bool,
}

#[derive(Debug, Clone)]
pub struct UniformAffineQuantizer {
    symetric: bool,
    num_bits: usize,
    n_levels: usize,
    delta: Option<Tensor>,
    zero_point: Option<Tensor>,
    inited: bool,
    leaf_param: bool,
    channel_wise: bool,
    scale_method: ScaleMethod,
    running_stat: bool,
    always_zero: bool,
    x_min: Option<Tensor>,
    x_max: Option<Tensor>,
}

impl Default for UniformAffineQuantizerCfg {
    fn default() -> Self {
        return UniformAffineQuantizerCfg {
            symmetric: false,
            num_bits: 8,
            leaf_param: false,
            channel_wise: false,
            scale_method: ScaleMethod::Max,
            always_zero: false,
        };
    }
}

impl UniformAffineQuantizer {
    pub fn load(cfg: UniformAffineQuantizerCfg, vb: &VarBuilder) -> Result<Self> {
        let n_levels = if !cfg.symmetric {
            2usize.pow(cfg.num_bits as u32)
        } else {
            2usize.pow((cfg.num_bits - 1) as u32)
        };
        return Ok({
            UniformAffineQuantizer {
                symetric: cfg.symmetric,
                num_bits: cfg.num_bits,
                n_levels,
                delta: None,
                zero_point: None,
                inited: false,
                leaf_param: cfg.leaf_param,
                channel_wise: cfg.channel_wise,
                scale_method: cfg.scale_method,
                running_stat: false,
                always_zero: cfg.always_zero,
                x_min: None,
                x_max: None,
            }
        });
    }

    pub fn init_quantization_scale(
        &mut self,
        x: &Tensor,
        channel_wise: bool,
    ) -> Result<(Tensor, Tensor)> {
        let empty_tensor = Tensor::new(&[0f32], &x.device().clone())?;
        let (mut delta, mut zero_point) = (empty_tensor.clone(), empty_tensor.clone());
        if channel_wise {
            let x_initial = x.clone();
            let n_channels = x_initial.dims()[0];
            let rank = x.dims().len();
            let x_max = if rank == 4 {
                x_initial
                    .abs()?
                    .max(D::Minus1)?
                    .i(0)?
                    .max(D::Minus1)?
                    .i(0)?
                    .max(D::Minus1)?
                    .i(0)?
            } else if rank == 3 {
                x_initial
                    .abs()?
                    .max(D::Minus1)?
                    .i(0)?
                    .max(D::Minus1)?
                    .i(0)?
            } else {
                x_initial.abs()?.max(D::Minus1)?.i(0)?
            };
            delta = x_max.clone();
            zero_point = x_max.clone();

            // determine the scale and zero point channel-by-channel
            let mut delta_c_v = vec![];
            let mut zero_point_c_v = vec![];
            (0..n_channels).for_each(|i| {
                let (delta_c, zero_point_c) =
                    self.init_quantization_scale(&x_initial, false).unwrap();
                delta_c_v.push(delta_c);
                zero_point_c_v.push(zero_point_c);
            });
            (delta, zero_point) = (
                Tensor::cat(&delta_c_v, 0)?,
                Tensor::cat(&zero_point_c_v, 0)?,
            );

            if rank == 4 {
                delta = delta.reshape(((), 1, 1, 1))?;
                zero_point = zero_point.reshape(((), 1, 1, 1))?;
            } else if rank == 3 {
                delta = delta.reshape(((), 1, 1))?;
                zero_point = zero_point.reshape(((), 1, 1))?;
            } else {
                delta = delta.reshape(((), 1))?;
                zero_point = zero_point.reshape(((), 1))?;
            }
        } else {
            if self.leaf_param {
                self.x_min = Some(x.flatten_all()?.min(0)?);
                self.x_max = Some(x.flatten_all()?.max(0)?);
            }

            // TODO revisit this and make it better, doing some janky inefficient stuff
            if self.scale_method == ScaleMethod::Max {
                // TODO need to check this dtype and standardize
                let x_min = x.flatten_all()?.min(0)?.to_vec1::<f32>()?[0];
                let x_min_clamped = x_min.min(0.0);
                let x_max = x.flatten_all()?.max(0)?.to_vec1::<f32>()?[0];
                let x_max_clamped = x_max.max(0.0);

                // TODO: is there a better absmax
                let x_abs_max = x_max_clamped.max(x_min_clamped.abs());

                let mut delta_scalar = if self.symetric {
                    x_abs_max / self.n_levels as f32
                } else {
                    x_max - x_min / (self.n_levels - 1) as f32
                };

                if delta_scalar < 1e-8 {
                    warn!("delta_scalar is too small, setting to 1e-8");
                    delta_scalar = 1e-8;
                }

                delta = Tensor::new(&[delta_scalar], &x.device().clone())?;
            } else {
                panic!("Not implemented");
            }
        };
        Ok((delta, zero_point))
    }

    pub fn forward(&mut self, x: Tensor) -> Result<Tensor> {
        let device = x.device().clone();

        if !self.inited {
            if self.leaf_param {
                let (delta, zero_point) = self.init_quantization_scale(&x, self.channel_wise)?;
                // TODO: need the equivalent here
                // self.delta = torch.nn.Parameter(delta)
                self.delta = Some(delta);
                self.zero_point = Some(zero_point);
            } else {
                let (delta, zero_point) = self.init_quantization_scale(&x, self.channel_wise)?;
                self.delta = Some(delta);
                self.zero_point = Some(zero_point);
            }
        };

        if self.running_stat {
            self.act_momentum_update(&x);
        };

        let x_int = round_ste(&(x / self.delta.clone().unwrap())?)?;
        let mut x_quant = x_int.clamp(&x_int, &Tensor::new(&[0f32], &x_int.device().clone())?)?;

        if self.symetric {
            let lower_bound = Tensor::new(&[(-1.0 * self.n_levels as f32) - 1.0], &device)?;
            let upper_bound = Tensor::new(&[0.0], &device)?;
            x_quant = x_int.clamp(&lower_bound, &upper_bound)?;
        } else {
            let lower_bound = Tensor::new(&[self.n_levels as f32], &device)?;
            let upper_bound = Tensor::new(&[0.0], &x_int.device().clone())?;
            x_quant = x_int.clamp(&lower_bound, &upper_bound)?;
        };
        let x_dequant =
            ((x_quant - self.zero_point.clone().unwrap())? * self.delta.clone().unwrap())?;
        Ok(x_dequant)
    }

    fn act_momentum_update(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.clone())
    }
}
