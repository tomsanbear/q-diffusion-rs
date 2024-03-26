#[cfg(feature = "accelerate")]
extern crate accelerate_src;

mod block;
mod layer;
mod quant;
mod stable_diffusion;
mod utils;

use anyhow::{Error as E, Result};
use candle_core::{utils::metal_is_available, DType, Device, IndexOp, Module, Tensor, D};
use clap::Parser;
use tokenizers::Tokenizer;

use crate::utils::save_image;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The prompt to be used for image generation.
    #[arg(
        long,
        default_value = "1girl, souryuu asuka langley, neon genesis evangelion, solo, upper body, v, smile, looking at viewer, outdoors, night"
    )]
    prompt: String,

    /// The negative prompt to be used for image generation.
    #[arg(
        long,
        default_value = "nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]"
    )]
    negative_prompt: String,

    /// Run on CPU rather than on GPU.
    #[arg(long, default_value = "false")]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// The height in pixels of the generated image.
    #[arg(long, default_value = "832")]
    height: Option<usize>,

    /// The width in pixels of the generated image.
    #[arg(long, default_value = "1216")]
    width: Option<usize>,

    /// The UNet weight file, in .safetensors format.
    #[arg(long, value_name = "FILE")]
    unet_weights: Option<String>,

    /// The CLIP weight file, in .safetensors format.
    #[arg(long, value_name = "FILE")]
    clip_weights: Option<String>,

    /// The VAE weight file, in .safetensors format.
    #[arg(long, value_name = "FILE")]
    vae_weights: Option<String>,

    #[arg(long, value_name = "FILE")]
    /// The file specifying the tokenizer to used for tokenization.
    tokenizer: Option<String>,

    /// The size of the sliced attention or 0 for automatic slicing (disabled by default)
    #[arg(long)]
    sliced_attention_size: Option<usize>,

    /// The number of steps to run the diffusion for.
    #[arg(long)]
    n_steps: Option<usize>,

    /// The number of samples to generate.
    #[arg(long, default_value_t = 1)]
    num_samples: i64,

    /// The name of the final image to generate.
    #[arg(long, value_name = "FILE", default_value = "output/example.png")]
    final_image: String,

    #[arg(long, value_enum, default_value = "xl")]
    sd_version: StableDiffusionVersion,

    /// Generate intermediary images at each step.
    #[arg(long, action)]
    intermediary_images: bool,

    #[arg(long)]
    use_flash_attn: bool,

    #[arg(long, default_value = "true")]
    use_f16: bool,

    #[arg(long, default_value = "7")]
    guidance_scale: Option<f64>,

    #[arg(long)]
    seed: Option<u64>,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
enum StableDiffusionVersion {
    Xl,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelFile {
    Tokenizer,
    Tokenizer2,
    Clip,
    Clip2,
    Unet,
    Vae,
}

impl StableDiffusionVersion {
    fn repo(&self) -> &'static str {
        match self {
            Self::Xl => "cagliostrolab/animagine-xl-3.1",
        }
    }

    fn unet_file(&self) -> &'static str {
        "unet/diffusion_pytorch_model.safetensors"
    }

    fn vae_file(&self) -> &'static str {
        "vae/diffusion_pytorch_model.safetensors"
    }

    fn clip_file(&self) -> &'static str {
        "text_encoder/model.safetensors"
    }

    fn clip2_file(&self) -> &'static str {
        "text_encoder_2/model.safetensors"
    }
}

impl ModelFile {
    fn get(
        &self,
        filename: Option<String>,
        version: StableDiffusionVersion,
    ) -> Result<std::path::PathBuf> {
        use hf_hub::api::sync::Api;
        match filename {
            Some(filename) => Ok(std::path::PathBuf::from(filename)),
            None => {
                let (repo, path) = match self {
                    Self::Tokenizer => {
                        let tokenizer_repo = match version {
                            StableDiffusionVersion::Xl => "openai/clip-vit-large-patch14",
                        };
                        (tokenizer_repo, "tokenizer.json")
                    }
                    Self::Tokenizer2 => {
                        ("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", "tokenizer.json")
                    }
                    Self::Clip => (version.repo(), version.clip_file()),
                    Self::Clip2 => (version.repo(), version.clip2_file()),
                    Self::Unet => (version.repo(), version.unet_file()),
                    Self::Vae => (version.repo(), version.vae_file()),
                };
                let filename = Api::new()?.model(repo.to_string()).get(path)?;
                Ok(filename)
            }
        }
    }
}

fn output_filename(
    basename: &str,
    sample_idx: i64,
    num_samples: i64,
    timestep_idx: Option<usize>,
) -> String {
    let filename = if num_samples > 1 {
        match basename.rsplit_once('.') {
            None => format!("{basename}.{sample_idx}.png"),
            Some((filename_no_extension, extension)) => {
                format!("{filename_no_extension}.{sample_idx}.{extension}")
            }
        }
    } else {
        basename.to_string()
    };
    match timestep_idx {
        None => filename,
        Some(timestep_idx) => match filename.rsplit_once('.') {
            None => format!("{filename}-{timestep_idx}.png"),
            Some((filename_no_extension, extension)) => {
                format!("{filename_no_extension}-{timestep_idx}.{extension}")
            }
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn text_embeddings(
    prompt: &str,
    negative_prompt: &str,
    tokenizer: Option<String>,
    clip_weights: Option<String>,
    sd_version: StableDiffusionVersion,
    sd_config: &stable_diffusion::StableDiffusionConfig,
    device: &Device,
    dtype: DType,
    use_guide_scale: bool,
    first: bool,
) -> Result<Tensor> {
    let tokenizer_file = if first {
        ModelFile::Tokenizer
    } else {
        ModelFile::Tokenizer2
    };
    let tokenizer = tokenizer_file.get(tokenizer, sd_version)?;
    let tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;
    let pad_id = match &sd_config.clip.pad_with {
        Some(padding) => *tokenizer.get_vocab(true).get(padding.as_str()).unwrap(),
        None => *tokenizer.get_vocab(true).get("<|endoftext|>").unwrap(),
    };
    println!("Running with prompt \"{prompt}\".");
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    if tokens.len() > sd_config.clip.max_position_embeddings {
        anyhow::bail!(
            "the prompt is too long, {} > max-tokens ({})",
            tokens.len(),
            sd_config.clip.max_position_embeddings
        )
    }
    while tokens.len() < sd_config.clip.max_position_embeddings {
        tokens.push(pad_id)
    }
    let tokens = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;

    println!("Building the Clip transformer.");
    let clip_weights_file = if first {
        ModelFile::Clip
    } else {
        ModelFile::Clip2
    };
    let clip_weights = clip_weights_file.get(clip_weights, sd_version)?;
    let clip_config = if first {
        &sd_config.clip
    } else {
        sd_config.clip2.as_ref().unwrap()
    };
    let text_model =
        stable_diffusion::build_clip_transformer(clip_config, clip_weights, device, DType::F32)?;
    let text_embeddings = text_model.forward(&tokens)?;

    let text_embeddings = if use_guide_scale {
        let mut uncond_tokens = tokenizer
            .encode(negative_prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        if uncond_tokens.len() > sd_config.clip.max_position_embeddings {
            anyhow::bail!(
                "the negative prompt is too long, {} > max-tokens ({})",
                uncond_tokens.len(),
                sd_config.clip.max_position_embeddings
            )
        }
        while uncond_tokens.len() < sd_config.clip.max_position_embeddings {
            uncond_tokens.push(pad_id)
        }

        let uncond_tokens = Tensor::new(uncond_tokens.as_slice(), device)?.unsqueeze(0)?;
        let uncond_embeddings = text_model.forward(&uncond_tokens)?;

        Tensor::cat(&[uncond_embeddings, text_embeddings], 0)?.to_dtype(dtype)?
    } else {
        text_embeddings.to_dtype(dtype)?
    };
    Ok(text_embeddings)
}

fn run(args: Args) -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let Args {
        prompt,
        negative_prompt,
        cpu,
        height,
        width,
        n_steps,
        tokenizer,
        final_image,
        sliced_attention_size,
        num_samples,
        sd_version,
        clip_weights,
        vae_weights,
        unet_weights,
        tracing,
        use_f16,
        guidance_scale,
        use_flash_attn,
        seed,
        ..
    } = args;

    let _guard = if tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    let guidance_scale = match guidance_scale {
        Some(guidance_scale) => guidance_scale,
        None => match sd_version {
            StableDiffusionVersion::Xl => 7.5,
        },
    };
    let n_steps = match n_steps {
        Some(n_steps) => n_steps,
        None => match sd_version {
            StableDiffusionVersion::Xl => 30,
        },
    };
    let dtype = if use_f16 { DType::F16 } else { DType::F32 };
    let sd_config = match sd_version {
        StableDiffusionVersion::Xl => {
            stable_diffusion::StableDiffusionConfig::sdxl(sliced_attention_size, height, width)
        }
    };

    let scheduler = sd_config.build_scheduler(n_steps)?;

    let device = if metal_is_available() {
        Device::new_metal(0)?
    } else if cpu {
        Device::Cpu
    } else {
        Device::new_cuda(0)?
    };

    if let Some(seed) = seed {
        device.set_seed(seed)?;
    }
    let use_guide_scale = guidance_scale > 1.0;

    let which = vec![true, false];
    let text_embeddings = which
        .iter()
        .map(|first| {
            text_embeddings(
                &prompt,
                &negative_prompt,
                tokenizer.clone(),
                clip_weights.clone(),
                sd_version,
                &sd_config,
                &device,
                dtype,
                use_guide_scale,
                *first,
            )
        })
        .collect::<Result<Vec<_>>>()?;

    let text_embeddings = Tensor::cat(&text_embeddings, D::Minus1)?;
    println!("{text_embeddings:?}");

    println!("Building the autoencoder.");
    let vae_weights = ModelFile::Vae.get(vae_weights, sd_version)?;
    let vae = sd_config.build_vae(vae_weights, &device, dtype)?;
    println!("Building the unet.");
    let unet_weights = ModelFile::Unet.get(unet_weights, sd_version)?;
    let unet = sd_config.build_unet(unet_weights, &device, 4, use_flash_attn, dtype)?;
    let t_start = 0;
    let bsize = 1;

    let vae_scale = match sd_version {
        StableDiffusionVersion::Xl => 0.18215,
    };

    for idx in 0..num_samples {
        let timesteps = scheduler.timesteps();
        let latents = {
            let latents = Tensor::randn(
                0f32,
                1f32,
                (bsize, 4, sd_config.height / 8, sd_config.width / 8),
                &device,
            )?;
            // scale the initial noise by the standard deviation required by the scheduler
            (latents * scheduler.init_noise_sigma())?
        };
        let mut latents = latents.to_dtype(dtype)?;

        println!("starting sampling");
        for (timestep_index, &timestep) in timesteps.iter().enumerate() {
            if timestep_index < t_start {
                continue;
            }
            let start_time = std::time::Instant::now();
            let latent_model_input = if use_guide_scale {
                Tensor::cat(&[&latents, &latents], 0)?
            } else {
                latents.clone()
            };

            let latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)?;
            let noise_pred =
                unet.forward(&latent_model_input, timestep as f64, &text_embeddings)?;

            let noise_pred = if use_guide_scale {
                let noise_pred = noise_pred.chunk(2, 0)?;
                let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);

                (noise_pred_uncond + ((noise_pred_text - noise_pred_uncond)? * guidance_scale)?)?
            } else {
                noise_pred
            };

            latents = scheduler.step(&noise_pred, timestep, &latents)?;
            let dt = start_time.elapsed().as_secs_f32();
            println!("step {}/{n_steps} done, {:.2}s", timestep_index + 1, dt);

            if args.intermediary_images {
                let image = vae.decode(&(&latents / vae_scale)?)?;
                let image = ((image / 2.)? + 0.5)?.to_device(&Device::Cpu)?;
                let image = (image * 255.)?.to_dtype(DType::U8)?.i(0)?;
                let image_filename =
                    output_filename(&final_image, idx + 1, num_samples, Some(timestep_index + 1));
                save_image(&image, image_filename)?
            }
        }

        println!(
            "Generating the final image for sample {}/{}.",
            idx + 1,
            num_samples
        );
        let image = vae.decode(&(&latents / vae_scale)?)?;
        let image = ((image / 2.)? + 0.5)?.to_device(&Device::Cpu)?;
        let image = (image.clamp(0f32, 1.)? * 255.)?.to_dtype(DType::U8)?.i(0)?;
        let image_filename = output_filename(&final_image, idx + 1, num_samples, None);
        save_image(&image, image_filename)?
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    run(args)
}
