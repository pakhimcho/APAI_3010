import argparse
import os
import torch
import numpy as np
from PIL import Image, ImageFilter
from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler
from transformers import CLIPTextModel, UNet2DConditionModel

parser = argparse.ArgumentParser(description="Inference")
parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained model.")
parser.add_argument("--validation_image", type=str, required=True, help="Path to target image.")
parser.add_argument("--validation_mask", type=str, required=True, help="Path to mask image.")
parser.add_argument("--output_dir", type=str, default="./test-infer/", help="Output directory.")
parser.add_argument("--seed", type=int, default=None, help="Seed for reproducible inference.")

args = parser.parse_args()

if __name__ == "__main__":
    os.makedirs(args.output_dir, exist_ok=True)
    generator = torch.Generator(device="cuda").manual_seed(args.seed) if args.seed else None

    # Load pipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
        revision=None
    )
    pipe.unet.load_state_dict(
        torch.load(os.path.join(args.model_path, "unet/diffusion_pytorch_model.bin"), map_location="cpu"),
        strict=False
    )
    pipe.text_encoder = CLIPTextModel.from_pretrained(
        args.model_path, subfolder="text_encoder", torch_dtype=torch.float16
    )
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")

    # Load and preprocess images
    target_image = Image.open(args.validation_image).convert("RGB").resize((512, 512))
    mask_image = Image.open(args.validation_mask).convert("L").resize((512, 512))

    # Process mask
    mask_array = np.array(mask_image)
    mask_array = np.where(mask_array > 128, 255, 0).astype(np.uint8)
    mask_image = Image.fromarray(mask_array)
    erode_kernel = ImageFilter.MaxFilter(3)
    mask_image = mask_image.filter(erode_kernel)
    blur_kernel = ImageFilter.BoxBlur(1)
    mask_image = mask_image.filter(blur_kernel)

    # Convert images to tensors
    target_tensor = pipe.image_processor.preprocess(target_image).to("cuda")
    mask_tensor = pipe.image_processor.preprocess(mask_image, height=512, width=512).to("cuda")

    # Encode to latent space
    with torch.no_grad():
        latents = pipe.vae.encode(target_tensor).latent_dist.sample() * pipe.vae.config.scaling_factor
        mask_latent = torch.nn.functional.interpolate(
            mask_tensor, size=(64, 64), mode="bilinear", align_corners=False
        )

    # Run inference
    for idx in range(16):
        result = pipe(
            prompt="a photo of sks",
            image=latents,
            mask_image=mask_latent,
            num_inference_steps=200,
            guidance_scale=1.0,
            generator=generator,
            height=512,
            width=512,
        ).images[0]

        result = Image.composite(result, target_image, mask_image)
        result.save(f"{args.output_dir}/{idx}.png")

    del pipe
    torch.cuda.empty_cache()
