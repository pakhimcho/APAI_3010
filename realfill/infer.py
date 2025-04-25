import argparse
import os
from pathlib import Path

import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for RealFill")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--validation_image", type=str, required=True, help="Path to the validation image")
    parser.add_argument("--validation_mask", type=str, required=True, help="Path to the validation mask")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for generated images")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    return parser.parse_args()

def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load the pipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
    ).to("cuda")

    # Load validation image and mask
    image = Image.open(args.validation_image).convert("RGB").resize((512, 512))
    mask = Image.open(args.validation_mask).convert("L").resize((64, 64))

    # Convert to tensors
    image = np.array(image).astype(np.float32) / 255.0
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = (mask > 0).astype(np.uint8)

    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to("cuda").to(torch.float16)
    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to("cuda")

    # Encode image to latent space
    with torch.no_grad():
        latents = pipe.vae.encode(image).latent_dist.sample() * pipe.vae.config.scaling_factor

    # Generate images
    prompt = "a photo of a woman with flowers"  # Adjusted for flowerwoman dataset
    num_images = 16
    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(num_images):
        with torch.no_grad():
            output = pipe(
                prompt=prompt,
                image=latents,
                mask_image=mask,
                num_inference_steps=250,
                guidance_scale=5.0,
            ).images[0]

        output.save(os.path.join(args.output_dir, f"{i}.png"))

if __name__ == "__main__":
    main()
