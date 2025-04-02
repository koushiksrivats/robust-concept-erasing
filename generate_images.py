from diffusers import StableDiffusionPipeline
import torch
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images from I2P dataset")

    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--unet_checkpoint", type=str, help="Path to erased unet checkpoint", default="")
    parser.add_argument("--prompt", type=str, help="Prompt for image generation")
    parser.add_argument("--num_images", type=int, help="Number of images to generate for testing", default=10)
    parser.add_argument('--num_inference_steps', help='num_inference_steps', type=int, required=False, default=50)
    parser.add_argument('--guidance_scale', help='guidance_scale', type=int, required=False, default=7.5)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    gen = torch.Generator(device)

    os.makedirs(args.output_dir, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(args.model_path, safety_checker=None, torch_dtype=torch.float16).to(device)

    if(args.unet_checkpoint != ""):
        print("Loading erased unet checkpoint from ", args.unet_checkpoint)
        pipe.unet.load_state_dict(torch.load(args.unet_checkpoint))

    with torch.no_grad():
        for i in range(args.num_images):
            gen.manual_seed(i)
            torch.manual_seed(i)
            out = pipe(prompt=[args.prompt], generator=gen, num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale)
            image = out.images[0]
            #  save image
            filename = '_'.join(args.prompt.split(" "))
            image.save(os.path.join(args.output_dir, f"{filename}_{i}.png"))
