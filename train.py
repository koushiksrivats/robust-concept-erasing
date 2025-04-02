import argparse
import os
import random
import torch
import numpy as np
from utils.stereo import stereo, attack_stereo
from utils.utils import StableDiffuser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concept Erasing and Textual Inversion")
    parser.add_argument("--erase_concept", required=True, help="Concept to erase")
    parser.add_argument("--train_method", required=True, help="Method for training (OPTIONS: noxattn/xattn)")
    parser.add_argument("--iterations", type=int, default=200, help="Number of iterations for the erasing objectives")
    parser.add_argument("--negative_guidance", type=float, default=2.0, help="Negative guidance value")
    parser.add_argument("--ste_lr", type=float, default=0.5e-5, help="Learning rate for erasing in search throughly enough stage")
    parser.add_argument("--reo_lr", type=float, default=2e-5, help="Learning rate for erasing in robustly erase once stage")
    parser.add_argument("--ci_lr", type=float, default=5e-3, help="Learning rate for textual inversion")
    parser.add_argument("--ti_max_train_steps", type=int, default=3000, help="Maximum training steps for textual inversion")
    parser.add_argument("--train_data_dir", type=str, required=False, help="Gallery images to be used during training")
    parser.add_argument("--learnable_property", type=str, required=False, help="object/style", default="object")
    parser.add_argument("--initializer_token", type=str, required=True, help="Initializer token (OPTIONS: person/object/art)")
    parser.add_argument('--device', help='cuda device to train on', type=str, required=False, default='cuda')
    parser.add_argument("--n_iterations", type=int, required=False, help="Total number of erasure-attack iterations", default=4)   
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for saving models")
    parser.add_argument("--generic_prompt", type=str, required=False, help="Generic prompt for textual inversion visualization", default="a photo of a")
    parser.add_argument("--anchor_concept_path", type=str, required=False, help="Path to anchor concept json used in REO stage", default='utils/anchor_prompts.json')
    parser.add_argument("--compositional_guidance_scale", type=float, required=False, help="Compositional guidance scale. The value has to be +1 of the scale you would like to set. If the intended scale is 1.0, then the value has to be 2.0", default=2.0)
    parser.add_argument("--mode", type=str, required=False, help="Mode of operation (OPTIONS: stereo/attack/both)", default="stereo")
    parser.add_argument("--unet_ckpt_to_attack", type=str, required=False, help="Path to the unet ckpt that has to be attacked to test its robustness", default="final_reo_unet.pt")
    parser.add_argument("--attack_eval_images", type=str, required=True, help="Gallery images to be used for attacking the model for evaluation")
    parser.add_argument("--center_crop", type=bool, required=False, help="Center crop the images during training", default=False)
    parser.add_argument("--num_of_adv_concepts", type=int, required=False, help="Number of adversarial concepts to use in REO", default=4)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Set the random seed for reproducibility
    seed = 42
    np.random.seed(seed)      # For numpy
    random.seed(seed)         # For the random module
    torch.manual_seed(seed)   # For PyTorch

    # For CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you have multiple GPUs

    # Ensure PyTorch operations are deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.mode == "stereo":
        stereo(args)
    elif args.mode == "attack":
        diffuser = StableDiffuser(scheduler='DDIM').to(args.device)
        attack_stereo(args, diffuser)
    elif args.mode == 'both':
        stereo(args)
        diffuser = StableDiffuser(scheduler='DDIM').to(args.device)
        attack_stereo(args, diffuser)