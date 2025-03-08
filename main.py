import os
import argparse
import torch
from detection.detection_manager import detect_protected_concepts
from rewriting.rewrite_agent import rewrite_prompt
from Models.CFG import test as protected_test
from Models.models import test as unprotected_test
import config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--concepts_json", type=str, default="data/protected_concepts.json")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--model", type=str, default="SDXL", choices=["SD21","SDXL","Flux","SDXLT","SD35L"])
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--num_imgs", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mixing_wt", type=float, default=0.5)
    parser.add_argument("--protect", action="store_true")
    args = parser.parse_args()

    user_prompt = args.prompt
    negative_prompt = args.negative_prompt
    detected_concepts = detect_protected_concepts(user_prompt, concepts_path=args.concepts_json)
    final_prompt = user_prompt
    if args.protect and detected_concepts:
        final_prompt = rewrite_prompt(user_prompt, detected_concepts, args.concepts_json)

    if args.protect:
        protected_test(
            user_prompt=user_prompt,
            protected_prompt=final_prompt,
            negative_prompt=negative_prompt,
            model_id=args.model,
            mixing_wt=args.mixing_wt,
            guidance_scale=args.guidance_scale,
            num_imgs=args.num_imgs,
            output_dir=args.output_dir,
            device=args.device,
            idx=0
        )
    else:
        unprotected_test(
            prompt=user_prompt,
            negative_prompt=negative_prompt,
            model_id=args.model,
            guidance_scale=args.guidance_scale,
            num_imgs=args.num_imgs,
            output_dir=args.output_dir,
            device=args.device,
            idx=0
        )

if __name__ == "__main__":
    main()
