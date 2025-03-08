 # All the models are defined here
from typing import Optional, Tuple
import torch
from diffusers import AutoPipelineForText2Image,StableDiffusionPipeline, DDIMScheduler, FluxPipeline, StableDiffusionXLPipeline
import os
import json
import glob
import numpy as np

from diffusers.utils import logging
logging.set_verbosity_warning

class SD21:
    def __init__(self,
                 model_id: str = "stabilityai/stable-diffusion-2-1",
                 device: str = "cuda"):
        self.device = device
        # initialize the model
        self.pipeline = StableDiffusionPipeline.from_pretrained(
                            model_id,
                            torch_dtype=torch.float32,
                            cache_dir='./cache_dir'
                            ).to(self.device)
        
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.to(device)
    
    def generate(self,
                 prompt:str,
                 num_images: int = 1,
                 height: int = 1024,
                 width: int = 1024,
                 guidance_scale:float = 7.5,
                 num_inference_steps: int = 50):
        
        # Get text embeddings
        tokens = self.pipeline.tokenizer(prompt, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_embeddings = self.pipeline.text_encoder(tokens.input_ids)[0]
        
        # Generate images
        with torch.no_grad():
            images = self.pipeline(
                prompt_embeds=text_embeddings,
                num_images_per_prompt=num_images,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            ).images
        return images
    
class SDXL:
    def __init__(self,
                 model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 device: str = "cuda"):
        self.device = device

        # Initialize SDXL pipeline
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32, #if device == "cuda" else torch.float32,
            #use_safetensors=True,
            cache_dir = './cache_dir'
            #variant="fp16" if device == "cuda" else None
        ).to(device)
        
        self.pipeline.enable_vae_slicing()
    
    def generate(self,
                 prompt: str,
                 num_images: int = 1,
                 mixing_weight: float = 0.1,
                 negative_prompt: Optional[str] = None,
                 guidance_scale: float = 7.5,
                 num_inference_steps: int = 50,
                 seed: Optional[int] = None,
                 height: int = 1024,  # SDXL default
                 width: int = 1024,   # SDXL default
                 target_size: Optional[Tuple[int, int]] = None,
                 original_size: Optional[Tuple[int, int]] = None,
                 crops_coords_top_left: Tuple[int, int] = (0, 0),
                 eta: float = 0.0 ):
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Set default sizes if not provided
        if target_size is None:
            target_size = (height, width)
        if original_size is None:
            original_size = (height, width)
            
        if negative_prompt is None:
            negative_prompt = ""
        
        try:
            # Generate images
            output = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                target_size=target_size,
                original_size=original_size,
                crops_coords_top_left=crops_coords_top_left,
                eta=eta
            )
            
            return output.images
            
        except Exception as e:
            print(f"Error generating images: {str(e)}")
            return None

class Flux:
    def __init__(self,
                 model_id: str = "black-forest-labs/FLUX.1-dev",
                 device: str = "cuda"):
        self.device = device
        self.pipeline = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            cache_dir='./cache_dir',
            token=os.getenv("HF"),
        ).to(device)
    
    def generate(self,
                 prompt: str,
                 negative_prompt: Optional[str] = None,
                 num_images: int = 1,
                 guidance_scale: float = 7.5,
                 num_inference_steps: int = 50,
                 seed: Optional[int] = None,
                 height: int = 1024,
                 width: int = 1024):
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        try:
            # Generate images
            output = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width
            )
            
            return output.images
            
        except Exception as e:
            print(f"Error generating images: {str(e)}")
            return None
        
class SDXLT:
    def __init__(self,
                 model_id: str = "stabilityai/sdxl-turbo",
                 device: str = "cuda:2"):
        self.device = device
        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            cache_dir='./cache_dir'
        ).to(device)
    
    def generate(self,
                 prompt: str,
                 negative_prompt: Optional[str] = None,
                 num_images: int = 1,
                 guidance_scale: float = 0.0,  # Default for Turbo is 0.0
                 num_inference_steps: int = 1,  # Turbo works well with just 1-4 steps
                 seed: Optional[int] = None,
                 height: int = 512,            # SDXL-Turbo default
                 width: int = 512):            # SDXL-Turbo default
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        try:
            # Generate images
            output = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width
            )
            
            return output.images
            
        except Exception as e:
            print(f"Error generating images: {str(e)}")
            return None
    
class SD35L:
    def __init__(self,
                 model_id: str = "stabilityai/stable-diffusion-3",  # This is the large model
                 device: str = "cuda:2"):
        self.device = device
        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            cache_dir='./cache_dir'
        ).to(device)
    
    def generate(self,
                 prompt: str,
                 negative_prompt: Optional[str] = None,
                 num_images: int = 1,
                 guidance_scale: float = 7.0,
                 num_inference_steps: int = 50,
                 seed: Optional[int] = None,
                 height: int = 1024,
                 width: int = 1024):
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        try:
            # Generate images
            output = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width
            )
            
            return output.images
            
        except Exception as e:
            print(f"Error generating images: {str(e)}")
            return None

def test(prompt: str, negative_prompt:str, model_id: str, guidance_scale:float, num_imgs:int, output_dir: str, device: str, idx:int):
    
    # Initialize the model
    if model_id == "SD21":
        model = SD21(device=device)
    elif model_id == "SDXL":
        model = SDXL(device=device)
    elif model_id == "Flux":
        model = Flux(device=device)
    elif model_id == "SDXLT":
        model = SDXLT(device=device)
    elif model_id == "SD35L":
        model = SD35L(device=device)
    else:
        raise ValueError(f"Invalid model_id: {model_id}")

    if negative_prompt is "":
        output_dir = f"{output_dir}/unprotected_{guidance_scale}/prompt{idx:04}"
    else:
        output_dir = f"{output_dir}/negative_prompting/unprotected_{guidance_scale}/prompt{idx:04}"

    if os.path.exists(f"{output_dir}"):
        png_files = glob.glob(os.path.join(output_dir, '*.png'))
        if len(png_files) == num_imgs:
            return

    os.makedirs(output_dir, exist_ok=True)
    
    # Generate images
    if num_imgs > 4:
        a = num_imgs//2
        if num_imgs % 2:
            b = num_imgs//2 + 1
    else:
        a = num_imgs
        b = 0
    images = model.generate(prompt=prompt, negative_prompt=negative_prompt, guidance_scale=guidance_scale, num_images=a)
    
    for i, img in enumerate(images):
        output_path = f"{output_dir}/img{i:03}.png"
        img.save(output_path)
        print(f"Saved image to: {output_path}")

    if b:
        images = model.generate(prompt=prompt, negative_prompt=negative_prompt, guidance_scale=guidance_scale, num_images=5)
        
        for i, img in enumerate(images):
            output_path = f"{output_dir}/img{i+a:03}.png"
            img.save(output_path)
            print(f"Saved image to: {output_path}")