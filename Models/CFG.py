import torch
import torch.nn as nn
import torch.nn.functional as F
# from optimum.quanto import freeze, qfloat8, quantize

from diffusers import StableDiffusionXLPipeline, AutoPipelineForText2Image,StableDiffusionPipeline, AutoencoderKL, DiffusionPipeline, FluxPipeline

from diffusers import FluxTransformer2DModel
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

# from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from diffusers import DDIMScheduler, FlowMatchEulerDiscreteScheduler, EulerAncestralDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection, T5EncoderModel, T5TokenizerFast, BitsAndBytesConfig
import numpy as np
from typing import Dict, List, Union, Tuple, Optional
from dataclasses import dataclass

import os
import glob
import sys
import json
import config
import gc

from diffusers.utils import logging

logging.set_verbosity_warning()

# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

class AdaptiveProtectedStableDiffusionXL:
    def __init__(self,
                 model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 device: str = "cuda"):
        
        self.device = device
        # Initialize SDXL pipeline
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32, #if device == "cuda" else torch.float32,
            #use_safetensors=True,
            cache_dir = './cache_dir',
            #variant="fp16" if device == "cuda" else None
        ).to(device)

        # self.pipeline.enable_sequential_cpu_offload()
        self.pipeline.enable_vae_slicing()
        
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config, steps_offset = 0)
        self.pipeline = self.pipeline.to(device)
    
    def generate(self,
                prompt: str,
                protected_prompt: str,
                num_images: int = 1,
                mixing_weight: float = 0.1,
                negative_prompt: Optional[str] = None,
                guidance_scale: float = 7.5,
                num_inference_steps: int = 50,
                return_metadata: bool = False,
                seed: Optional[int] = 20,
                height: int = 1024,  # SDXL default
                width: int = 1024,   # SDXL default
                target_size: Optional[Tuple[int, int]] = None,
                original_size: Optional[Tuple[int, int]] = None,
                crops_coords_top_left: Tuple[int, int] = (0, 0),
                aesthetic_score: float = 6.0,
                negative_aesthetic_score: float = 2.5,
                eta: float = 0.0 
                ) -> Union[List[np.ndarray], Tuple[List[np.ndarray], dict]]:
        """Generate images with adaptive concept protection for SDXL"""
        
        if seed is not None:
            torch.manual_seed(seed)
        
        # Set default sizes if not provided
        if target_size is None:
            target_size = (height, width)
        if original_size is None:
            original_size = (height, width)
        
        # Tokenize prompt for both encoders
        tokens = self.pipeline.tokenizer(
            num_images*[prompt],
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        tokens_2 = self.pipeline.tokenizer_2(
            num_images*[prompt],
            padding="max_length",
            max_length=self.pipeline.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        if negative_prompt is None:
            negative_prompt = ""
            
        # Tokenize negative prompt for both encoders
        neg_tokens = self.pipeline.tokenizer(
            num_images*[negative_prompt],
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        neg_tokens_2 = self.pipeline.tokenizer_2(
            num_images*[negative_prompt],
            padding="max_length",
            max_length=self.pipeline.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Tokenize protected_prompt for both encoders
        protected_tokens = self.pipeline.tokenizer(
            num_images*[protected_prompt],
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        protected_tokens_2 = self.pipeline.tokenizer_2(
            num_images*[protected_prompt],
            padding="max_length",
            max_length=self.pipeline.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        
        # Get embeddings from both encoders
        with torch.no_grad():
            text_encoder_output_1  = self.pipeline.text_encoder(tokens.input_ids, output_hidden_states=True)
            text_encoder_output_2    = self.pipeline.text_encoder_2(tokens_2.input_ids, output_hidden_states=True)
            
            negative_encoder_output_1  = self.pipeline.text_encoder(neg_tokens.input_ids, output_hidden_states=True)
            negative_encoder_output_2  = self.pipeline.text_encoder_2(neg_tokens_2.input_ids, output_hidden_states=True)
            
            protected_encoder_output_1  = self.pipeline.text_encoder(protected_tokens.input_ids, output_hidden_states=True)
            protected_encoder_output_2  = self.pipeline.text_encoder_2(protected_tokens_2.input_ids, output_hidden_states=True)
        
        pooled_prompt_embeds = text_encoder_output_2[0]
        text_embeddings_1 = text_encoder_output_1.hidden_states[-2]
        text_embeddings_2 = text_encoder_output_2.hidden_states[-2]
        
        negative_pooled_prompt_embeds = negative_encoder_output_2[0]
        negative_embeddings_1 = negative_encoder_output_1.hidden_states[-2]
        negative_embeddings_2 = negative_encoder_output_2.hidden_states[-2]
        
        protected_pooled_prompt_embeds = protected_encoder_output_2[0]
        protected_embeddings_1 = protected_encoder_output_1.hidden_states[-2]
        protected_embeddings_2 = protected_encoder_output_2.hidden_states[-2]

        # # print the embeddings:
        # print(f"text_embeddings_1 shape: {text_embeddings_1.shape}")
        # print(f"text_embeddings_2 shape: {text_embeddings_2.shape}")
        
        # Get time embeddings for SDXL
        add_time_ids = self._get_add_time_ids(
            original_size,
            target_size,
            crops_coords_top_left,
            aesthetic_score,
            negative_aesthetic_score,
            dtype=text_embeddings_1.dtype
        )
        add_time_ids = add_time_ids.to(self.device)
        add_time_ids = add_time_ids.repeat(1 * num_images, 1)
        
        latents = torch.randn(
            (1*num_images, self.pipeline.unet.config.in_channels, height // 8, width // 8),
            device=self.device,
            dtype=text_embeddings_1.dtype
        )
        
        self.pipeline.scheduler.set_timesteps(num_inference_steps, device=self.device)
        latents = latents * self.pipeline.scheduler.init_noise_sigma
        
        # Prepare extra step kwargs
        extra_step_kwargs = self.pipeline.prepare_extra_step_kwargs(generator=None, eta=eta)
        
        images = []
        # for _ in range(num_images):
        current_latents = latents
            
        for t in self.pipeline.progress_bar(self.pipeline.scheduler.timesteps):
            
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([current_latents] * 2)
            latent_model_input = self.pipeline.scheduler.scale_model_input(latent_model_input, t)
            
            # Get mixed text embeddings
            mixed_embeddings_1 = torch.cat([
                negative_embeddings_1,
                (1 - mixing_weight) * text_embeddings_1 + mixing_weight * protected_embeddings_1
            ], dim = 0)
            
            mixed_embeddings_2 = torch.cat([
                negative_embeddings_2,
                (1 - mixing_weight) * text_embeddings_2 + mixing_weight * protected_embeddings_2
            ], dim = 0)
            
            # Ensure both embeddings have same dimensions
            if mixed_embeddings_1.dim() != mixed_embeddings_2.dim():
                if mixed_embeddings_1.dim() == 2:
                    mixed_embeddings_1 = mixed_embeddings_1.unsqueeze(0)
                if mixed_embeddings_2.dim() == 2:
                    mixed_embeddings_2 = mixed_embeddings_2.unsqueeze(0)
            
            # Now concatenate the embeddings
            
            concatenated_embeddings = torch.cat([mixed_embeddings_1, mixed_embeddings_2], dim=-1)
            
            # Prepare added conditions
            # Expand time embeddings
            text_embeds = protected_pooled_prompt_embeds
            text_embeds = torch.cat([negative_pooled_prompt_embeds, text_embeds], dim=0)
            added_cond_kwargs = {"text_embeds": text_embeds,"time_ids": add_time_ids}
            
            # # Print shapes for debugging
            # print(f"Shape of embeddings_1: {mixed_embeddings_1.shape}")
            # print(f"Shape of embeddings_2: {mixed_embeddings_2.shape}")
            # print(f"Shape of concatenated: {concatenated_embeddings.shape}")
            # print(f"Shape of time ids: {add_time_ids.shape}")
            # print(f"Shape of text embeds : {text_embeds.shape}")

            
            # Predict noise with concatenated embeddings
            with torch.no_grad():
                noise_pred = self.pipeline.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=concatenated_embeddings,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False
                )[0]
            
            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous noisy sample
            current_latents = self.pipeline.scheduler.step(
                noise_pred,
                t,
                current_latents,
                **extra_step_kwargs
            ).prev_sample
        
        # Decode latents
        with torch.no_grad():
            images = self.pipeline.vae.decode(
                current_latents / self.pipeline.vae.config.scaling_factor,
                return_dict=False
            )[0]
        
        # Convert to image
        images = self.pipeline.image_processor.postprocess(
            images,
            output_type="pil"
        )
        
        return images

    def _get_add_time_ids(self,
                        original_size: Tuple[int, int],
                        target_size: Tuple[int, int],
                        crops_coords_top_left: Tuple[int, int],
                        aesthetic_score: float,
                        negative_aesthetic_score: float,
                        dtype: torch.dtype) -> torch.Tensor:
        """Helper to get SDXL's additional time embeddings"""
        """ Note: Aesthetic score is for SDXL img2img pipeline, need to enable `pipe.register_to_config(requires_aesthetic_score=True)`"""
        if not self.pipeline.text_encoder_2:
            raise ValueError("Missing text encoder 2, incorrect call of `get_add_time_ids` on non-SDXL pipeline.")
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        
        # if (
        #     aesthetic_score is not None
        #     and negative_aesthetic_score is not None
        # ):
        #     add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
        #     add_neg_time_ids = list(original_size + crops_coords_top_left + (negative_aesthetic_score,))
        # else:
        #     add_time_ids = list(original_size + crops_coords_top_left + target_size)
        #     add_neg_time_ids = None
        
        passed_add_embed_dim = (
            self.pipeline.unet.config.addition_time_embed_dim * len(add_time_ids) + self.pipeline.text_encoder_2.config.projection_dim
        )
        expected_add_embed_dim = self.pipeline.unet.add_embedding.linear_1.in_features
        if (
            expected_add_embed_dim > passed_add_embed_dim
            and (expected_add_embed_dim - passed_add_embed_dim) == self.pipeline.unet.config.addition_time_embed_dim # type: ignore[attr-defined]
        ):
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to enable `requires_aesthetic_score` with `pipe.register_to_config(requires_aesthetic_score=True)` to make sure `aesthetic_score` {aesthetic_score} and `negative_aesthetic_score` {negative_aesthetic_score} is correctly used by the model."
            )
        elif (
            expected_add_embed_dim < passed_add_embed_dim
            and (passed_add_embed_dim - expected_add_embed_dim) == self.pipeline.unet.config.addition_time_embed_dim # type: ignore[attr-defined]
        ):
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to disable `requires_aesthetic_score` with `pipe.register_to_config(requires_aesthetic_score=False)` to make sure `target_size` {target_size} is correctly used by the model."
            )
        elif expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)  # type: ignore
        # if add_neg_time_ids is not None:
        #     add_neg_time_ids = torch.tensor([add_neg_time_ids], dtype=dtype)  # type: ignore

        # For Classifier free guidance
        # add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim = 0)
        return add_time_ids

class AdaptiveProtectedStableDiffusion21:

    
    def __init__(self,
                 model_id: str = "stabilityai/stable-diffusion-2-1",
                 protected_concepts: Dict[str, str] = None,
                 replacement_concepts: Dict[str, str] = None,
                 device: str = "cuda"):
        
        self.device = device
        # Initialize SD pipeline
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            cache_dir = './cache_dir'
            ).to(device)
        
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.to(device)
        
        # logger.info(f"Adaptive Protected Stable Diffusion initialized on {device}")
    
    def generate(self,
                prompt: str,
                protected_prompt: str,
                num_images: int = 1,
                mixing_weight: float = 0.1,
                negative_prompt: Optional[str] = None,
                guidance_scale: float = 7.5,
                num_inference_steps: int = 50,
                return_metadata: bool = False,
                seed: Optional[int] = 20,
                height: int = 1024,
                width: int = 1024
                ) -> Union[List[np.ndarray], Tuple[List[np.ndarray], dict]]:
        """Generate images with adaptive concept protection"""
        
        if seed is not None:
            torch.manual_seed(seed)
        
        # Get text embeddings
        tokens = self.pipeline.tokenizer(
                        num_images*[prompt],
                        padding="max_length",
                        max_length=self.pipeline.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    ).to(self.device)
        
        protected_tokens = self.pipeline.tokenizer(
                                num_images*[protected_prompt],
                                padding="max_length",
                                max_length=self.pipeline.tokenizer.model_max_length,
                                truncation=True,
                                return_tensors="pt"
                            ).to(self.device)
        
        if negative_prompt is None:
            negative_prompt = ""
            
        text_input_neg = self.pipeline.tokenizer(
            num_images*[negative_prompt],
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            text_embeddings = self.pipeline.text_encoder(tokens.input_ids)[0]
            protected_embeddings = self.pipeline.text_encoder(protected_tokens.input_ids)[0]
            negative_embeds = self.pipeline.text_encoder(text_input_neg.input_ids.to(self.device))[0]
        
        
        # Prepare latents
        latents = torch.randn(
            (num_images, self.pipeline.unet.in_channels, height // 8, width // 8),
            device=self.device,
            dtype = self.pipeline.unet.dtype
        )
        
        self.pipeline.scheduler.set_timesteps(num_inference_steps, device=self.device)
        
        
        images = []
        # for _ in range(num_images):
            
        latents = latents * self.pipeline.scheduler.init_noise_sigma
        current_latents=latents.clone()
        
        for t in self.pipeline.progress_bar(self.pipeline.scheduler.timesteps):
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([current_latents] * 2)   
            latent_model_input = self.pipeline.scheduler.scale_model_input(latent_model_input, t)
            
            # Get mixed text embeddings
            mix_embeddings = torch.cat([negative_embeds, 
                                    (1 - mixing_weight) * text_embeddings + mixing_weight * protected_embeddings])
            
            
            with torch.no_grad():
                noise_pred = self.pipeline.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=mix_embeddings
                ).sample
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous noisy sample
            current_latents = self.pipeline.scheduler.step(noise_pred, t, current_latents).prev_sample
            
        # Decode latents to image
        with torch.no_grad():
            images = self.pipeline.decode_latents(current_latents)
            
        # Convert to PIL Image
        images = self.pipeline.numpy_to_pil(images)
        # images.append(image)

        return images

class AdaptiveProtectedFlux:
    def __init__(self,
                 model_id: str = "black-forest-labs/FLUX.1-dev",
                 ckpt_4bit_id: str = "sayakpaul/flux.1-dev-nf4-pkg",
                 device:str = "cuda",
                 dtype:torch.dtype= torch.bfloat16
                ):
        self.ckpt_4bit_id = ckpt_4bit_id
        self.model_id = model_id
        self.device = device

        self.pipeline = FluxPipeline.from_pretrained(
                self.model_id,
                cache_dir="./cache_dir",
                torch_dtype=torch.float16,
                token=os.getenv("HF"),
        )
        self.pipeline.enable_model_cpu_offload()
        self.pipeline.enable_vae_tiling()

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)
        
    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2) # (1, 16, 32, 2, 32, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5) # (1, 32, 32, 16, 2, 2)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4) # (1, 1024, 64)
        # print(f"Latents shape : {latents.shape}")
        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2) # (1, 32, 32, 16, 2, ,2)
        latents = latents.permute(0, 3, 1, 4, 2, 5) # (1, 16, 32, 2, 32, 2)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width) # (1, 16, 64, 64)

        return latents

    def calculate_shift(
            self,
            image_seq_len,
            base_seq_len: int = 256,
            max_seq_len: int = 4096,
            base_shift: float = 0.5,
            max_shift: float = 1.16,
        ):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu
    
    def generate(self,
                 prompt:str,
                 protected_prompt: str,
                 prompt2: Optional[str] = None,
                 negative_prompt: str = "",
                 negative_prompt2: Optional[str] = None,
                 true_cfg_scale: float = 2.5,
                 height: Optional[int] = 1024,
                 width: Optional[int] = 1024,
                 num_inference_steps: int = 50,
                 guidance_scale: float = 8,
                 num_images: Optional[int] = 1,
                 mixing_weight: float = 0.5):

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        # print(height, width)
        
        do_true_cfg = true_cfg_scale > 1 and negative_prompt is not None

        # Encode the prompts and get the embeddings
        # Tokenize prompt
        tokens = self.pipeline.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Tokenize prompt2
        if prompt2 is None:
            prompt2 = prompt
            
        tokens2 = self.pipeline.tokenizer_2(
            [prompt2],
            padding="max_length",
            max_length=self.pipeline.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Tokenize protected_prompt for both encoders
        protected_tokens = self.pipeline.tokenizer(
            [protected_prompt],
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        protected_tokens_2 = self.pipeline.tokenizer_2(
            [protected_prompt],
            padding="max_length",
            max_length=self.pipeline.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            # We only use the pooled prompt output from the CLIPTextModel
            pooled_prompt_embeds = self.pipeline.text_encoder((tokens.input_ids).to(self.device), output_hidden_states=False)
            pooled_prompt_embeds = pooled_prompt_embeds.pooler_output
            pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=self.pipeline.text_encoder.dtype, device = self.device)
            
            # Prompt Embeddings from T5 Tokenizer
            prompt_embeds = self.pipeline.text_encoder_2((tokens2.input_ids).to(self.device), output_hidden_states=False)[0]
            prompt_embeds = prompt_embeds.to(dtype=self.pipeline.text_encoder.dtype, device=self.device)
            
            # Protected Prompt Embeddings from CLIPTextModel
            protected_pooled_prompt_embeds = self.pipeline.text_encoder((protected_tokens.input_ids).to(self.device), output_hidden_states=False) 
            protected_pooled_prompt_embeds = protected_pooled_prompt_embeds.pooler_output
            protected_pooled_prompt_embeds = protected_pooled_prompt_embeds.to(dtype=self.pipeline.text_encoder.dtype, device=self.device)
            
            # Protected Prompt Embeddings from T5 Tokenizer
            protected_prompt_embeds = self.pipeline.text_encoder_2((protected_tokens_2.input_ids).to(self.device), output_hidden_states=False)[0]
            protected_prompt_embeds = protected_prompt_embeds.to(dtype=self.pipeline.text_encoder.dtype, device=self.device)
        
        text_ids = torch.zeros(prompt_embeds.shape[1], 3, dtype=self.pipeline.text_encoder.dtype, device = pooled_prompt_embeds.device)
        
        if do_true_cfg:
            neg_tokens = self.pipeline.tokenizer(
                [negative_prompt],
                padding="max_length",
                max_length=self.pipeline.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
                
            if negative_prompt2 is None:
                negative_prompt2 = negative_prompt
                
            neg_tokens2 = self.pipeline.tokenizer_2(
                [negative_prompt2],
                padding="max_length",
                max_length=self.pipeline.tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                # We only use the pooled prompt output from the CLIPTextModel
                pooled_neg_prompt_embeds = self.pipeline.text_encoder((neg_tokens.input_ids).to(self.device), output_hidden_states=False)
                pooled_neg_prompt_embeds = pooled_neg_prompt_embeds.pooler_output
                pooled_neg_prompt_embeds = pooled_neg_prompt_embeds.to(dtype=self.pipeline.text_encoder.dtype, device=self.device)
                
                # Negative Prompt Embeddings from T5 Tokenizer
                neg_prompt_embeds = self.pipeline.text_encoder_2((neg_tokens2.input_ids).to(self.device), output_hidden_states=False)[0]
                neg_prompt_embeds = neg_prompt_embeds.to(dtype=self.pipeline.text_encoder.dtype, device=self.device)

        # Prepare the latents
        num_channels_latents = self.pipeline.transformer.config.in_channels // 4
        height = 2 * (int(height) // (self.pipeline.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.pipeline.vae_scale_factor * 2))
        # print(height, width)

        with torch.no_grad():
            latents = torch.randn(
                (1*num_images, num_channels_latents, height , width ), # 1, 16, height // 8, width // 8
                device=pooled_prompt_embeds.device,
                dtype=prompt_embeds.dtype,
            )
            latents = self._pack_latents(latents, 1*num_images, num_channels_latents, height, width)
            latent_image_ids = self._prepare_latent_image_ids(1*num_images, height // 2, width // 2, pooled_prompt_embeds.device, prompt_embeds.dtype)
        
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = self.calculate_shift(
            image_seq_len,
            self.pipeline.scheduler.config.base_image_seq_len,
            self.pipeline.scheduler.config.max_image_seq_len,
            self.pipeline.scheduler.config.base_shift,
            self.pipeline.scheduler.config.max_shift,
        )
        if sigmas is None:
            self.pipeline.scheduler.set_timesteps(num_inference_steps, device=self.pipeline.transformer.device)
        else:
            self.pipeline.scheduler.set_timesteps(sigmas = sigmas, mu = mu, device=self.pipeline.transformer.device)
        
        
        # handle guidance
        with torch.no_grad():
            if self.pipeline.transformer.config.guidance_embeds:
                guidance = torch.full([1], guidance_scale, device=pooled_prompt_embeds.device, dtype=prompt_embeds.dtype)
                guidance = guidance.expand(latents.shape[0])
            else:
                guidance = None

        # Mixed embeddings
        pooled_mix_embeds = (1 - mixing_weight) * pooled_prompt_embeds + mixing_weight * protected_pooled_prompt_embeds
        mix_embeds = (1 - mixing_weight) * prompt_embeds + mixing_weight * protected_prompt_embeds

        current_latents = latents
        # last_noise_pred = None
        for t in self.pipeline.progress_bar(self.pipeline.scheduler.timesteps):
            timestep = t.expand(current_latents.shape[0]).to(current_latents.dtype)
            
            with torch.no_grad():
                # breakpoint()
                noise_pred = self.pipeline.transformer(
                    hidden_states=current_latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_mix_embeds,
                    encoder_hidden_states=mix_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
            """ Classifier free guidance """  
            if do_true_cfg:
                with torch.no_grad():
                    neg_noise_pred = self.pipeline.transformer(
                        hidden_states=current_latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_neg_prompt_embeds,
                        encoder_hidden_states=neg_prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        return_dict=False,
                    )[0]
                
                noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
            
            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = current_latents.dtype
            current_latents = self.pipeline.scheduler.step(noise_pred, t, current_latents, return_dict=False)[0]
        
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_vae_tiling()
        self.pipeline.vae = self.pipeline.vae.to(dtype=torch.float32)
        
        current_latents = self._unpack_latents(current_latents, height, width, self.pipeline.vae_scale_factor)
        current_latents = (current_latents / self.pipeline.vae.config.scaling_factor) + self.pipeline.vae.config.shift_factor
        
        with torch.no_grad():
            images = self.pipeline.vae.decode(
                current_latents.to(dtype=torch.float32),
                return_dict=False
            )[0]
        
        # Convert to image
        images = self.pipeline.image_processor.postprocess(images, output_type="pil")
            
        return images


def test(user_prompt:str = "", protected_prompt:str = "", negative_prompt:str = "", model_id:str = "SDXL", mixing_wt:float = 0.5, guidance_scale:float = 7.5, num_imgs=4, output_dir: str = "", device:str = "cuda", idx:int=0):

    device = device
    if negative_prompt == "":
        output_dir = f"{output_dir}/mixing_wt{mixing_wt}/protected_{guidance_scale}/prompt{idx:04}"
    else:
        output_dir = f"{output_dir}/negative_prompting/mixing_wt{mixing_wt}/protected_{guidance_scale}/prompt{idx:04}"
        
    if os.path.exists(f"{output_dir}"):
        png_files = glob.glob(os.path.join(output_dir, '*.png'))
        if len(png_files) == num_imgs:
            return
    
    model_class_name, *model_args = config.MODEL_IDs[model_id].values()
    model_class = globals()[model_class_name]
    model_id_arg = model_args[0] if model_args else "stabilityai/stable-diffusion-2-1"
    
    protected_sd = model_class(model_id=model_id_arg, device=device)
    # print(f"model_class: {model_class_name} \n dir: {output_dir} \n model_id: {model_id_arg}")
    a = num_imgs
    b = 0
    os.makedirs(output_dir, exist_ok=True)
    if num_imgs > 4:
        a = num_imgs//2
        if num_imgs % 2:
            b = num_imgs//2 + 1
        b = a
    images = protected_sd.generate(prompt=user_prompt, protected_prompt=protected_prompt, negative_prompt=negative_prompt, mixing_weight=mixing_wt, guidance_scale=guidance_scale, num_images=a)


    for i, img in enumerate(images):
        output_path = f"{output_dir}/img{i:03}.png"
        img.save(output_path)
        print(f"Saved image to: {output_path}")
    if b > 0:
        images = protected_sd.generate(prompt=user_prompt, protected_prompt=protected_prompt, negative_prompt=negative_prompt, mixing_weight=mixing_wt, guidance_scale=guidance_scale, num_images=b)


        for i, img in enumerate(images):
            output_path = f"{output_dir}/img{i+a:03}.png"
            img.save(output_path)
            print(f"Saved image to: {output_path}")
    torch.cuda.empty_cache()
    return
        