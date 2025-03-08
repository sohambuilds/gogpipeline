import argparse
from tqdm import tqdm
from detection.detection_manager import detect_protected_concepts
from detection.concepts_manager import load_protected_concepts, load_test_prompts
from rewriting.rewrite_agent import rewrite_prompt
from Models.CFG import test
import Models.models as models
import config
import torch
import warnings
import os
import json
import wandb
import time
from torch.cuda import get_device_properties
warnings.filterwarnings("ignore")
# wandb.login(key=os.environ.get("WANDB_API"))

def main():

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    
    parser = argparse.ArgumentParser(description="Copyright-protected concept detection and rewriting.")
    # parser.add_argument("--prompt", type=str, required=True, help="User's image prompt")
    parser.add_argument("--concepts_json", type=str, default="data/protected_concepts.json", help="Path to the JSON file with protected concepts.")
    parser.add_argument("--test_prompts", type=str, default="data/test_prompts.py", help="Path to the test prompts file")
    parser.add_argument("--protected_dataset", type=str, default="data/protected_dataset.json", help="Path to the protected dataset file")
    parser.add_argument("--wt", type=float, default=0.5, help="Weight for mixing the embeddings")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="how closely the AI-generated image matches the text prompt")
    parser.add_argument("--model", type=str, default="SDXL", help="Model to choose from SD21, SDXL, Flux, SDXT, SD35L")
    parser.add_argument("--protect", action="store_true", help="Protect the concepts or not [boolean]")
    parser.add_argument("--no-protect", action="store_false", help="Protect the concepts or not [boolean]")
    parser.add_argument("--num_img", type=int, default = 4, help="Number of images to be generated")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the generated images")
    
    args = parser.parse_args()

    
    # Retrieve the Arguments
    concepts_json = args.concepts_json
    mixing_wt = args.wt
    guidance_scale=args.guidance_scale
    model_id = args.model
    protect= args.protect
    protected_dataset_path = args.protected_dataset
    test_prompts_path = args.test_prompts
    test_prompts = load_test_prompts(test_prompts_path)
    print(len(test_prompts))
    num_imgs = args.num_img
    output_dir = args.output_dir

    # Load the protected dataset
    if os.path.exists(protected_dataset_path):
        with open(protected_dataset_path, "r") as f:
            protected_dataset = json.load(f)

    # Check if the model_id is valid
    assert model_id in config.MODEL_IDs, f"Invalid model_id: {model_id}"
    
    # # Initialize wandb run
    # if protect:
    #     run = wandb.init(
    #         project="Concept_Protection",
    #         config={
    #             **vars(args),  # Log all command line arguments
    #             "device": device,
    #             "similarity_threshold": config.SIMILARITY_THRESHOLD,
    #             "embed_model": config.EMBED_MODEL,
    #             "chat_model": config.CHAT_MODEL,
    #         },
    #         name=f"{model_id}_wt{mixing_wt}",
    #         tags=["Concept_Protection", model_id]
    #     )
    
    # Print config and arguments
    print("-"*50)
    print("CONFIG AND ARGIMENTS")
    print("-"*50)
    print(f"Model ID: {model_id}")
    print(f"Mixing Weight: {mixing_wt}")
    print(f"Concept Protection (True/False): {protect}")
    print(f"Output Directory: {output_dir}")
    print(f"Similarity Threshold: {config.SIMILARITY_THRESHOLD}")
    print(f"Embedding Model: {config.EMBED_MODEL}")
    print(f"Chat Model: {config.CHAT_MODEL}")
    print(f"Guidance Scale: {guidance_scale}")
    print("-"*50)
    

    targets = ['Wonder Woman', 'Shrek', 'Elsa', 'Buzz Lightyear', 
               'Spiderman', 'Mario', 'R2D2', 'Pikachu', 'Iron Man', 
               'Batman', 'Minions', 'Elon Musk', 'Keanu Reeves', 'Beyoncé', 'Chris Hemsworth'
               'Meryl Streep', 'Emma Stone', 'Dwayne Johnson', 'Taylor Swift', 
               'Leonardo DiCaprio', 'Tesla', 'Starbucks', 'Nike', 'McDonald\'s', 
               'Coca-Cola', 'Apple', 'LEGO', 'BMW', 'The Starry Night', 'The Last Supper', 
               'Mona Lisa', 'Creation of Adam', 'Raft of the Medusa', 'Girl with a Pearl Earring']

    targets_indirect = ['Wonder Woman', 'Shrek', 'Elsa', 'Buzz Lightyear', 
                       'Spiderman', 'Mario', 'R2D2', 'Pikachu', 'Iron Man', 
                       'Batman', 'Minions']
    
    # for idx, user_prompt in enumerate(tqdm(test_prompts)):
        
    for idx, (user_prompt, protected_prompt) in enumerate(tqdm(protected_dataset.items())):
        
        # if idx > 3:
        #     t_idx = (idx + 1)//3
        # else:
        #     t_idx = idx // 3
            
        start_time = time.time()
        if protect:
            if protected_prompt == "NA":
                print(f"Generating from User's prompt: {user_prompt}")
                # models.test(user_prompt, targets[t_idx], model_id, guidance_scale, output_dir, device, idx)
                models.test(user_prompt, "", model_id, guidance_scale, num_imgs, output_dir, device, idx)
            else:
                print(f"Generating from Protected prompt: {protected_prompt}")
                test(user_prompt, protected_prompt, "", model_id, mixing_wt, guidance_scale, num_imgs, output_dir, device, idx)
                # test(user_prompt, protected_prompt, targets[t_idx], model_id, mixing_wt, guidance_scale, output_dir, device, idx)
        else:
            print(f"Generating from User's prompt: {user_prompt}")
            models.test(user_prompt, "", model_id, guidance_scale, num_imgs, output_dir, device, idx)
            # models.test(user_prompt, targets[t_idx], model_id, guidance_scale, output_dir, device, idx)
            
        # Calculate metrics
        end_time = time.time()
        generation_time = end_time - start_time
        progress = (idx + 1) / len(protected_dataset) * 100
        
        print(f"Generation Time: {generation_time:.2f}s | Prompt Index: {idx} | Progress: {progress:.2f}%")
        
        # Log metrics to wandb
        # if protect:
        #     run.log({
        #         "protected_dataset": {
        #             "generation_time": generation_time,
        #             "prompt_idx": idx,
        #             "progress": progress,
        #             "prompt_info": {
        #                 "original": user_prompt,
        #                 "protected": protected_prompt
        #             }
        #         }
        #     })
    
    # Finish the wandb run
    # if protect:
    #     run.finish()

if __name__ == "__main__":
    main()
