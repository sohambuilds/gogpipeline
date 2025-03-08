import argparse
from tqdm import tqdm
from detection.detection_manager import detect_protected_concepts
from detection.concepts_manager import load_protected_concepts, load_test_prompts
from rewriting.rewrite_agent import rewrite_prompt
# from Models.CFG import test
# import Models.models as models
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
    
    dataset = {}
    concepts_json = 'data2/dataset.json'
    test_prompts_path = 'data2/indirect_prompts.py'
    dataset_dir = 'data2/protected_indirect_dataset_v2.json'
    test_prompts = load_test_prompts(test_prompts_path)


    # Initialize wandb run
    # run = wandb.init(
    #     project="Concept_Protection",
    #     name="Prep-Dataset-Indirect",
    # )
    time_taken = 0
    detected_time = 0
    for idx, user_prompt in enumerate(tqdm(test_prompts)):
        start_time = time.time()
        detected = detect_protected_concepts(user_prompt, concepts_json)
        detected_time += time.time() - start_time
        if detected:
            # print(f"Detected protected concepts in prompt: {user_prompt}")
            protected_prompt = rewrite_prompt(user_prompt, detected, concepts_json=concepts_json)
            # run.log({"User_Prompt": user_prompt, "Protected_Prompt": protected_prompt})
            print(f"Rewritten prompt: {protected_prompt}")
            dataset[user_prompt] = protected_prompt
        else:
            print("No protected concepts detected.")
            # run.log({"User_Prompt": user_prompt, "Protected_Prompt": "NA"})
            dataset[user_prompt] = user_prompt
            
        end_time = time.time()
        print(f"Time taken: {end_time - start_time}secs")
        time_taken += end_time - start_time

    print(f"Avg Execution time: {time_taken / len(test_prompts)} secs")
    print(f"Avg Detection time: {detected_time/len(test_prompts)} secs")
    
    # with open(dataset_dir, 'w') as f:
    #     json.dump(dataset, f, indent=4)
    # print("Prep Completed")
    

if __name__ == "__main__":
    main()
