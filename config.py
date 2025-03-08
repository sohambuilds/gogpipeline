import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Embedding model[IMPORTANT TO CHANGE AND exp]
EMBED_MODEL = "text-embedding-ada-002"

CHAT_MODEL = "gpt-4o-mini" #gpt-4  

SIMILARITY_THRESHOLD = 0.8

# MODEL IDs with their class name and model name
MODEL_IDs = {
    "SD21": {
        "class_name": "AdaptiveProtectedStableDiffusion21",
        "model_id": "stabilityai/stable-diffusion-2-1"
        },
    "SDXL": {
        "class_name": "AdaptiveProtectedStableDiffusionXL",
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0"
    },
    "Flux": {
        "class_name": "AdaptiveProtectedFlux",
        "model_id": "black-forest-labs/FLUX.1-dev"
    },
    "SDXLT": {
        "class_name": "AdaptiveProtectedStableDiffusionXL-T",
        "model_id": "stabilityai/sdxl-turbo"
    },
    "SD35L": {
        "class_name": "AdaptiveProtectedStableDiffusion3-5-Large",
        "model_id": "stabilityai/stable-diffusion-3.5-large"
    }
}
