# Guardians of Generation

**Guardians of Generation: Dynamic Inference-Time Copyright Shielding with Adaptive Guidance for AI Image Generation**

This repository implements an inference-time pipeline that safeguards AI-generated images against copyright infringement. The approach detects protected concepts in user prompts, rewrites them using a large language model (LLM) if necessary, and then generates images using adaptive classifier-free guidance that blends the original and rewritten prompts.

## Repository Structure

- **data/**  
  Contains JSON files (e.g., `protected_concepts.json`) and test prompt scripts.

- **detection/**

  - `concepts_manager.py`: Loads the protected concepts JSON file.
  - `detection_manager.py`: Implements concept detection and LLM-based disambiguation.
  - `embed_utils.py`: Provides functions for computing text embeddings and cosine similarity.

- **rewriting/**

  - `rewrite_agent.py`: Contains the LLM-based prompt rewriting logic to sanitize inputs.

- **Models/**

  - `CFG.py`: Implements adaptive protected generation pipelines (e.g., for SDXL, SD21, Flux, etc.) that blend original and rewritten prompt embeddings.
  - `models.py`: Implements standard (unprotected) generation pipelines.

- **config.py**  
  Global configuration for API keys, model identifiers, thresholds, and other parameters.

- **main.py**  
  End-to-end integration script that detects protected concepts, rewrites prompts if necessary, and generates images.

- **main2.py**  
  An alternative entry point for testing and experiments.

## Installation

1. **Clone the Repository**

   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Set Up a Python Environment**
   It is recommended to use Python 3.8 or higher.

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows
   ```

3. **Install Dependencies**
   Install the required packages (ensure that `torch`, `diffusers`, `transformers`, etc. are included):

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Environment Variables**
   Export your OpenAI API key and any other necessary credentials:
   ```bash
   export OPENAI_API_KEY="your_openai_api_key"
   ```

## Usage

### Running the Full Pipeline

The `main_integration.py` script integrates detection, rewriting, and image generation into one end-to-end flow.

#### Protected Generation

This mode detects protected concepts in your prompt and rewrites them before generation using Adaptive CFG.

```bash
python main.py \
  --prompt "A detailed portrait of Spider-Man in a futuristic cityscape" \
  --protect \
  --mixing_wt 0.6 \
  --model SDXL \
  --output_dir outputs \
  --num_imgs 3
```

#### Unprotected Generation

This mode directly uses your prompt without rewriting and CFG.

```bash
python main.py \
  --prompt "A detailed portrait of Spider-Man in a futuristic cityscape" \
  --model SDXL \
  --output_dir outputs \
  --num_imgs 3
```

### Command-Line Arguments

- `--prompt`: Text prompt for image generation (required).
- `--negative_prompt`: (Optional) A negative prompt to steer generation.
- `--concepts_json`: Path to the protected concepts JSON file (default: `data/protected_concepts.json`).
- `--output_dir`: Directory where generated images will be saved (default: `outputs`).
- `--model`: Model to use (options: `SD21`, `SDXL`, `Flux`, `SDXLT`, `SD35L`; default: `SDXL`).
- `--guidance_scale`: Classifier-free guidance scale (default: 7.5).
- `--num_inference_steps`: Number of denoising steps (default: 50).
- `--num_imgs`: Number of images to generate (default: 4).
- `--device`: Device to run on (e.g., `cuda` or `cpu`; default: `cuda`).
- `--mixing_wt`: Mixing weight for blending original and rewritten prompt embeddings (default: 0.5).
- `--protect`: Flag to enable the adaptive protection pipeline.

## Customization

- **Prompt Rewriting:**  
  Modify the rewriting instructions in `rewriting/rewrite_agent.py` to adjust how the prompt is sanitized.

- **Detection Threshold:**  
  Adjust `SIMILARITY_THRESHOLD` in `config.py` to control sensitivity to protected concepts.

- **Model Settings:**  
  Update model identifiers and configuration in `config.py` to experiment with different diffusion models.

## Acknowledgments

This repository accompanies the paper:

> _Guardians of Generation: Dynamic Inference-Time Copyright Shielding with Adaptive Guidance for AI Image Generation_

For a detailed explanation of the methodology and experimental results, please refer to the paper.

## License
