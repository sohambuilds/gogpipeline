import openai
import numpy as np
from config import OPENAI_API_KEY, EMBED_MODEL

openai.api_key = OPENAI_API_KEY

def get_text_embedding(text: str) -> np.ndarray:
    """
    Call OpenAI to get an embedding for the given text using EMBED_MODEL.
    """
    response = openai.embeddings.create(
        model=EMBED_MODEL,
        input=[text]  # list of texts here just one text
    )
  
    embedding = response.data[0].embedding
    return np.array(embedding, dtype=np.float32)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot_product / (norm1 * norm2))
