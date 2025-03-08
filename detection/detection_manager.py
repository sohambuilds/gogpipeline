import openai
from typing import List
from detection.concepts_manager import load_protected_concepts
from detection.embed_utils import get_text_embedding, cosine_similarity
from config import SIMILARITY_THRESHOLD, OPENAI_API_KEY, CHAT_MODEL

openai.api_key = OPENAI_API_KEY

def disambiguate_reference(user_prompt: str, concept_name: str) -> bool:
    """
    Asks the LLM if the user's prompt truly references 'concept_name'.
    Returns True if yes, False if no.
    """
    system_msg = {
        "role": "system",
        "content": (
            "You are a detection system. "
            "You receive a user prompt and a specific concept name. "
            "Answer ONLY 'YES' or 'NO' depending on whether the user prompt is referring "
            "to that concept in any way (directly or indirectly)."
        )
    }

    user_msg = {
        "role": "user",
        "content": (
            f"User prompt: '{user_prompt}'\n"
            f"Concept name: '{concept_name}'\n\n"
            "Does the user prompt actually refer to the concept name? Please answer YES or NO."
        )
    }

    try:
        response = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[system_msg, user_msg],
            temperature=0.0
        )
        #parse the response We expect "YES" or "NO."
        answer = response.choices[0].message.content.strip().upper()
        return answer.startswith("YES")
    except Exception as e:
        print(f"[ERROR in disambiguation] {e}")
        # if there is no disambiguation then, assume it's not referencing
        return False

def detect_protected_concepts(user_prompt: str, concepts_path: str) -> List[str]:
    """
    Returns a list of validated concepts that the user_prompt references (above threshold and LLM-confirmed).
    """
    data = load_protected_concepts(concepts_path)
    prompt_embedding = get_text_embedding(user_prompt)

    prelim_matches = []  # raw matches by threshold
    for entry in data:
        concept_name = entry["concept"]
        synonyms = entry.get("synonyms", [])

        #Check the concept name
        concept_embedding = get_text_embedding(concept_name)
        sim_score = cosine_similarity(prompt_embedding, concept_embedding)
        if sim_score >= SIMILARITY_THRESHOLD:
            prelim_matches.append(concept_name)
            # print(f"simscore: {sim_score}")
            continue

        # check synonyms
        for syn in synonyms:
            syn_embedding = get_text_embedding(syn)
            sim_score_syn = cosine_similarity(prompt_embedding, syn_embedding)
            if sim_score_syn >= SIMILARITY_THRESHOLD:
                prelim_matches.append(concept_name)
                # print(f"synonyms simscores: {sim_score_syn}")
                break 

    # remove duplicates
    prelim_matches = list(set(prelim_matches))

    validated_matches = []
    for concept in prelim_matches:
        if disambiguate_reference(user_prompt, concept):
            validated_matches.append(concept)

    return validated_matches
