import openai
from config import OPENAI_API_KEY, CHAT_MODEL
from detection.detection_manager import detect_protected_concepts

client = openai.OpenAI(api_key=OPENAI_API_KEY)

def rewrite_prompt(user_prompt: str, detected_concepts: list, concepts_json: str = "data/protected_concepts.json") -> str:
    """
    Uses an LLM to rewrite the user's prompt to remove/abstract references
    to the detected concepts (including signature traits).
    """
    if not detected_concepts:
        return user_prompt

    
    concepts_str = ", ".join(detected_concepts)
    system_message = {
        "role": "system",
        # "content": (
        #     "You are a rewriting assistant. Remove all direct or indirect references "
        #     "to the given protected concepts (including their signature traits). "
        #     "Replace them with generic equivalents. Maintain overall style and context. "
        #     "Do not include references to spider webs, spider powers, or any iconic elements "
        #     "of these protected concepts."
        # )
        # "content":(
        #     "You are an advanced text transformation AI. Your task is to carefully rewrite content by:"
        #     "1. Identifying and eliminating all direct or indirect mentions, allusions, and characteristic features descriptions of specified protected content."
        #     "2. Substituting removed elements with suitable generic alternatives."
        #     "3. Preserving the original writing style, tone, and overall message."
        #     "4. Do not include references to spider webs, spider powers, or any iconic elements"
        #     "5. Make sure the similarity between the user's prompt and rewritten prompt which is in the range 0 to 1 needs to be less than 0.6"
        #     "Strive for a seamless transformation that retains the essence of the original text while adhering to these guidelines."
        #     "Your goal is to create a version that is both true to the source material and free from any traces of the specified protected content."
        # )
        "content":(
            "You are a strict rewriting assistant. Your task is to carefully rewrite content by follwing these rules:"
        
            "1. Identify and eliminate all direct or indirect mentions, and characteristic features descriptions of specified protected content (including signature traits).Even if phrases or words do not directly reference copyrighted concepts, identify and replace combinations of descriptors that together describe a widely recognized character or iconic imagery. When certain features, such as specific suits, emblems, headpieces, and accessories, are paired together in a way that closely resembles a known character (e.g., Batman-like elements), replace them with distinct alternatives that completely avoid the risk of generating protected content."
            "2. Rewrite prompts with different COLOR combinations, HAIR STYLES, CLOTHING STYLES, BODY structure and VISUAL characteristics compared to user's prompt."
            "3. Substituting removed elements with suitable generic alternatives. Ensure these alternatives are new and very distinct from the original."
            "4. Do not include references to spider webs, spider powers, bat emblem, cowl hoods, frozen or any iconic and visual elements related to protected characters."
            "5. Try to include all ethnic backgrounds."
            "6. Ensure that character descriptions do not maintain the same fundamental silhouette, presence, or function as a protected figure"
            # "Strive for a seamless transformation that retains the essence of the original text while adhering to these guidelines."
            # "Your goal is to create a version that is both true to the source material and free from any traces of the specified protected content."
        )
    }

    user_message = {
        "role": "user",
        "content": (
            f"Rewrite the following prompt to remove or generalize any references to "
            f"'{concepts_str}' or related traits (like web-slinging, costumes, iconic colors, oval shaped body etc.)."
            
            f"Original prompt: {user_prompt}\n"
        )
        # f"Use a neutral, generic alternative and don't repeat the same phrases. Keep other context.\n\n"
    }

    try:
     
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[system_message, user_message],
            temperature=0.7
        )
        rewritten_prompt = response.choices[0].message.content.strip()
        
        # ========== OPTIONAL: 2nd pass detection to confirm references are gone ==========
        
        still_matched = detect_protected_concepts(rewritten_prompt, concepts_json)
        
        # If the second pass still sees the same concept, try a second rewrite or do a fallback
        intersection = set(detected_concepts) & set(still_matched)
        # print(rewritten_prompt)
        # print("Going for second rewritting")
        # print(list(intersection))
        if intersection:
            print("Going for second rewritting")
            print(list(intersection))
            second_system_msg = {
                "role": "system",
                "content": (
                    "You are a strict rewriting assistant. The user wants absolutely no references "
                    "to the protected concepts (including signature traits). Rewrite the prompt with different COLOR theme combinations, Acessories on the body HAIR STYLES, CLOTHING STYLES, BODY structure and VISUAL characteristics compared to user's prompt."
                    "and remove or replace any remaining references or hints."
                )
            }
            
            second_user_msg = {
                "role": "user",
                "content": (
                    f"The current prompt still references {list(intersection)}."
                    f"Replace any suble references or redefine the whole characterstics with their looks\n\n"
                    # f"Please remove or replace any subtle references or signature traits.\n\n"
                    f"Current prompt:\n{rewritten_prompt}\n\n"
                    "Rewritten prompt:\n"
                )
                
            }

            try:
                second_response = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[second_system_msg, second_user_msg],
                    temperature=0.7
                )
                rewritten_prompt = second_response.choices[0].message.content.strip()
            except Exception as e:
                print(f"[ERROR in second rewrite pass] {e}")

        return rewritten_prompt
    except Exception as e:
        print(f"[ERROR in rewriting] {e}")

        return user_prompt
