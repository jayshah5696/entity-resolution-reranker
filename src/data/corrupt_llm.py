import json
import re
import random
from typing import Any
from tqdm import tqdm
from openai import OpenAI

def _build_prompt(record: dict) -> str:
    fn = record.get('first_name', '')
    ln = record.get('last_name', '')
    eth = record.get('ethnicity_group', 'unknown')
    
    prompt = f"""
    You are an expert in global naming conventions, particularly for {eth} names.
    Generate exactly 3 realistic data entry corruptions or variations for the following name:
    First Name: {fn}
    Last Name: {ln}
    
    Return ONLY a raw JSON array of objects. Each object must have a "variation" string and a "type" code.
    Allowed "type" codes (select the 3 most appropriate):
    - "NL1": Transliteration/Romanization difference
    - "NL2": Order swap (e.g. surname first)
    - "NL3": Dropped middle or part of name
    - "NL4": Common spelling error/typo
    - "NL5": Phonetic substitution
    - "NL6": Honorific/title addition or removal
    - "NL7": Script mixture or encoding issue
    
    Do not include markdown formatting or explanations.
    Example: [{{"variation": "Chen Wei", "type": "NL2"}}, {{"variation": "Way Chen", "type": "NL5"}}]
    """
    return prompt.strip()

def _parse_response(response_text: str) -> list[dict]:
    # Strip markdown block quotes if the model adds them despite instructions
    text = response_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
        
    try:
        parsed = json.loads(text.strip())
        if isinstance(parsed, list) and all(isinstance(i, dict) and "variation" in i and "type" in i for i in parsed):
            return parsed
    except json.JSONDecodeError:
        pass
        
    return []

def generate_nonlatin_corruptions(records: list[dict], client: Any, model: str, batch_size: int = 20) -> list[dict]:
    import time
    results = []
    
    for i in tqdm(range(0, len(records), batch_size), desc="Generating LLM Corruptions"):
        batch = records[i:i+batch_size]
        
        # We process the batch sequentially because openrouter api isn't batched for text generation,
        # but the grouping allows us to simulate batched control flow if needed.
        for record in batch:
            if record.get("ethnicity_group") == "us_uk_english":
                continue
                
            prompt = _build_prompt(record)
            
            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                    )
                    content = response.choices[0].message.content
                    corruptions = _parse_response(content)
                    
                    if not corruptions and attempt < max_retries:
                        continue # retry on malformed
                        
                    for c_obj in corruptions:
                        c_name = c_obj["variation"]
                        c_type = c_obj["type"]
                        
                        # Better naive split
                        parts = c_name.split()
                        if len(parts) == 0:
                            continue
                        elif len(parts) == 1:
                            new_fn = parts[0]
                            new_ln = ""
                        else:
                            new_fn = parts[0]
                            new_ln = " ".join(parts[1:])
                        
                        new_record = record.copy()
                        new_record["first_name"] = new_fn
                        new_record["last_name"] = new_ln
                        new_record["corruption_code"] = c_type
                        results.append(new_record)
                        
                    break # success
                except Exception as e:
                    if attempt == max_retries:
                        print(f"Error generating corruption for {record.get('entity_id')}: {e}")
                    else:
                        time.sleep(1) # short backoff
            
    return results

def filter_by_ce_score(pairs: list[dict], stock_ce: Any, min_score: float = 0.35) -> list[dict]:
    if not pairs:
        return []
        
    # Assume pairs have text_a and text_b for scoring
    inputs = [(p.get("text_a", ""), p.get("text_b", "")) for p in pairs]
    
    try:
        import numpy as np
        scores = stock_ce.predict(inputs)
        filtered = []
        for pair, score in zip(pairs, scores):
            if score >= min_score:
                filtered.append(pair)
        return filtered
    except Exception as e:
        print(f"Error filtering by CE score: {e}")
        return pairs # fallback to returning all if scoring fails
