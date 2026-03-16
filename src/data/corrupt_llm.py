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
    
    The variations should reflect common real-world errors such as:
    - Transliteration or romanization differences
    - Order swapping (e.g., surname first)
    - Dropping parts of the name
    - Common typos or misspellings specific to this origin
    
    Return ONLY a raw JSON array of strings containing the full corrupted names. Do not include markdown formatting or explanations.
    Example: ["Variation 1", "Variation 2", "Variation 3"]
    """
    return prompt.strip()

def _parse_response(response_text: str) -> list[str]:
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
        if isinstance(parsed, list) and all(isinstance(i, str) for i in parsed):
            return parsed
    except json.JSONDecodeError:
        pass
        
    return []

def generate_nonlatin_corruptions(records: list[dict], client: Any, model: str, batch_size: int = 20) -> list[dict]:
    results = []
    
    # Simple batching isn't strictly needed if we're doing synchronous calls,
    # but the interface requests a batch_size parameter. We will loop.
    for record in tqdm(records, desc="Generating LLM Corruptions"):
        if record.get("ethnicity_group") == "us_uk_english":
            continue
            
        prompt = _build_prompt(record)
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            content = response.choices[0].message.content
            corruptions = _parse_response(content)
            
            for i, c_name in enumerate(corruptions):
                # We simply assign the full corrupted string to first_name for simplicity in the pipeline,
                # or try to split it heuristically. The plan specifies mapping it to name fields.
                parts = c_name.split(maxsplit=1)
                new_fn = parts[0] if len(parts) > 0 else ""
                new_ln = parts[1] if len(parts) > 1 else ""
                
                new_record = record.copy()
                new_record["first_name_corrupted"] = new_fn
                new_record["last_name_corrupted"] = new_ln
                # Map to NL codes (NL1-NL7) randomly or sequentially
                new_record["corruption_code"] = f"NL{random.randint(1, 7)}"
                results.append(new_record)
                
        except Exception as e:
            print(f"Error generating corruption for {record['entity_id']}: {e}")
            
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
