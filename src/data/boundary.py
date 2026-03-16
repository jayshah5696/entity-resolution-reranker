import numpy as np
import polars as pl
import json
import time
from typing import Any
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from src.data.serialize import pipe_serialize
from pydantic import BaseModel, Field

class LabelsResponse(BaseModel):
    labels: list[str] = Field(description="A list of strings. Each string must be exactly one of: 'MATCH', 'NON-MATCH', or 'AMBIGUOUS'. The length must match the number of pairs provided.")

def load_phase1_biencoder() -> SentenceTransformer:
    # Loads the GTE model fine-tuned from Phase 1. 
    # For now, we return the base modernbert or just load the HF id.
    # The actual ID specified in the plan is jayshah5696/er-gte-modernbert-base-pipe-ft
    # If not found or access denied, this will throw, but it's correct per plan.
    model = SentenceTransformer("jayshah5696/er-gte-modernbert-base-pipe-ft", trust_remote_code=True)
    return model

def encode_pool(pool: pl.DataFrame, model: SentenceTransformer, batch_size: int = 512) -> np.ndarray:
    records = pool.to_dicts()
    texts = [pipe_serialize(r) for r in records]
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

def find_boundary_candidates(embeddings: np.ndarray, pool: pl.DataFrame, low: float = 0.60, high: float = 0.90, max_pairs: int = 5000) -> list[tuple[dict, dict, float]]:
    # To avoid O(N^2) memory, we process in chunks
    records = pool.to_dicts()
    n = len(records)
    candidates = []
    
    chunk_size = 2000
    for i in tqdm(range(0, n, chunk_size), desc="Finding boundary candidates"):
        end_i = min(i + chunk_size, n)
        chunk_emb = embeddings[i:end_i]
        
        # We compute similarity of this chunk against the entire set
        sims = cosine_similarity(chunk_emb, embeddings)
        
        # Find indices where sim is in [low, high]
        # To avoid duplicates (A,B) and (B,A) and self-matches, we enforce j > global_i
        for local_i in range(end_i - i):
            global_i = i + local_i
            # Find j > global_i
            valid_j = np.where((sims[local_i] >= low) & (sims[local_i] <= high))[0]
            valid_j = valid_j[valid_j > global_i]
            
            for j in valid_j:
                if records[global_i]["entity_id"] != records[j]["entity_id"]:
                    candidates.append((records[global_i], records[j], float(sims[local_i, j])))
                    if len(candidates) >= max_pairs:
                        return candidates
                        
    return candidates

def _build_labeling_prompt(pairs: list[tuple[dict, dict, float]]) -> str:
    prompt = """
You are an expert entity resolution judge. I will give you pairs of profiles.
For each pair, determine if they refer to the exact same real-world person.
Some differences are just typos, nicknames, or missing fields (which is a MATCH).
If they represent different people (e.g. different names with no phonetic relation, or different companies/emails that clearly conflict), it is a NON-MATCH.
If there is not enough information to decide confidently, it is AMBIGUOUS.

Pairs to evaluate:
"""
    for idx, (a, b, score) in enumerate(pairs):
        prompt += f"\nPair {idx+1}:\n"
        prompt += f"A: {a.get('first_name')} {a.get('last_name')} | {a.get('company')} | {a.get('title')} | {a.get('email')}\n"
        prompt += f"B: {b.get('first_name')} {b.get('last_name')} | {b.get('company')} | {b.get('title')} | {b.get('email')}\n"
        
    return prompt.strip()

def label_with_llm(pairs: list[tuple[dict, dict, float]], client: Any, model: str, batch_size: int = 50) -> list[dict]:
    from google import genai
    results = []
    
    for i in tqdm(range(0, len(pairs), batch_size), desc="LLM Labeling"):
        batch = pairs[i:i+batch_size]
        prompt = _build_labeling_prompt(batch)
        
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=LabelsResponse,
                        temperature=0.0,
                    ),
                )
                
                if not response.parsed or not response.parsed.labels:
                    raise ValueError("Failed to parse structured output")
                    
                labels = response.parsed.labels
                
                if len(labels) != len(batch):
                    if attempt < max_retries:
                        time.sleep(1)
                        continue
                    else:
                        print(f"Failed to get correct number of labels for batch starting at {i}")
                        break
                        
                for (a, b, score), label_text in zip(batch, labels):
                    label_int = 1 if label_text == "MATCH" else (0 if label_text == "NON-MATCH" else -1)
                    results.append({
                        "entity_id_a": a["entity_id"],
                        "entity_id_b": b["entity_id"],
                        "record_a": a,
                        "record_b": b,
                        "strategy": "BOUNDARY",
                        "score": score,
                        "label_text": label_text,
                        "label": label_int
                    })
                break
                
            except Exception as e:
                if attempt == max_retries:
                    print(f"Error labeling batch starting at {i}: {e}")
                else:
                    time.sleep(1)
                    
    return results

def discard_ambiguous(labeled_pairs: list[dict]) -> list[dict]:
    return [p for p in labeled_pairs if p["label_text"] != "AMBIGUOUS"]
