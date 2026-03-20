import math
import numpy as np
from sklearn.metrics import f1_score
from typing import List, Tuple, Dict, Any

# === Phase 1 canonical functions (copy verbatim) ===

def recall_at_k(retrieved_ids: list[str], relevant_id: str, k: int) -> float:
    """1.0 if relevant_id appears in top-k retrieved_ids, else 0.0."""
    if k <= 0:
        return 0.0
    return 1.0 if relevant_id in retrieved_ids[:k] else 0.0

def precision_at_k(retrieved_ids: list[str], relevant_id: str, k: int) -> float:
    """
    Fraction of top-k that are relevant.
    With a single relevant document this equals recall_at_k / k.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    n_relevant = sum(1 for rid in top_k if rid == relevant_id)
    return n_relevant / k

def reciprocal_rank(retrieved_ids: list[str], relevant_id: str) -> float:
    """1/rank if relevant_id is found, else 0.0. Used for MRR computation."""
    for i, rid in enumerate(retrieved_ids):
        if rid == relevant_id:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(retrieved_ids: list[str], relevant_id: str, k: int) -> float:
    """
    NDCG@k for a single relevant document.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    for i, rid in enumerate(top_k):
        if rid == relevant_id:
            # rank is 1-indexed; log base 2 of (rank + 1)
            dcg = 1.0 / math.log2(i + 2)
            idcg = 1.0  # ideal: doc at rank 1 -> 1/log2(2) = 1.0
            return dcg / idcg
    return 0.0

def compute_metrics(
    retrieved_ids: list[str],
    relevant_id: str,
    ks: list[int] | None = None,
) -> dict[str, float]:
    """
    Compute all retrieval metrics for a single query.
    """
    if ks is None:
        ks = [1, 5, 10]

    result: dict[str, float] = {}

    for k in ks:
        result[f"recall_at_{k}"] = recall_at_k(retrieved_ids, relevant_id, k)
        result[f"ndcg_at_{k}"] = ndcg_at_k(retrieved_ids, relevant_id, k)

    result["precision_at_5"] = precision_at_k(retrieved_ids, relevant_id, 5)
    
    mrr_list = retrieved_ids[:10]
    result["mrr_at_10"] = reciprocal_rank(mrr_list, relevant_id)

    return result

def aggregate_metrics(per_query_metrics: list[dict[str, float]]) -> dict[str, float]:
    """
    Compute mean of each metric across all queries.
    """
    if not per_query_metrics:
        return {}

    keys = list(per_query_metrics[0].keys())
    aggregated: dict[str, float] = {}
    for key in keys:
        values = [m[key] for m in per_query_metrics]
        aggregated[key] = float(np.mean(values))
    return aggregated

# === Phase 2 additions ===

def compute_f1_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    if len(scores) == 0:
        return 0.0
    preds = (scores >= threshold).astype(int)
    return float(f1_score(labels, preds, zero_division=0.0))

def compute_recall_retention(stage1_ranking: List[Dict[str, Any]], stage2_ranking: List[Dict[str, Any]], true_id: str) -> float:
    """
    If the true_id was found in stage1_ranking, what is the probability it remains in stage2_ranking?
    """
    in_stage1 = any(r.get("entity_id") == true_id for r in stage1_ranking)
    in_stage2 = any(r.get("entity_id") == true_id for r in stage2_ranking)
    
    if in_stage1 and in_stage2:
        return 1.0
    return 0.0

def calibrate_threshold(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    Returns (best_f1, best_threshold)
    """
    if len(scores) == 0:
        return 0.0, 0.5
        
    best_f1 = 0.0
    best_t = 0.5
    
    thresholds = np.linspace(0.01, 0.99, 99)
    for t in thresholds:
        f1 = compute_f1_at_threshold(scores, labels, t)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
            
    return best_f1, best_t
