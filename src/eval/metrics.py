import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve
from typing import List, Tuple, Dict, Any

def compute_f1_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    if len(scores) == 0:
        return 0.0
    preds = (scores >= threshold).astype(int)
    return float(f1_score(labels, preds, zero_division=0.0))

def compute_pr_curve(scores: np.ndarray, labels: np.ndarray) -> List[Tuple[float, float, float]]:
    if len(scores) == 0:
        return []
    
    # sklearn precision_recall_curve returns precision, recall, thresholds
    # where thresholds is length N, and precision/recall are length N+1 (last element corresponds to recall=0)
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    
    curve = []
    # We zip up to len(thresholds) to drop the last element which doesn't have a corresponding threshold
    for t, p, r in zip(thresholds, precision[:-1], recall[:-1]):
        curve.append((float(t), float(p), float(r)))
        
    # Sort by threshold descending just to be clean
    curve.sort(key=lambda x: x[0], reverse=True)
    return curve

def compute_recall_retention(stage1_ranking: List[Dict[str, Any]], stage2_ranking: List[Dict[str, Any]], true_id: str) -> float:
    """
    If the true_id was found in stage1_ranking, what is the probability it remains in stage2_ranking?
    This returns 1.0 if it was retained, 0.0 if it was dropped or never present.
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
