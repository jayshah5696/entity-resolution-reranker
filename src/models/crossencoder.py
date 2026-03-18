import numpy as np
import torch
from sentence_transformers import CrossEncoder
from src.data.serialize import colval_serialize
from sklearn.metrics import f1_score

class CrossEncoderReranker:
    def __init__(self, model_key: str, cfg: dict, device: str = None, model_path: str | None = None):
        self.model_key = model_key
        path_to_load = model_path if model_path else cfg.get("hf_id")
        
        # Default device resolution
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
                
        self.model = CrossEncoder(path_to_load, device=device, trust_remote_code=True)
        
    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.array([])
            
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Handle nan bug by casting via sigmoid if needed, but since it's MS MARCO, 
        # it outputs raw logits [-10, 10]. We manually apply sigmoid to squash to [0,1]
        # if the activation function is Identity (as is standard on older models)
        if isinstance(self.model.model, torch.nn.Module):
            # Check if scores are outside [0,1] meaning they are logits
            if len(scores) > 0 and (np.nanmax(scores) > 1.0 or np.nanmin(scores) < 0.0):
                import scipy.special
                scores = scipy.special.expit(scores)
                
        return np.nan_to_num(scores, nan=0.0) # Safety catch for CPU nan bug
        
    def rerank(self, query: dict, candidates: list[dict], top_k: int) -> list[dict]:
        if not candidates:
            return []
            
        q_str = colval_serialize(query)
        pairs = [(q_str, colval_serialize(c)) for c in candidates]
        
        scores = self.predict(pairs)
        
        # Zip candidates with scores
        scored_candidates = []
        for c, s in zip(candidates, scores):
            c_copy = c.copy()
            c_copy["ce_score"] = float(s)
            scored_candidates.append(c_copy)
            
        # Sort descending
        scored_candidates.sort(key=lambda x: x["ce_score"], reverse=True)
        
        return scored_candidates[:top_k]
        
    def calibrate_threshold(self, val_scores: np.ndarray, val_labels: np.ndarray) -> float:
        """
        Sweep threshold to find the F1-maximizing decision boundary.
        """
        best_f1 = 0.0
        best_t = 0.5
        
        # Sweep from 0.01 to 0.99
        thresholds = np.linspace(0.01, 0.99, 99)
        for t in thresholds:
            preds = (val_scores >= t).astype(int)
            f1 = f1_score(val_labels, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
                
        return best_t
