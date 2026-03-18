import pytest
import numpy as np
from src.models.crossencoder import CrossEncoderReranker

def test_ce_loads_stock_minilm():
    cfg = {"hf_id": "cross-encoder/ms-marco-MiniLM-L-6-v2"}
    ce = CrossEncoderReranker("minilm_reranker", cfg)
    assert ce.model is not None

def test_score_range_is_0_to_1():
    cfg = {"hf_id": "cross-encoder/ms-marco-MiniLM-L-6-v2"}
    ce = CrossEncoderReranker("minilm_reranker", cfg)
    scores = ce.predict([("A", "B"), ("A", "A")])
    assert all(0.0 <= s <= 1.0 for s in scores)

def test_match_scores_higher_than_nonmatch():
    cfg = {"hf_id": "cross-encoder/ms-marco-MiniLM-L-6-v2"}
    ce = CrossEncoderReranker("minilm_reranker", cfg)
    
    match_score = ce.predict([("COL fn VAL Jay COL ln VAL Shah", "COL fn VAL Jay COL ln VAL Shah")])[0]
    nonmatch_score = ce.predict([("COL fn VAL Jay COL ln VAL Shah", "COL fn VAL Bob COL ln VAL Jones")])[0]
    
    assert match_score > nonmatch_score

def test_colval_input_accepted():
    cfg = {"hf_id": "cross-encoder/ms-marco-MiniLM-L-6-v2"}
    ce = CrossEncoderReranker("minilm_reranker", cfg)
    # The predict method accepts standard tuples. We'll pass colval format explicitly
    scores = ce.predict([("COL fn VAL A", "COL fn VAL B")])
    assert len(scores) == 1

def test_batch_and_single_consistent():
    cfg = {"hf_id": "cross-encoder/ms-marco-MiniLM-L-6-v2"}
    ce = CrossEncoderReranker("minilm_reranker", cfg)
    
    pair1 = ("A", "B")
    pair2 = ("C", "D")
    
    s1 = ce.predict([pair1])[0]
    s2 = ce.predict([pair2])[0]
    
    batch_scores = ce.predict([pair1, pair2])
    
    np.testing.assert_allclose([s1, s2], batch_scores, rtol=1e-5)

def test_rerank_returns_sorted_by_score():
    cfg = {"hf_id": "cross-encoder/ms-marco-MiniLM-L-6-v2"}
    ce = CrossEncoderReranker("minilm_reranker", cfg)
    
    query = {"first_name": "Jay", "last_name": "Shah"}
    candidates = [
        {"entity_id": "1", "first_name": "Bob", "last_name": "Jones"},
        {"entity_id": "2", "first_name": "Jay", "last_name": "Shah"}
    ]
    
    reranked = ce.rerank(query, candidates, top_k=2)
    
    assert len(reranked) == 2
    assert reranked[0]["entity_id"] == "2" # High score match should be first
    assert "ce_score" in reranked[0]
    assert reranked[0]["ce_score"] >= reranked[1]["ce_score"]

def test_calibrate_threshold_not_naive_05():
    cfg = {"hf_id": "cross-encoder/ms-marco-MiniLM-L-6-v2"}
    ce = CrossEncoderReranker("minilm_reranker", cfg)
    
    val_scores = np.array([0.1, 0.4, 0.6, 0.9])
    val_labels = np.array([0, 1, 1, 1])
    
    threshold = ce.calibrate_threshold(val_scores, val_labels)
    
    # Naive threshold = 0.5. At 0.5, we predict [0, 0, 1, 1] -> 2 TP, 1 FN, 0 FP -> F1 = 0.8
    # Calibrated threshold = 0.4. At 0.4, we predict [0, 1, 1, 1] -> 3 TP, 0 FN, 0 FP -> F1 = 1.0
    assert threshold < 0.5
