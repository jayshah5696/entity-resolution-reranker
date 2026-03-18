import pytest
import numpy as np
from src.eval.metrics import (
    compute_f1_at_threshold,
    compute_pr_curve,
    compute_recall_retention,
    calibrate_threshold
)

def test_compute_f1_at_threshold():
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    labels = np.array([0, 1, 1, 1])
    
    # At thresh 0.5: [0, 0, 1, 1] -> TP=2, FP=0, FN=1 -> F1 = 2*2 / (2*2 + 0 + 1) = 4/5 = 0.8
    f1_05 = compute_f1_at_threshold(scores, labels, 0.5)
    assert np.isclose(f1_05, 0.8)
    
    # At thresh 0.3: [0, 1, 1, 1] -> TP=3, FP=0, FN=0 -> F1 = 1.0
    f1_03 = compute_f1_at_threshold(scores, labels, 0.3)
    assert np.isclose(f1_03, 1.0)

def test_compute_pr_curve():
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
    labels = np.array([1, 1, 0, 1, 0, 0, 1, 0])
    
    pr_curve = compute_pr_curve(scores, labels)
    # pr_curve is list of (threshold, precision, recall)
    # as threshold decreases, recall should generally increase or stay same
    recalls = [r for t, p, r in pr_curve]
    
    for i in range(len(recalls) - 1):
        # because threshold is decreasing (we sweep from high to low or sorted scores)
        # recall should be monotonically increasing
        assert recalls[i] <= recalls[i+1]

def test_compute_recall_retention():
    stage1_ranking = [{"entity_id": "A"}, {"entity_id": "B"}, {"entity_id": "C"}]
    # Stage 2 keeps A in top 2
    stage2_ranking = [{"entity_id": "B"}, {"entity_id": "A"}]
    
    # A is true match
    retention = compute_recall_retention(stage1_ranking, stage2_ranking, "A")
    assert retention == 1.0
    
    # C was in stage 1, but dropped out of stage 2
    retention_drop = compute_recall_retention(stage1_ranking, stage2_ranking, "C")
    assert retention_drop == 0.0

    # D was never in stage 1, so retention is irrelevant but defined as 0.0 or ignored
    # Usually it's 0.0
    retention_never = compute_recall_retention(stage1_ranking, stage2_ranking, "D")
    assert retention_never == 0.0

def test_calibrate_threshold():
    scores = np.array([0.1, 0.3, 0.4, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1, 1])
    
    best_f1, best_t = calibrate_threshold(scores, labels)
    
    # Naive threshold 0.5 gives: [0, 0, 0, 1, 1] -> F1 = 0.8 (TP=2, FN=1)
    # Best threshold should be ~0.4 -> [0, 0, 1, 1, 1] -> F1 = 1.0
    f1_naive = compute_f1_at_threshold(scores, labels, 0.5)
    
    assert best_f1 >= f1_naive
    assert np.isclose(best_f1, 1.0)
    assert best_t <= 0.45

def test_edge_cases():
    # All predictions correct
    scores_perfect = np.array([0.1, 0.9])
    labels_perfect = np.array([0, 1])
    assert compute_f1_at_threshold(scores_perfect, labels_perfect, 0.5) == 1.0
    
    # All predictions wrong
    scores_wrong = np.array([0.9, 0.1])
    labels_wrong = np.array([0, 1])
    assert compute_f1_at_threshold(scores_wrong, labels_wrong, 0.5) == 0.0
    
    # Empty input
    assert compute_f1_at_threshold(np.array([]), np.array([]), 0.5) == 0.0
    
    best_f1, best_t = calibrate_threshold(np.array([]), np.array([]))
    assert best_f1 == 0.0
