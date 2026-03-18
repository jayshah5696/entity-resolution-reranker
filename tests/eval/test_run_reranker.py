import json
from pathlib import Path
import pytest
from src.eval.run_reranker import process_end_to_end, build_results_json

def test_run_reranker_end_to_end():
    # Mocking end-to-end to verify structure
    results = build_results_json(
        experiment_id="001",
        stage1_latency=10.5,
        stage2_latency=45.2,
        metrics={
            "overall": {
                "recall_at_10": 0.95,
                "recall_at_50": 0.99,
                "mrr": 0.85,
                "ndcg_at_10": 0.88,
                "f1_best": 0.92,
                "f1_threshold": 0.45,
                "recall_retention": 0.99
            },
            "per_bucket": {
                "exact_match": {"recall_at_10": 1.0},
                "typo_name": {"recall_at_10": 0.98},
                "missing_email": {"recall_at_10": 0.90},
                "missing_company": {"recall_at_10": 0.85},
                "missing_email_company": {"recall_at_10": 0.70},
                "severe_corruption": {"recall_at_10": 0.60}
            }
        }
    )
    
    assert results["experiment_id"] == "001"
    assert results["latency"]["stage1_ms"] == 10.5
    assert results["latency"]["stage2_ms"] == 45.2
    assert "overall" in results["metrics"]
    assert "per_bucket" in results["metrics"]
    
    buckets = results["metrics"]["per_bucket"].keys()
    expected_buckets = {
        "exact_match", "typo_name", "missing_email", 
        "missing_company", "missing_email_company", "severe_corruption"
    }
    assert expected_buckets.issubset(set(buckets))
