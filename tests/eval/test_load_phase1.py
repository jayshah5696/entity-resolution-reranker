import pytest
from pathlib import Path
import yaml
from src.eval.load_phase1 import (
    load_phase1_index,
    load_phase1_eval_queries,
    load_bm25_index
)

def test_load_phase1_index_reachable():
    cfg_path = Path("configs/eval.yaml")
    if not cfg_path.exists():
        pytest.skip("eval.yaml not found, skipping index test")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    
    index_root = Path(cfg["phase1"]["index_root"])
    if not index_root.exists():
        pytest.skip(f"Phase 1 index root {index_root} not found")
        
    # Test loading - assumes standard LanceDB behavior
    try:
        table, model = load_phase1_index(index_root, "gte_modernbert_base", "cpu")
        assert table is not None
        assert model is not None
    except FileNotFoundError:
        pytest.skip(f"Specific index not found in {index_root}")

def test_load_phase1_eval_queries():
    # If the phase 1 repo doesn't exist locally, we just check schema validation mock
    # since we are creating eval data mock for test cases
    p1_path = Path("../entity-resolution-poc/data/eval")
    if not p1_path.exists():
        pytest.skip("Phase 1 eval dir not found")
        
    queries_df = load_phase1_eval_queries(p1_path)
    assert len(queries_df) > 0
    assert "query_id" in queries_df.columns
    assert "entity_id" in queries_df.columns
    assert "bucket" in queries_df.columns
    assert "query_text_pipe" in queries_df.columns

def test_load_bm25_index():
    bm25_path = Path("../entity-resolution-poc/experiments/002_bm25_baseline/index.json")
    if not bm25_path.exists():
        pytest.skip("BM25 index not found")
        
    bm25_obj = load_bm25_index(bm25_path)
    assert bm25_obj is not None
