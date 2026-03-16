import pytest
import numpy as np
import polars as pl
from unittest.mock import Mock
from src.data.boundary import (
    find_boundary_candidates,
    label_with_llm,
    discard_ambiguous,
    LabelsResponse
)

def test_find_boundary_candidates():
    # Mock pool and embeddings
    pool = pl.DataFrame([
        {"entity_id": "1", "first_name": "A"},
        {"entity_id": "2", "first_name": "B"},
        {"entity_id": "3", "first_name": "C"}
    ])
    
    # 3x3 similarity matrix implicitly formed by these embeddings
    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.8, 0.6, 0.0],
        [0.5, 0.866025, 0.0]
    ])
    
    candidates = find_boundary_candidates(embeddings, pool, low=0.60, high=0.90, max_pairs=10)
    assert len(candidates) == 1
    # Candidate should be tuple of dicts (record_a, record_b, score)
    a, b, score = candidates[0]
    assert a["entity_id"] == "1"
    assert b["entity_id"] == "2"
    assert 0.79 < score < 0.81

def test_label_with_llm():
    mock_client = Mock()
    mock_response = Mock()
    
    # Mock the google-genai Pydantic parsed response
    mock_response.parsed = LabelsResponse(labels=["MATCH", "NON-MATCH"])
    mock_client.models.generate_content.return_value = mock_response
    
    pairs = [
        ({"entity_id": "1", "name": "A"}, {"entity_id": "2", "name": "A'"}, 0.8),
        ({"entity_id": "3", "name": "B"}, {"entity_id": "4", "name": "C"}, 0.65)
    ]
    
    labeled = label_with_llm(pairs, mock_client, "gemini-3.1-flash-lite-preview", batch_size=2)
    assert len(labeled) == 2
    assert labeled[0]["label_text"] == "MATCH"
    assert labeled[1]["label_text"] == "NON-MATCH"
    assert labeled[0]["label"] == 1
    assert labeled[1]["label"] == 0
    assert labeled[0]["strategy"] == "BOUNDARY"

def test_discard_ambiguous():
    labeled_pairs = [
        {"entity_id_a": "1", "label_text": "MATCH"},
        {"entity_id_a": "2", "label_text": "AMBIGUOUS"},
        {"entity_id_a": "3", "label_text": "NON-MATCH"}
    ]
    filtered = discard_ambiguous(labeled_pairs)
    assert len(filtered) == 2
    assert "AMBIGUOUS" not in [p["label_text"] for p in filtered]
