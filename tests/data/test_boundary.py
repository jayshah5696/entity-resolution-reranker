import pytest
import numpy as np
import polars as pl
from unittest.mock import Mock
from src.data.boundary import (
    find_boundary_candidates,
    parse_llm_labels,
    label_with_llm,
    discard_ambiguous
)

def test_find_boundary_candidates():
    # Mock pool and embeddings
    pool = pl.DataFrame([
        {"entity_id": "1", "first_name": "A"},
        {"entity_id": "2", "first_name": "B"},
        {"entity_id": "3", "first_name": "C"}
    ])
    
    # 3x3 similarity matrix implicitly formed by these embeddings
    # E1 & E2 cosine sim = 0.8
    # E1 & E3 cosine sim = 0.5
    # E2 & E3 cosine sim = 0.95
    # Vectors to approximate this:
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.8, 0.6]) # dot(e1, e2) = 0.8
    # E1.E3 = 0.5 -> e3 = [0.5, sqrt(0.75)]
    e3 = np.array([0.5, np.sqrt(0.75)]) 
    # Let's just mock the distance matrix in the actual test if needed, or use vectors that work out.
    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.8, 0.6, 0.0],
        [0.5, 0.866025, 0.0]
    ])
    # sim(0,1) = 0.8  -> in boundary [0.6, 0.9]
    # sim(0,2) = 0.5  -> below boundary
    # sim(1,2) = 0.4 + 0.519 = 0.919 -> above boundary
    
    candidates = find_boundary_candidates(embeddings, pool, low=0.60, high=0.90, max_pairs=10)
    assert len(candidates) == 1
    # Candidate should be tuple of dicts (record_a, record_b, score)
    a, b, score = candidates[0]
    assert a["entity_id"] == "1"
    assert b["entity_id"] == "2"
    assert 0.79 < score < 0.81

def test_parse_llm_labels():
    response = '```json\n["MATCH", "NON-MATCH", "AMBIGUOUS"]\n```'
    labels = parse_llm_labels(response)
    assert labels == ["MATCH", "NON-MATCH", "AMBIGUOUS"]
    
    malformed = 'MATCH, NON-MATCH'
    assert parse_llm_labels(malformed) == []

def test_label_with_llm():
    mock_client = Mock()
    mock_response = Mock()
    # 2 pairs requested, 2 labels returned
    mock_response.choices = [Mock(message=Mock(content='["MATCH", "NON-MATCH"]'))]
    mock_client.chat.completions.create.return_value = mock_response
    
    pairs = [
        ({"entity_id": "1", "name": "A"}, {"entity_id": "2", "name": "A'"}, 0.8),
        ({"entity_id": "3", "name": "B"}, {"entity_id": "4", "name": "C"}, 0.65)
    ]
    
    labeled = label_with_llm(pairs, mock_client, "model-id", batch_size=2)
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
