import pytest
import json
from unittest.mock import Mock, patch
from src.data.corrupt_llm import (
    _build_prompt,
    _parse_response,
    generate_nonlatin_corruptions,
    filter_by_ce_score
)

def test_build_prompt():
    record = {
        "first_name": "Wei",
        "last_name": "Chen",
        "ethnicity_group": "chinese",
        "name_script": "cjk"
    }
    prompt = _build_prompt(record)
    assert "chinese" in prompt.lower()
    assert "Wei" in prompt
    assert "Chen" in prompt
    assert "JSON array" in prompt

def test_parse_response_valid():
    response_text = '```json\n[{"variation": "W. Chen", "type": "NL3"}, {"variation": "Chen Wei", "type": "NL2"}]\n```'
    parsed = _parse_response(response_text)
    assert len(parsed) == 2
    assert parsed[0]["variation"] == "W. Chen"
    
def test_parse_response_malformed():
    response_text = 'Here are the corruptions:\n1. W. Chen\n2. Chen Wei'
    parsed = _parse_response(response_text)
    assert parsed == []

def test_generate_nonlatin_corruptions():
    # Setup mock without @patch decorator since we pass client manually
    mock_client = Mock()
    mock_response = Mock()
    
    # Mock the google-genai Pydantic parsed response
    from src.data.corrupt_llm import CorruptionsResponse, VariationObj
    mock_response.parsed = CorruptionsResponse(corruptions=[
        VariationObj(variation="Wei C.", type="NL3"),
        VariationObj(variation="Chen Wei", type="NL2")
    ])
    mock_client.models.generate_content.return_value = mock_response
    
    records = [
        {"entity_id": "1", "first_name": "Wei", "last_name": "Chen", "ethnicity_group": "chinese"}
    ]
    
    # We pass the mocked client directly
    results = generate_nonlatin_corruptions(records, client=mock_client, model="gemini-3.1-flash-lite-preview", batch_size=20)
    
    # Each valid corruption should produce a new pair dict, overwriting first/last
    assert len(results) == 2
    assert results[0]["first_name"] == "Wei" 
    assert results[0]["last_name"] == "C." 
    assert results[0]["corruption_code"] == "NL3"
    
def test_filter_by_ce_score():
    pairs = [
        {"entity_id_a": "1", "entity_id_b": "1", "text_a": "A", "text_b": "B"}, # high score
        {"entity_id_a": "2", "entity_id_b": "2", "text_a": "C", "text_b": "D"}  # low score
    ]
    
    # Mock a CE model
    class MockCE:
        def predict(self, inputs):
            # return high score for first, low for second
            import numpy as np
            return np.array([0.8, 0.2])
            
    filtered = filter_by_ce_score(pairs, MockCE(), min_score=0.35)
    assert len(filtered) == 1
    assert filtered[0]["text_a"] == "A"
