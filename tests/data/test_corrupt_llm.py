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
    response_text = '```json\n["W. Chen", "Chen Wei", "Way Chen"]\n```'
    parsed = _parse_response(response_text)
    assert parsed == ["W. Chen", "Chen Wei", "Way Chen"]
    
def test_parse_response_malformed():
    response_text = 'Here are the corruptions:\n1. W. Chen\n2. Chen Wei'
    parsed = _parse_response(response_text)
    assert parsed == []

@patch("src.data.corrupt_llm.OpenAI")
def test_generate_nonlatin_corruptions(mock_openai_class):
    # Setup mock
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content='["Wei C.", "Chen Wei"]'))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_class.return_value = mock_client
    
    records = [
        {"entity_id": "1", "first_name": "Wei", "last_name": "Chen", "ethnicity_group": "chinese"}
    ]
    
    # We pass the mocked client directly to avoid recreating it
    results = generate_nonlatin_corruptions(records, client=mock_client, model="google/gemini-2.5-flash-lite-preview", batch_size=20)
    
    # Each valid corruption should produce a new pair dict
    assert len(results) == 2
    # The first corruption is "Wei C." so parts are "Wei" and "C."
    assert results[0]["first_name_corrupted"] == "Wei" 
    assert results[0]["last_name_corrupted"] == "C." 
    assert results[0]["corruption_code"].startswith("NL")
    
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
