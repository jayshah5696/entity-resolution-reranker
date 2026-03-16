import pytest
from src.data.serialize import colval_serialize, colval_pair, pipe_serialize

def test_colval_serialize_basic():
    record = {
        "first_name": "Jay",
        "last_name": "Shah",
        "company": "Google Inc",
        "title": "Software Engineer",
        "country": "USA"
    }
    expected = "COL fn VAL Jay COL ln VAL Shah COL org VAL Google Inc COL title VAL Software Engineer COL country VAL USA"
    assert colval_serialize(record) == expected

def test_colval_serialize_missing_fields():
    # title is missing
    record = {
        "first_name": "Jay",
        "last_name": "Shah",
        "company": "Google Inc",
        "country": "USA"
    }
    expected = "COL fn VAL Jay COL ln VAL Shah COL org VAL Google Inc COL country VAL USA"
    assert colval_serialize(record) == expected
    assert "title" not in colval_serialize(record)

def test_colval_serialize_empty_and_whitespace():
    record = {
        "first_name": "Jay",
        "last_name": "",
        "company": "  ",
        "title": None,
        "country": "USA"
    }
    expected = "COL fn VAL Jay COL country VAL USA"
    assert colval_serialize(record) == expected

def test_colval_serialize_field_order():
    # Order should be fn, ln, org, title, country regardless of input order
    record = {
        "country": "USA",
        "company": "Google Inc",
        "last_name": "Shah",
        "first_name": "Jay",
        "title": "Engineer"
    }
    expected = "COL fn VAL Jay COL ln VAL Shah COL org VAL Google Inc COL title VAL Engineer COL country VAL USA"
    assert colval_serialize(record) == expected

def test_colval_pair():
    record_a = {"first_name": "Jay", "last_name": "Shah"}
    record_b = {"first_name": "J", "last_name": "Shah"}
    expected = "COL fn VAL Jay COL ln VAL Shah [SEP] COL fn VAL J COL ln VAL Shah"
    assert colval_pair(record_a, record_b) == expected

def test_pipe_serialize_phase1_compat():
    # Phase 1: (first_name, last_name, company, email, country)
    record = {
        "first_name": "Jay",
        "last_name": "Shah",
        "company": "Google Inc",
        "email": "jay@google.com",
        "country": "USA"
    }
    expected = "Jay | Shah | Google Inc | jay@google.com | USA"
    assert pipe_serialize(record) == expected

def test_colval_serialize_special_chars():
    record = {
        "first_name": "Jay-Michael",
        "last_name": "O'Neil",
        "company": "At&t",
    }
    expected = "COL fn VAL Jay-Michael COL ln VAL O'Neil COL org VAL At&t"
    assert colval_serialize(record) == expected

def test_consistency():
    record = {"first_name": "Jay", "last_name": "Shah"}
    assert colval_serialize(record) == colval_serialize(record)
