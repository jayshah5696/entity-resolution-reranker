import pytest
import polars as pl
from src.data.negatives import (
    mine_same_company_diff_person,
    mine_phonetic_neighbor,
    mine_common_name_diff_company,
    mine_title_function_swap,
    mine_title_level_swap,
    mine_random,
    mine_bm25_hard_negatives,
    is_false_negative,
    apply_deterministic_filter
)

@pytest.fixture
def mock_pool():
    return pl.DataFrame([
        {"entity_id": "1", "first_name": "Jay", "last_name": "Shah", "company": "Google", "title": "Senior Engineer", "email": "jay@google.com"},
        {"entity_id": "2", "first_name": "Bob", "last_name": "Smith", "company": "Google", "title": "VP Engineering", "email": "bob@google.com"},
        {"entity_id": "3", "first_name": "Jai", "last_name": "Shah", "company": "Apple", "title": "Senior Engineer", "email": "jai@apple.com"},
        {"entity_id": "4", "first_name": "Bob", "last_name": "Jones", "company": "Microsoft", "title": "Senior Sales", "email": "bob@microsoft.com"},
        {"entity_id": "5", "first_name": "Alice", "last_name": "Smith", "company": "Apple", "title": "VP Sales", "email": "alice@apple.com"},
        {"entity_id": "6", "first_name": "Ali", "last_name": "Jones", "company": "Amazon", "title": "Junior Engineer", "email": "ali@amazon.com"}
    ])

def test_neg1_same_company_diff_person(mock_pool):
    pairs = mine_same_company_diff_person(mock_pool)
    assert len(pairs) > 0
    for p in pairs:
        assert p["record_a"]["company"] == p["record_b"]["company"]
        assert p["record_a"]["entity_id"] != p["record_b"]["entity_id"]

def test_neg2_phonetic_neighbor(mock_pool):
    pairs = mine_phonetic_neighbor(mock_pool)
    assert len(pairs) > 0
    # Jay and Jai
    found = any(p["record_a"]["first_name"] in ["Jay", "Jai"] and p["record_b"]["first_name"] in ["Jay", "Jai"] for p in pairs)
    assert found
    for p in pairs:
        assert p["record_a"]["entity_id"] != p["record_b"]["entity_id"]

def test_neg3_common_name(mock_pool):
    census = pl.DataFrame({"name": ["SMITH", "JONES"], "count": [100, 50]})
    pairs = mine_common_name_diff_company(mock_pool, census, top_n=2)
    assert len(pairs) > 0
    for p in pairs:
        assert p["record_a"]["last_name"] == p["record_b"]["last_name"]
        assert p["record_a"]["company"] != p["record_b"]["company"]
        assert p["record_a"]["entity_id"] != p["record_b"]["entity_id"]

def test_neg4_title_function_swap(mock_pool):
    onet_groups = {"Engineer": ["Senior Engineer", "VP Engineering"], "Sales": ["Senior Sales", "VP Sales"]}
    pairs = mine_title_function_swap(mock_pool, onet_groups)
    assert len(pairs) > 0
    for p in pairs:
        assert p["record_a"]["entity_id"] != p["record_b"]["entity_id"]

def test_neg5_title_level_swap(mock_pool):
    pairs = mine_title_level_swap(mock_pool)
    assert len(pairs) > 0
    for p in pairs:
        assert p["record_a"]["entity_id"] != p["record_b"]["entity_id"]

def test_neg6_random(mock_pool):
    pairs = mine_random(mock_pool)
    assert len(pairs) > 0
    for p in pairs:
        assert p["record_a"]["entity_id"] != p["record_b"]["entity_id"]

def test_neg7_bm25(mock_pool):
    pairs = mine_bm25_hard_negatives(mock_pool)
    assert len(pairs) > 0
    for p in pairs:
        assert p["record_a"]["entity_id"] != p["record_b"]["entity_id"]

def test_is_false_negative():
    a = {"company": "Google", "email": "j@google.com"}
    b = {"company": "Google Inc", "email": "b@google.com"}
    assert is_false_negative(a, b) == True # Domains match

    a2 = {"company": "Microsoft", "email": "a@ms.com"}
    b2 = {"company": "Microsoft Corp", "email": "b@mc.com"}
    assert is_false_negative(a2, b2) == True # Jaro-Winkler > 0.92 and company normalized matches closely
    
    a3 = {"company": "Amazon", "email": "j@gmail.com"}
    b3 = {"company": "Apple", "email": "b@gmail.com"}
    assert is_false_negative(a3, b3) == False # public domain match doesn't count

def test_apply_deterministic_filter():
    pairs = [
        {"record_a": {"company": "A", "email": "a@x.com"}, "record_b": {"company": "B", "email": "b@y.com"}},
        {"record_a": {"company": "A", "email": "a@x.com"}, "record_b": {"company": "A", "email": "b@x.com"}} # FN
    ]
    filtered = apply_deterministic_filter(pairs)
    assert len(filtered) == 1
    assert filtered[0]["record_b"]["company"] == "B"