import pytest
import random
from src.data.corrupt import (
    CORRUPTION_CODES,
    CORRUPTION_HANDLERS,
    corrupt_record,
    # Handlers
    corrupt_c1, corrupt_c2, corrupt_c3, corrupt_c4, corrupt_c5, corrupt_c6, corrupt_c7, corrupt_c8,
    corrupt_n1, corrupt_n2, corrupt_n3, corrupt_n4, corrupt_n5, corrupt_n6, corrupt_n7, corrupt_n8,
    corrupt_n9, corrupt_n10, corrupt_n11, corrupt_n12, corrupt_n13, corrupt_n14, corrupt_n15, corrupt_n16,
    corrupt_t1, corrupt_t2, corrupt_t3, corrupt_t4, corrupt_t5, corrupt_t6,
    corrupt_e1, corrupt_e2
)

def test_all_corruption_codes_have_handlers():
    for code in CORRUPTION_CODES:
        assert code in CORRUPTION_HANDLERS
        assert callable(CORRUPTION_HANDLERS[code])

# --- Company Tests (C1-C8) ---

def test_c1_legal_suffix_swap():
    rng = random.Random(42)
    record = {"company": "Acme LLC"}
    res = corrupt_c1(record.copy(), rng)
    assert res["company"] != "Acme LLC"
    assert res["company"].startswith("Acme ")

def test_c2_suffix_drop():
    rng = random.Random(42)
    record = {"company": "Microsoft Corporation"}
    res = corrupt_c2(record.copy(), rng)
    assert res["company"] == "Microsoft"

def test_c3_the_prefix_drop():
    rng = random.Random(42)
    record = {"company": "The Home Depot"}
    res = corrupt_c3(record.copy(), rng)
    assert res["company"] == "Home Depot"

def test_c4_ampersand_normalize():
    rng = random.Random(42)
    record = {"company": "Johnson & Johnson"}
    res = corrupt_c4(record.copy(), rng)
    assert res["company"] == "Johnson and Johnson"
    
    record2 = {"company": "Procter and Gamble"}
    res2 = corrupt_c4(record2.copy(), rng)
    assert res2["company"] == "Procter & Gamble"

def test_c5_company_abbreviation():
    rng = random.Random(42)
    # C5 replaces names using known abbreviation logic or acronyms
    record = {"company": "International Business Machines"}
    res = corrupt_c5(record.copy(), rng)
    assert res["company"] == "IBM" or res["company"] != "International Business Machines"

def test_c6_word_truncation():
    rng = random.Random(42)
    record = {"company": "Goldman Sachs Group"}
    res = corrupt_c6(record.copy(), rng)
    # Drops the last word if multiple words exist
    assert res["company"] == "Goldman Sachs"

def test_c7_rebrand():
    rng = random.Random(42)
    record = {"company": "Facebook Inc"} # known rebrand pair
    res = corrupt_c7(record.copy(), rng)
    # Assuming C7 will use a known dict for rebrands or just fallback to generic
    # the test expects the output to change if it's in the rebrand dict
    # We will provide a small fixture/dict in the module
    assert res["company"] != "Facebook Inc"

def test_c8_shorten_with_abbrev():
    rng = random.Random(42)
    record = {"company": "Advanced Micro Devices Corporation"}
    res = corrupt_c8(record.copy(), rng)
    assert res["company"] == "AMD" or res["company"] != "Advanced Micro Devices Corporation"

# --- Name Latin Tests (N1-N16) ---

def test_n1_diacritic_strip():
    rng = random.Random(42)
    record = {"last_name": "García"}
    res = corrupt_n1(record.copy(), rng)
    assert res["last_name"] == "Garcia"

def test_n2_single_char_delete():
    rng = random.Random(42)
    record = {"first_name": "William"}
    res = corrupt_n2(record.copy(), rng)
    assert len(res["first_name"]) == len("William") - 1

def test_n3_keyboard_sub():
    rng = random.Random(42)
    record = {"last_name": "Smith"}
    res = corrupt_n3(record.copy(), rng)
    assert len(res["last_name"]) == len("Smith")
    assert res["last_name"] != "Smith"

def test_n4_ocr_sub():
    rng = random.Random(42)
    record = {"first_name": "Illiana"}
    res = corrupt_n4(record.copy(), rng)
    assert res["first_name"] != "Illiana"

def test_n5_char_transposition():
    rng = random.Random(42)
    record = {"last_name": "Brown"}
    res = corrupt_n5(record.copy(), rng)
    assert len(res["last_name"]) == len("Brown")
    assert res["last_name"] != "Brown"

def test_n6_name_field_swap():
    rng = random.Random(42)
    record = {"first_name": "Jay", "last_name": "Shah"}
    res = corrupt_n6(record.copy(), rng)
    assert res["first_name"] == "Shah"
    assert res["last_name"] == "Jay"

def test_n7_east_asian_order_swap():
    rng = random.Random(42)
    record = {"first_name": "Wei", "last_name": "Chen"}
    res = corrupt_n7(record.copy(), rng)
    assert res["first_name"] == "Chen"
    assert res["last_name"] == "Wei"

def test_n8_first_initial():
    rng = random.Random(42)
    record = {"first_name": "Jay", "last_name": "Smith"}
    res = corrupt_n8(record.copy(), rng)
    assert res["first_name"] == "J." or res["first_name"] == "J"
    assert res["last_name"] == "Smith"

def test_n9_first_middle_initial():
    rng = random.Random(42)
    record = {"first_name": "Jay Michael", "last_name": "Smith"}
    res = corrupt_n9(record.copy(), rng)
    assert res["first_name"] == "J. M." or res["first_name"] == "J M"

def test_n10_drop_middle():
    rng = random.Random(42)
    record = {"first_name": "Jay Michael", "last_name": "Smith"}
    res = corrupt_n10(record.copy(), rng)
    assert res["first_name"] == "Jay"

def test_n11_middle_initial_only():
    rng = random.Random(42)
    record = {"first_name": "Jay Michael", "last_name": "Smith"}
    res = corrupt_n11(record.copy(), rng)
    assert res["first_name"] in ["Jay M.", "Jay M"]

def test_n12_last_initial():
    rng = random.Random(42)
    record = {"first_name": "Jay", "last_name": "Smith"}
    res = corrupt_n12(record.copy(), rng)
    assert res["last_name"] in ["S.", "S"]

def test_n13_nickname_sub():
    rng = random.Random(42)
    record = {"first_name": "William"}
    res = corrupt_n13(record.copy(), rng)
    assert res["first_name"] in {"Bill", "Will", "Billy", "Liam"}

def test_n14_phonetic_sub_english():
    rng = random.Random(42)
    record = {"first_name": "Stephen"}
    res = corrupt_n14(record.copy(), rng)
    assert res["first_name"] == "Steven"

def test_n15_hyphen_add_remove():
    rng = random.Random(42)
    record1 = {"first_name": "Mary Jane"}
    res1 = corrupt_n15(record1.copy(), rng)
    assert res1["first_name"] == "Mary-Jane"
    
    record2 = {"first_name": "Mary-Jane"}
    res2 = corrupt_n15(record2.copy(), rng)
    assert res2["first_name"] == "Mary Jane"

def test_n16_prefix_suffix_drop():
    rng = random.Random(42)
    record = {"last_name": "Smith Jr.", "first_name": "Dr. Jay"}
    res = corrupt_n16(record.copy(), rng)
    assert res["last_name"] == "Smith"
    assert res["first_name"] == "Jay"

# --- Title Tests (T1-T6) ---

def test_t1_title_abbreviation():
    rng = random.Random(42)
    record = {"title": "Chief Executive Officer"}
    res = corrupt_t1(record.copy(), rng)
    assert res["title"] == "CEO"

def test_t2_title_expansion():
    rng = random.Random(42)
    record = {"title": "CTO"}
    res = corrupt_t2(record.copy(), rng)
    assert res["title"] == "Chief Technology Officer"

def test_t3_title_reorder():
    rng = random.Random(42)
    record = {"title": "Director of Engineering"}
    res = corrupt_t3(record.copy(), rng)
    assert res["title"] != "Director of Engineering"

def test_t4_seniority_drop():
    rng = random.Random(42)
    record = {"title": "Senior Software Engineer"}
    res = corrupt_t4(record.copy(), rng)
    assert res["title"] == "Software Engineer"

def test_t5_seniority_synonym():
    rng = random.Random(42)
    record = {"title": "Senior Engineer"}
    res = corrupt_t5(record.copy(), rng)
    assert res["title"] == "Sr. Engineer" or res["title"] == "Sr Engineer"
    
def test_t6_dept_abbreviation():
    rng = random.Random(42)
    record = {"title": "VP of Engineering"}
    res = corrupt_t6(record.copy(), rng)
    assert res["title"] == "VP of Eng"

# --- Email Tests (E1-E2) ---

def test_e1_email_format_variant():
    rng = random.Random(42)
    record = {"first_name": "Jay", "last_name": "Shah", "email": "jay.shah@google.com"}
    res = corrupt_e1(record.copy(), rng)
    assert res["email"] != "jay.shah@google.com"
    assert "google.com" in res["email"]

def test_e2_domain_swap():
    rng = random.Random(42)
    record = {"email": "jay@google.com"}
    res = corrupt_e2(record.copy(), rng)
    assert "google.com" not in res["email"]
    assert res["email"].split("@")[1] in ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com"]

# --- Cross-cutting Tests ---

def test_corrupt_record_applies_sequence():
    rng = random.Random(42)
    record = {
        "entity_id": "E2-1",
        "first_name": "William",
        "last_name": "García",
        "company": "Microsoft Corporation"
    }
    # C2 drops suffix, N13 changes William to Bill/etc, N1 strips accent
    res = corrupt_record(record.copy(), ["C2", "N13", "N1"], rng)
    
    assert res["company"] == "Microsoft"
    assert res["first_name"] in {"Bill", "Will", "Billy", "Liam"}
    assert res["last_name"] == "Garcia"

def test_corrupt_record_preserves_id():
    rng = random.Random(42)
    record = {"entity_id": "E2-999", "first_name": "Jay"}
    res = corrupt_record(record.copy(), ["N2"], rng)
    assert res["entity_id"] == "E2-999"

def test_corruption_produces_different_text():
    rng = random.Random(42)
    record = {"company": "Acme LLC", "first_name": "Jay", "last_name": "Shah", "title": "Senior Engineer", "email": "jay@acme.com"}
    for code in ["C1", "N6", "T4", "E2"]:
        res = corrupt_record(record.copy(), [code], rng)
        # It must change at least one value
        changed = any(res[k] != record[k] for k in ["company", "first_name", "last_name", "title", "email"])
        assert changed, f"Code {code} did not change the record"
