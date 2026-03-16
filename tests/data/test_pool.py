import pytest
import polars as pl
from pathlib import Path
from src.data.pool import (
    build_name_pool, build_company_pool, build_title_pool,
    generate_email, assemble_base_pool, validate_pool
)

def test_generate_email():
    import random
    rng = random.Random(42)
    email = generate_email("Jay", "Shah", "Google Inc", rng)
    assert email.endswith("@google.com")
    assert "jay" in email or "shah" in email

def test_pool_schema_and_size():
    pool_path = Path("data/pool/base_pool.parquet")
    if not pool_path.exists():
        pytest.skip("Pool not generated yet")
    df = pl.read_parquet(pool_path)
    
    assert len(df) == 50_000
    expected_cols = {"entity_id", "first_name", "last_name", "company", "title", "email", "country", "ethnicity_group", "name_script"}
    assert expected_cols.issubset(set(df.columns))
    
    assert df["entity_id"].n_unique() == 50_000
    assert df["email"].n_unique() == 50_000

def test_ethnicity_distribution():
    pool_path = Path("data/pool/base_pool.parquet")
    if not pool_path.exists():
        pytest.skip("Pool not generated yet")
    df = pl.read_parquet(pool_path)
    dist = df["ethnicity_group"].value_counts(normalize=True)
    dist_dict = dict(zip(dist["ethnicity_group"], dist["proportion"]))
    
    # Expected from config: us_uk_english: 0.25, indian: 0.20, etc.
    assert 0.20 < dist_dict.get("us_uk_english", 0) < 0.30
    assert 0.15 < dist_dict.get("indian", 0) < 0.25

def test_name_frequency_weighted():
    pool_path = Path("data/pool/base_pool.parquet")
    if not pool_path.exists():
        pytest.skip("Pool not generated yet")
    df = pl.read_parquet(pool_path)
    english_df = df.filter(pl.col("ethnicity_group") == "us_uk_english")
    ln_counts = english_df["last_name"].value_counts().sort("count", descending=True)
    
    top_name = ln_counts.row(0)[0].upper()
    assert top_name == "SMITH"
    top_count = ln_counts.row(0)[1]
    
    # Check that the distribution is not uniform
    assert top_count >= 20

def test_no_phase1_overlap():
    pool_path = Path("data/pool/base_pool.parquet")
    if not pool_path.exists():
        pytest.skip("Pool not generated yet")
    df = pl.read_parquet(pool_path)
    
    phase1_triplets = Path("../entity-resolution-poc/data/processed/triplet_source.parquet")
    phase1_eval = Path("../entity-resolution-poc/data/eval/eval_profiles.parquet")
    
    phase1_ids = set()
    if phase1_triplets.exists():
        p1 = pl.read_parquet(phase1_triplets)
        phase1_ids.update(p1["entity_id"].to_list())
    if phase1_eval.exists():
        p2 = pl.read_parquet(phase1_eval)
        phase1_ids.update(p2["entity_id"].to_list())
        
    ids = set(df["entity_id"].to_list())
    assert all(i.startswith("E2-") for i in ids)
    assert len(ids.intersection(phase1_ids)) == 0

def test_validate_pool_raises():
    bad_df = pl.DataFrame({"entity_id": ["E2-1", "E2-1"], "email": ["a@b.com", "a@b.com"]})
    with pytest.raises(ValueError):
        validate_pool(bad_df)
