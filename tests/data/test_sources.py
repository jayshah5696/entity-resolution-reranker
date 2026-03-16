import pytest
from pathlib import Path
import polars as pl
import cleanco

from src.data.sources import (
    load_names_dataset,
    load_nicknames,
    load_census_surnames,
    load_ssa_names,
    parse_gleif,
    parse_gleif_aliases,
    parse_onet_alternates,
    parse_onet_reported,
    parse_edgar_tickers
)

def test_names_dataset():
    names, counts = load_names_dataset("IN", n=50)
    assert len(names) > 0
    assert len(names) == len(counts)

def test_nicknames():
    nn = load_nicknames()
    assert "william" in nn
    assert "bill" in nn["william"]

def test_cleanco():
    assert cleanco.basename("Acme Corporation") == "Acme"

@pytest.mark.slow
def test_census_surnames():
    path = Path("data/raw/census_surnames.csv")
    df = load_census_surnames(path)
    assert len(df) > 160000
    assert "name" in df.columns and "count" in df.columns
    # Smith is rank 1
    assert df.row(0)[0].upper() == "SMITH"

@pytest.mark.slow
def test_ssa_names():
    path = Path("data/raw/ssa_names")
    df = load_ssa_names(path)
    assert "name" in df.columns and "sex" in df.columns and "count" in df.columns

@pytest.mark.slow
def test_gleif_parse():
    path = Path("data/raw/gleif_golden_copy.csv")
    df = parse_gleif(path)
    assert len(df) > 1000000
    assert "legal_name" in df.columns

@pytest.mark.slow
def test_gleif_aliases():
    path = Path("data/raw/gleif_golden_copy.csv")
    df = parse_gleif(path)
    aliases = parse_gleif_aliases(df)
    assert isinstance(aliases, dict)

@pytest.mark.slow
def test_onet_alternates():
    path = Path("data/raw/onet_alternate_titles.txt")
    alts = parse_onet_alternates(path)
    assert len(alts) > 27000

@pytest.mark.slow
def test_onet_reported():
    path = Path("data/raw/onet_reported_titles.txt")
    reps = parse_onet_reported(path)
    assert len(reps) > 35000

@pytest.mark.slow
def test_edgar_tickers():
    path = Path("data/raw/company_tickers_exchange.json")
    df = parse_edgar_tickers(path)
    assert len(df) > 5000
