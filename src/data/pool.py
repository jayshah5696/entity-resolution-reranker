import polars as pl
import random
import yaml
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from src.data.sources import (
    load_census_surnames, load_ssa_names, load_names_dataset,
    parse_gleif, parse_edgar_tickers, parse_onet_alternates, parse_onet_reported,
    download_census_surnames, download_ssa_names, download_edgar_tickers,
    download_gleif, download_onet
)
import cleanco

@dataclass
class NamePool:
    given_names: list[str]
    given_weights: list[int]
    surnames: list[str]
    surname_weights: list[int]
    script: str
    country: str

@dataclass
class CompanyRecord:
    name: str
    country: str

@dataclass
class TitlePool:
    titles: list[str]
    functions: list[str]

def build_name_pool(ethnicity_targets: dict, census_df: pl.DataFrame, ssa_df: pl.DataFrame) -> dict[str, NamePool]:
    pool = {}
    
    # Process US/UK English names from Census and SSA
    if "us_uk_english" in ethnicity_targets:
        surnames = census_df["name"].to_list()
        surname_weights = census_df["count"].to_list()
        
        given_df = ssa_df.sort("count", descending=True).head(5000)
        given_names = given_df["name"].to_list()
        given_weights = given_df["count"].to_list()
        
        pool["us_uk_english"] = NamePool(
            given_names=given_names, given_weights=given_weights,
            surnames=surnames, surname_weights=surname_weights,
            script="latin", country="US"
        )
        
    # Process other ethnicities via names_dataset
    mapping = {
        "indian": ("IN", "latin"),
        "chinese": ("CN", "cjk"),
        "hispanic": ("MX", "latin"),
        "arabic": ("EG", "arabic"),
        "other": ("FR", "latin")
    }
    
    for eth, (country_code, script) in mapping.items():
        if eth in ethnicity_targets:
            first_names, first_counts, last_names, last_counts = load_names_dataset(country_code, n=5000)
            pool[eth] = NamePool(
                given_names=first_names, given_weights=first_counts,
                surnames=last_names, surname_weights=last_counts,
                script=script, country=country_code
            )
            
    return pool

def build_company_pool(gleif_df: pl.DataFrame, edgar_df: pl.DataFrame, n_companies: int = 40_000) -> list[CompanyRecord]:
    companies = []
    
    # Add SEC EDGAR companies first
    if len(edgar_df) > 0 and "title" in edgar_df.columns:
        for title in edgar_df["title"].drop_nulls().to_list():
            companies.append(CompanyRecord(name=title, country="US"))
            
    # Fill remaining from GLEIF
    if len(gleif_df) > 0 and "legal_name" in gleif_df.columns:
        valid_gleif = gleif_df.filter(pl.col("legal_name").is_not_null())
        gleif_sample = valid_gleif.sample(n=min(n_companies, len(valid_gleif)), seed=42)
        
        for row in gleif_sample.iter_rows(named=True):
            if len(companies) >= n_companies:
                break
            name = row.get("legal_name")
            country = row.get("country", "Unknown")
            if name:
                companies.append(CompanyRecord(name=name, country=country))
                
    return companies

def build_title_pool(onet_alternates: dict, onet_reported: list[str]) -> TitlePool:
    titles = set(onet_reported)
    functions = list(onet_alternates.keys())
    for code, alts in onet_alternates.items():
        for alt in alts:
            titles.add(alt)
    
    return TitlePool(titles=list(titles), functions=functions)

import re

def generate_email(first: str, last: str, company: str, rng: random.Random) -> str:
    patterns = [
        f"{first.lower()}.{last.lower()}",
        f"{first.lower()[0]}{last.lower()}",
        f"{first.lower()}_{last.lower()}",
        f"{first.lower()}{last.lower()}",
        f"{first.lower()}",
    ]
    pattern = rng.choices(patterns, weights=[50, 20, 10, 10, 10], k=1)[0]
    domain = cleanco.basename(company).lower()
    domain = re.sub(r'[^a-z0-9]', '', domain)
    if not domain:
        domain = "example"
    return f"{pattern}@{domain}.com"

def assemble_base_pool(n: int = 50_000, seed: int = 42) -> pl.DataFrame:
    rng = random.Random(seed)
    
    # Load config
    config_path = Path("configs/data.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    eth_targets = config["ethnicity_distribution"]
    
    # Load data
    raw_dir = Path("data/raw")
    census_df = load_census_surnames(raw_dir / "census_surnames.csv")
    ssa_df = load_ssa_names(raw_dir / "ssa_names")
    
    gleif_df = parse_gleif(raw_dir / "gleif_golden_copy.csv")
    edgar_df = parse_edgar_tickers(raw_dir / "company_tickers_exchange.json")
    
    onet_alts = parse_onet_alternates(raw_dir / "onet_alternate_titles.txt")
    onet_reps = parse_onet_reported(raw_dir / "onet_reported_titles.txt")
    
    # Build pools
    name_pool = build_name_pool(eth_targets, census_df, ssa_df)
    comp_pool = build_company_pool(gleif_df, edgar_df, n_companies=40_000)
    title_pool = build_title_pool(onet_alts, onet_reps)
    
    if not name_pool or not comp_pool or not title_pool.titles:
        raise ValueError("Failed to build pools due to missing data")
        
    records = []
    used_emails = set()
    
    # Calculate counts per ethnicity
    for eth, target_pct in eth_targets.items():
        count = int(n * target_pct)
        if eth not in name_pool:
            continue
            
        npool = name_pool[eth]
        
        for _ in range(count):
            first = rng.choices(npool.given_names, weights=npool.given_weights, k=1)[0].title()
            last = rng.choices(npool.surnames, weights=npool.surname_weights, k=1)[0].title()
            
            comp = rng.choice(comp_pool)
            title = rng.choice(title_pool.titles)
            
            # Ensure unique email
            email = generate_email(first, last, comp.name, rng)
            attempts = 0
            while email in used_emails and attempts < 10:
                email = generate_email(first, last, comp.name, rng) + str(rng.randint(1, 99))
                attempts += 1
            used_emails.add(email)
            
            records.append({
                "entity_id": f"E2-{len(records)+1:06d}",
                "first_name": first,
                "last_name": last,
                "company": comp.name,
                "title": title,
                "email": email,
                "country": comp.country,
                "ethnicity_group": eth,
                "name_script": npool.script
            })
            
    df = pl.DataFrame(records)
    # If slight rounding error leaves us short
    if len(df) < n:
        diff = n - len(df)
        eth = "us_uk_english"
        npool = name_pool[eth]
        extra_records = []
        for _ in range(diff):
            first = rng.choices(npool.given_names, weights=npool.given_weights, k=1)[0].title()
            last = rng.choices(npool.surnames, weights=npool.surname_weights, k=1)[0].title()
            comp = rng.choice(comp_pool)
            title = rng.choice(title_pool.titles)
            email = generate_email(first, last, comp.name, rng)
            while email in used_emails:
                email = generate_email(first, last, comp.name, rng) + str(rng.randint(1, 99))
            used_emails.add(email)
            extra_records.append({
                "entity_id": f"E2-{len(records)+len(extra_records)+1:06d}",
                "first_name": first,
                "last_name": last,
                "company": comp.name,
                "title": title,
                "email": email,
                "country": comp.country,
                "ethnicity_group": eth,
                "name_script": npool.script
            })
        df = pl.concat([df, pl.DataFrame(extra_records)])
        
    validate_pool(df)
    return df

def validate_pool(df: pl.DataFrame) -> None:
    if df["entity_id"].n_unique() != len(df):
        raise ValueError("Duplicate entity_ids found")
    if df["email"].n_unique() != len(df):
        raise ValueError("Duplicate emails found")
    required_cols = {"entity_id", "first_name", "last_name", "company", "title", "email", "country", "ethnicity_group", "name_script"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Missing required columns. Found: {df.columns}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/pool/base_pool.parquet"))
    args = parser.parse_args()
    
    print(f"Assembling base pool of 50,000 entities...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    df = assemble_base_pool()
    df.write_parquet(args.output)
    print(f"Saved to {args.output}")
    
    # Generate stats
    stats = {
        "row_count": len(df),
        "ethnicity_distribution": df["ethnicity_group"].value_counts().to_dicts(),
        "script_distribution": df["name_script"].value_counts().to_dicts()
    }
    stats_path = args.output.parent / "pool_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {stats_path}")
