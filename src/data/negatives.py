import polars as pl
import random
import itertools
from doublemetaphone import doublemetaphone
import jellyfish
from cleanco import basename

def format_pair(a: dict, b: dict, strategy: str) -> dict:
    return {
        "entity_id_a": a["entity_id"],
        "entity_id_b": b["entity_id"],
        "record_a": a,
        "record_b": b,
        "strategy": strategy,
        "label": 0
    }

def mine_same_company_diff_person(pool: pl.DataFrame) -> list[dict]:
    # Group by company
    pairs = []
    grouped = pool.group_by("company")
    # For a real implementation, we would sample pairs within groups
    for _, group in grouped:
        if len(group) > 1:
            records = group.to_dicts()
            # Randomly pair up to some limit
            # To be efficient, we can shuffle and pair adjacent
            random.shuffle(records)
            for i in range(len(records) - 1):
                if records[i]["first_name"] != records[i+1]["first_name"]:
                    pairs.append(format_pair(records[i], records[i+1], "NEG1"))
    return pairs

def mine_phonetic_neighbor(pool: pl.DataFrame) -> list[dict]:
    # Compute double metaphone on first name
    records = pool.to_dicts()
    for r in records:
        r["dm"] = doublemetaphone(r.get("first_name", ""))[0]
        
    pairs = []
    # group by dm
    dm_groups = {}
    for r in records:
        if r["dm"]:
            dm_groups.setdefault(r["dm"], []).append(r)
            
    for dm, group in dm_groups.items():
        if len(group) > 1:
            random.shuffle(group)
            for i in range(len(group) - 1):
                if group[i]["first_name"] != group[i+1]["first_name"] and group[i]["entity_id"] != group[i+1]["entity_id"]:
                    pairs.append(format_pair(group[i], group[i+1], "NEG2"))
    return pairs

def mine_common_name_diff_company(pool: pl.DataFrame, census_df: pl.DataFrame, top_n: int = 500) -> list[dict]:
    top_names = set(census_df.head(top_n)["name"].str.to_uppercase().to_list())
    records = pool.to_dicts()
    
    name_groups = {}
    for r in records:
        ln = r.get("last_name", "").upper()
        if ln in top_names:
            name_groups.setdefault(ln, []).append(r)
            
    pairs = []
    for ln, group in name_groups.items():
        if len(group) > 1:
            random.shuffle(group)
            for i in range(len(group) - 1):
                if group[i]["company"] != group[i+1]["company"] and group[i]["entity_id"] != group[i+1]["entity_id"]:
                    pairs.append(format_pair(group[i], group[i+1], "NEG3"))
    return pairs

def mine_title_function_swap(pool: pl.DataFrame, onet_groups: dict) -> list[dict]:
    # Same seniority, different function
    # group by seniority prefix.
    records = pool.to_dicts()
    seniorities = ["Senior", "Junior", "VP", "Chief", "Director", "Lead"]
    
    sen_groups = {s: [] for s in seniorities}
    for r in records:
        t = r.get("title", "")
        for s in seniorities:
            if s in t:
                sen_groups[s].append(r)
                break
                
    pairs = []
    for s, group in sen_groups.items():
        if len(group) > 1:
            random.shuffle(group)
            for i in range(len(group) - 1):
                if group[i]["title"] != group[i+1]["title"] and group[i]["entity_id"] != group[i+1]["entity_id"]:
                    pairs.append(format_pair(group[i], group[i+1], "NEG4"))
    return pairs

def mine_title_level_swap(pool: pl.DataFrame) -> list[dict]:
    # Same function, different seniority
    # Extract base title by removing seniority
    records = pool.to_dicts()
    seniorities = ["Senior ", "Junior ", "VP ", "Chief ", "Director of ", "Lead "]
    
    base_groups = {}
    for r in records:
        t = r.get("title", "")
        base = t
        for s in seniorities:
            if t.startswith(s):
                base = t[len(s):]
                break
        base_groups.setdefault(base, []).append(r)
        
    pairs = []
    for base, group in base_groups.items():
        if len(group) > 1:
            random.shuffle(group)
            for i in range(len(group) - 1):
                if group[i]["title"] != group[i+1]["title"] and group[i]["entity_id"] != group[i+1]["entity_id"]:
                    pairs.append(format_pair(group[i], group[i+1], "NEG5"))
    return pairs

def mine_random(pool: pl.DataFrame) -> list[dict]:
    records = pool.to_dicts()
    pairs = []
    random.shuffle(records)
    for i in range(len(records) - 1):
        pairs.append(format_pair(records[i], records[i+1], "NEG6"))
    return pairs

def mine_bm25_hard_negatives(pool: pl.DataFrame) -> list[dict]:
    # Mock BM25 implementation for negatives
    # In practice this would index the pool and search it.
    # Actually, we can just group by last_name to simulate it.
    records = pool.to_dicts()
    pairs = []
    random.shuffle(records)
    
    ln_groups = {}
    for r in records:
        ln_groups.setdefault(r.get("last_name", ""), []).append(r)
        
    for ln, group in ln_groups.items():
        if len(group) > 1:
            for i in range(len(group) - 1):
                if group[i]["first_name"] != group[i+1]["first_name"] and group[i]["entity_id"] != group[i+1]["entity_id"]:
                    pairs.append(format_pair(group[i], group[i+1], "NEG7"))
    return pairs

def get_domain(email: str) -> str:
    if email and "@" in email:
        return email.split("@")[1].lower()
    return ""

def is_false_negative(anchor: dict, candidate: dict) -> bool:
    # Deterministic check to ensure we aren't creating a false negative
    domain_a = get_domain(anchor.get("email", ""))
    domain_b = get_domain(candidate.get("email", ""))
    
    # 1. Email domains match AND domains are not public/generic
    public_domains = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "example.com"}
    if domain_a and domain_a == domain_b and domain_a not in public_domains:
        return True
        
    # 2. Normalized company names match AND Jaro-Winkler > 0.92
    comp_a = basename(anchor.get("company", "")).lower()
    comp_b = basename(candidate.get("company", "")).lower()
    if comp_a and comp_b:
        jw = jellyfish.jaro_winkler_similarity(comp_a, comp_b)
        if jw > 0.92:
            return True
            
    return False

def apply_deterministic_filter(pairs: list[dict]) -> list[dict]:
    filtered = []
    for p in pairs:
        if not is_false_negative(p["record_a"], p["record_b"]):
            filtered.append(p)
    return filtered
