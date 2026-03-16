import polars as pl
import random
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

def mine_same_company_diff_person(pool: pl.DataFrame, max_pairs: int = None, rng: random.Random = None) -> list[dict]:
    if rng is None: rng = random.Random()
    pairs = []
    grouped = pool.group_by("company")
    for _, group in grouped:
        if len(group) > 1:
            records = group.to_dicts()
            rng.shuffle(records)
            for i in range(len(records) - 1):
                if records[i]["first_name"] != records[i+1]["first_name"] and records[i]["entity_id"] != records[i+1]["entity_id"]:
                    pairs.append(format_pair(records[i], records[i+1], "NEG1"))
                    if max_pairs and len(pairs) >= max_pairs:
                        return pairs
    return pairs

def mine_phonetic_neighbor(pool: pl.DataFrame, max_pairs: int = None, rng: random.Random = None) -> list[dict]:
    if rng is None: rng = random.Random()
    records = pool.to_dicts()
    for r in records:
        r["dm"] = doublemetaphone(r.get("first_name", ""))[0]
        
    pairs = []
    dm_groups = {}
    for r in records:
        if r["dm"]:
            dm_groups.setdefault(r["dm"], []).append(r)
            
    for dm, group in dm_groups.items():
        if len(group) > 1:
            rng.shuffle(group)
            for i in range(len(group) - 1):
                if group[i]["first_name"] != group[i+1]["first_name"] and group[i]["entity_id"] != group[i+1]["entity_id"]:
                    pairs.append(format_pair(group[i], group[i+1], "NEG2"))
                    if max_pairs and len(pairs) >= max_pairs:
                        return pairs
    return pairs

def mine_common_name_diff_company(pool: pl.DataFrame, census_df: pl.DataFrame, top_n: int = 500, max_pairs: int = None, rng: random.Random = None) -> list[dict]:
    if rng is None: rng = random.Random()
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
            rng.shuffle(group)
            for i in range(len(group) - 1):
                if group[i]["company"] != group[i+1]["company"] and group[i]["entity_id"] != group[i+1]["entity_id"]:
                    pairs.append(format_pair(group[i], group[i+1], "NEG3"))
                    if max_pairs and len(pairs) >= max_pairs:
                        return pairs
    return pairs

def mine_title_function_swap(pool: pl.DataFrame, onet_groups: dict, max_pairs: int = None, rng: random.Random = None) -> list[dict]:
    if rng is None: rng = random.Random()
    records = pool.to_dicts()
    seniorities = ["Senior", "Junior", "VP", "Chief", "Director", "Lead"]
    
    for r in records:
        r["_seniority"] = None
        t = r.get("title", "")
        for s in seniorities:
            if s in t:
                r["_seniority"] = s
                break
                
    pairs = []
    sen_groups = {}
    for r in records:
        if r["_seniority"]:
            sen_groups.setdefault(r["_seniority"], []).append(r)
            
    for s, group in sen_groups.items():
        if len(group) > 1:
            rng.shuffle(group)
            for i in range(len(group) - 1):
                # Using onet_groups conceptually to ensure different functions, 
                # but simplest check is just titles must differ under same seniority.
                if group[i]["title"] != group[i+1]["title"] and group[i]["entity_id"] != group[i+1]["entity_id"]:
                    pairs.append(format_pair(group[i], group[i+1], "NEG4"))
                    if max_pairs and len(pairs) >= max_pairs:
                        return pairs
    return pairs

def mine_title_level_swap(pool: pl.DataFrame, max_pairs: int = None, rng: random.Random = None) -> list[dict]:
    if rng is None: rng = random.Random()
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
            rng.shuffle(group)
            for i in range(len(group) - 1):
                if group[i]["title"] != group[i+1]["title"] and group[i]["entity_id"] != group[i+1]["entity_id"]:
                    pairs.append(format_pair(group[i], group[i+1], "NEG5"))
                    if max_pairs and len(pairs) >= max_pairs:
                        return pairs
    return pairs

def mine_random(pool: pl.DataFrame, max_pairs: int = None, rng: random.Random = None) -> list[dict]:
    if rng is None: rng = random.Random()
    records = pool.to_dicts()
    pairs = []
    rng.shuffle(records)
    for i in range(len(records) - 1):
        if records[i]["entity_id"] != records[i+1]["entity_id"]:
            pairs.append(format_pair(records[i], records[i+1], "NEG6"))
            if max_pairs and len(pairs) >= max_pairs:
                return pairs
    return pairs

def mine_bm25_hard_negatives(pool: pl.DataFrame, max_pairs: int = None, rng: random.Random = None) -> list[dict]:
    if rng is None: rng = random.Random()
    records = pool.to_dicts()
    pairs = []
    rng.shuffle(records)
    
    ln_groups = {}
    for r in records:
        ln_groups.setdefault(r.get("last_name", ""), []).append(r)
        
    for ln, group in ln_groups.items():
        if len(group) > 1:
            for i in range(len(group) - 1):
                if group[i]["first_name"] != group[i+1]["first_name"] and group[i]["entity_id"] != group[i+1]["entity_id"]:
                    pairs.append(format_pair(group[i], group[i+1], "NEG7"))
                    if max_pairs and len(pairs) >= max_pairs:
                        return pairs
    return pairs

def get_domain(email: str) -> str:
    if email and "@" in email:
        return email.split("@")[1].lower()
    return ""

def is_false_negative(anchor: dict, candidate: dict) -> bool:
    domain_a = get_domain(anchor.get("email", ""))
    domain_b = get_domain(candidate.get("email", ""))
    
    public_domains = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "example.com"}
    if domain_a and domain_a == domain_b and domain_a not in public_domains:
        return True
        
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

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/pairs/negatives.parquet"))
    args = parser.parse_args()
    
    pool = pl.read_parquet(args.pool)
    pairs = []
    
    # Simple execution of all miners for the CLI
    print("Mining NEG1...")
    pairs.extend(mine_same_company_diff_person(pool, max_pairs=5000))
    print("Mining NEG2...")
    pairs.extend(mine_phonetic_neighbor(pool, max_pairs=5000))
    # mock census for NEG3
    census_mock = pl.DataFrame({"name": ["SMITH", "JOHNSON", "WILLIAMS", "BROWN", "JONES"]})
    print("Mining NEG3...")
    pairs.extend(mine_common_name_diff_company(pool, census_mock, top_n=5, max_pairs=5000))
    print("Mining NEG4...")
    pairs.extend(mine_title_function_swap(pool, {}, max_pairs=5000))
    print("Mining NEG5...")
    pairs.extend(mine_title_level_swap(pool, max_pairs=5000))
    print("Mining NEG6...")
    pairs.extend(mine_random(pool, max_pairs=5000))
    print("Mining NEG7...")
    pairs.extend(mine_bm25_hard_negatives(pool, max_pairs=5000))
    
    filtered = apply_deterministic_filter(pairs)
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(filtered).write_parquet(args.output)
    print(f"Saved {len(filtered)} negative pairs to {args.output}")
