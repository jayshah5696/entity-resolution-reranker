import polars as pl
import random
import json
from pathlib import Path
from datasketch import MinHash, MinHashLSH
from typing import Tuple

def assemble_pairs(
    rule_positives: list[dict],
    llm_positives: list[dict],
    negatives: list[dict],
    boundary_pairs: list[dict]
) -> pl.DataFrame:
    
    # Combine all
    all_pairs = rule_positives + llm_positives + negatives + boundary_pairs
    
    # Ensure they have a consistent schema. 
    # For now, we expect them to be dictionaries that can form a DataFrame.
    # At minimum: entity_id_a, entity_id_b, record_a, record_b, label, strategy
    df = pl.DataFrame(all_pairs)
    
    # Add string representations for easy dedup if not present
    # Assuming colval_pair logic could be run here, or we just stringify the records
    if "text_a" not in df.columns:
        df = df.with_columns(
            pl.struct(["record_a"]).map_elements(lambda x: str(x["record_a"]), return_dtype=pl.Utf8).alias("text_a"),
            pl.struct(["record_b"]).map_elements(lambda x: str(x["record_b"]), return_dtype=pl.Utf8).alias("text_b")
        )
    return df

def minhash_dedup(pairs: pl.DataFrame, threshold: float = 0.90) -> pl.DataFrame:
    if len(pairs) == 0:
        return pairs
        
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    
    kept_indices = []
    
    # Fast iteration
    records = pairs.to_dicts()
    
    for i, row in enumerate(records):
        text = f"{row.get('text_a', '')} [SEP] {row.get('text_b', '')}"
        m = MinHash(num_perm=128)
        for d in text.encode('utf8').split():
            m.update(d)
            
        result = lsh.query(m)
        if not result:
            lsh.insert(str(i), m)
            kept_indices.append(i)
            
    return pairs[kept_indices]

def deterministic_split(
    pairs: pl.DataFrame, 
    seed: int = 42, 
    ratios: Tuple[float, float, float] = (0.60, 0.20, 0.20)
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    
    # 1. Global Undersampling to balance labels strictly across the entire dataset FIRST
    pos_df = pairs.filter(pl.col("label") == 1)
    neg_df = pairs.filter(pl.col("label") == 0)
    
    # We want exactly 50/50 overall
    min_len = min(len(pos_df), len(neg_df))
    rng = random.Random(seed)
    
    if len(pos_df) > min_len:
        pos_df = pos_df.sample(n=min_len, seed=seed)
    if len(neg_df) > min_len:
        neg_df = neg_df.sample(n=min_len, seed=seed)
        
    pairs = pl.concat([pos_df, neg_df]).sample(fraction=1.0, seed=seed) # Shuffle
    
    # We must split such that test set entity_ids are completely disjoint from train/val.
    records = pairs.to_dicts()
    
    # Create an adjacency list to find connected components of entity_ids
    adj = {}
    for r in records:
        ea = r["entity_id_a"]
        eb = r["entity_id_b"]
        adj.setdefault(ea, set()).add(eb)
        adj.setdefault(eb, set()).add(ea)
        
    visited = set()
    components = []
    
    for node in adj.keys():
        if node not in visited:
            comp = set()
            stack = [node]
            while stack:
                curr = stack.pop()
                if curr not in visited:
                    visited.add(curr)
                    comp.add(curr)
                    stack.extend(adj[curr])
            components.append(comp)
            
    # Calculate sizes of each component (number of records within it)
    comp_metrics = []
    for comp in components:
        comp_records = [r for r in records if r["entity_id_a"] in comp]
        comp_metrics.append({
            "ids": comp,
            "size": len(comp_records),
            "records": comp_records
        })
        
    # Shuffle components
    rng = random.Random(seed)
    rng.shuffle(comp_metrics)
    
    total_records = sum(m["size"] for m in comp_metrics)
    target_train = int(total_records * ratios[0])
    target_val = int(total_records * ratios[1])
    target_test = total_records - target_train - target_val # Remaining
    
    bins = {
        "train": {"records": [], "size": 0, "target": target_train},
        "val": {"records": [], "size": 0, "target": target_val},
        "test": {"records": [], "size": 0, "target": target_test}
    }
    
    for metric in comp_metrics:
        # Fill based on proportion of components rather than hard limits, which allows the big component to fall wherever
        r = rng.random()
        if r < ratios[0]:
            bins["train"]["records"].extend(metric["records"])
            bins["train"]["size"] += metric["size"]
        elif r < ratios[0] + ratios[1]:
            bins["val"]["records"].extend(metric["records"])
            bins["val"]["size"] += metric["size"]
        else:
            bins["test"]["records"].extend(metric["records"])
            bins["test"]["size"] += metric["size"]
            
    # Post-split balancing: Now that the disjoint IDs are guaranteed, we randomly undersample each split internally
    def balance_split(records_list):
        if not records_list:
            return records_list
        pos = [r for r in records_list if r["label"] == 1]
        neg = [r for r in records_list if r["label"] == 0]
        min_c = min(len(pos), len(neg))
        rng.shuffle(pos)
        rng.shuffle(neg)
        return pos[:min_c] + neg[:min_c]
        
    train_records = balance_split(bins["train"]["records"])
    val_records = balance_split(bins["val"]["records"])
    test_records = balance_split(bins["test"]["records"])
    
    train_df = pl.DataFrame(train_records) if train_records else pl.DataFrame(schema=pairs.schema)
    val_df = pl.DataFrame(val_records) if val_records else pl.DataFrame(schema=pairs.schema)
    test_df = pl.DataFrame(test_records) if test_records else pl.DataFrame(schema=pairs.schema)
    
    return train_df, val_df, test_df

def validate_split(train: pl.DataFrame, val: pl.DataFrame, test: pl.DataFrame) -> None:
    # 1. No entity_id overlap between test and train/val
    train_ids = set()
    if len(train) > 0:
        train_ids.update(train["entity_id_a"].to_list() + train["entity_id_b"].to_list())
        
    val_ids = set()
    if len(val) > 0:
        val_ids.update(val["entity_id_a"].to_list() + val["entity_id_b"].to_list())
        
    test_ids = set()
    if len(test) > 0:
        test_ids.update(test["entity_id_a"].to_list() + test["entity_id_b"].to_list())
        
    if test_ids.intersection(train_ids):
        raise ValueError("Data leak: Test entity_ids found in Train")
    if test_ids.intersection(val_ids):
        raise ValueError("Data leak: Test entity_ids found in Val")
    if train_ids.intersection(val_ids):
        raise ValueError("Data leak: Train entity_ids found in Val")
        
    # 2. Check Phase 1 overlap (if phase 1 data exists)
    phase1_triplets = Path("../entity-resolution-poc/data/processed/triplet_source.parquet")
    phase1_eval = Path("../entity-resolution-poc/data/eval/eval_profiles.parquet")
    
    p1_ids = set()
    if phase1_triplets.exists():
        p1 = pl.read_parquet(phase1_triplets)
        p1_ids.update(p1["entity_id"].to_list())
    if phase1_eval.exists():
        p2 = pl.read_parquet(phase1_eval)
        p1_ids.update(p2["entity_id"].to_list())
        
    all_phase2_ids = train_ids | val_ids | test_ids
    overlap = all_phase2_ids.intersection(p1_ids)
    if overlap:
        raise ValueError(f"Phase 2 data contains {len(overlap)} entity_ids from Phase 1. Overlap is strictly forbidden.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/pairs"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    args = parser.parse_args()
    
    # Load pairs
    print("Loading pairs...")
    try:
        rule_pos = pl.read_parquet(args.output_dir / "rule_positives.parquet").to_dicts()
    except FileNotFoundError:
        rule_pos = []
    
    try:
        llm_pos = pl.read_parquet(args.output_dir / "llm_positives.parquet").to_dicts()
    except FileNotFoundError:
        llm_pos = []
        
    try:
        negatives = pl.read_parquet(args.output_dir / "negatives.parquet").to_dicts()
    except FileNotFoundError:
        negatives = []
        
    try:
        boundary = pl.read_parquet(args.output_dir / "boundary_pairs.parquet").to_dicts()
    except FileNotFoundError:
        boundary = []
        
    print(f"Loaded {len(rule_pos)} rule pos, {len(llm_pos)} llm pos, {len(negatives)} neg, {len(boundary)} bound.")
    
    print("Assembling...")
    df = assemble_pairs(rule_pos, llm_pos, negatives, boundary)
    
    print(f"Deduping {len(df)} pairs...")
    df = minhash_dedup(df)
    
    print(f"Splitting {len(df)} pairs...")
    train, val, test = deterministic_split(df)
    
    print("Validating splits...")
    validate_split(train, val, test)
    
    print("Saving parquets...")
    train.write_parquet(args.output_dir / "ce_train.parquet")
    val.write_parquet(args.output_dir / "ce_val.parquet")
    test.write_parquet(args.output_dir / "ce_test.parquet")
    
    print("Writing manifest...")
    manifest = {
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "total_rows": len(train) + len(val) + len(test)
    }
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    with open(args.manifest_dir / "split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Splits saved to {args.output_dir}, manifest saved to {args.manifest_dir}")
