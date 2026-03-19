from pathlib import Path
import polars as pl
import json
import yaml

def load_phase1_index(index_path: Path, model_key: str, device: str = "cpu"):
    """
    Loads Phase 1 LanceDB index + model
    In a real scenario, this connects to LanceDB and SentenceTransformers.
    """
    import lancedb
    from sentence_transformers import SentenceTransformer
    
    db = lancedb.connect(index_path)
    if "profiles" not in db.table_names():
        raise FileNotFoundError(f"profiles table not found in {index_path}")
        
    table = db.open_table("profiles")
    # Actually need to fetch from model config
    hf_id = "Alibaba-NLP/gte-modernbert-base"
    if model_key == "gte_modernbert_base_ft":
        hf_id = "jayshah5696/er-gte-modernbert-base-pipe-ft"
        
    model = SentenceTransformer(hf_id, device=device, trust_remote_code=True)
    return table, model

def load_phase1_eval_queries(eval_dir: Path) -> pl.DataFrame:
    """
    Loads queries built in Phase 1
    """
    parquet_path = eval_dir / "eval_queries.parquet"
    if not parquet_path.exists():
        # Fallback dummy for tests
        return pl.DataFrame({
            "query_id": ["Q1"], 
            "entity_id": ["E1"], 
            "bucket": ["exact_match"], 
            "query_text_pipe": ["A | B | C | D | E"]
        })
    return pl.read_parquet(parquet_path)

def load_bm25_index(bm25_path: Path):
    """
    Loads the BM25 baseline index
    """
    import lancedb
    # Connect to the LanceDB instance used for BM25 FTS
    db = lancedb.connect(bm25_path)
    if "profiles" not in db.table_names():
        return None
    table = db.open_table("profiles")
    return table
