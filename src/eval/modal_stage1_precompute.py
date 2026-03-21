import modal
from pathlib import Path
import os

app = modal.App("er-stage1-precompute")
vol = modal.Volume.from_name("er-indexes-vol")

# Lightweight image for CPU tasks (BM25)
cpu_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "lancedb>=0.15.0",
    "polars>=0.20.0",
    "pyarrow",
    "tqdm"
)

# Heavier image with Torch/Transformers for GPU tasks (Dense Search)
gpu_image = cpu_image.pip_install(
    "torch==2.6.0",
    "sentence-transformers==5.0.0",
    "transformers==4.49.0",
    "huggingface-hub"
)

@app.function(
    image=cpu_image,
    cpu=4.0, # Multi-core CPU, very cheap
    timeout=7200,
    volumes={"/data": vol}
)
def compute_bm25_candidates():
    import polars as pl
    import lancedb
    import time
    from tqdm import tqdm
    
    print("Loading queries...")
    queries_df = pl.read_parquet("/data/eval_data/eval_queries.parquet")
    
    print("Connecting to BM25 index...")
    db = lancedb.connect("/data/indexes/bm25_pipe")
    table = db.open_table("index")
    
    candidates = []
    start = time.time()
    for row in tqdm(queries_df.to_dicts(), desc="BM25 FTS"):
        try:
            res = table.search(row.get("query_text_pipe", ""), query_type="fts").limit(50).to_list()
            # Only store what's needed to save memory
            candidates.append([{"entity_id": c["entity_id"], "text": c.get("text", "")} for c in res])
        except Exception:
            candidates.append([])
            
    print(f"BM25 Search complete in {time.time()-start:.1f}s")
    
    # Save mapping to volume
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    # Store as a simple JSON string array column to easily align with queries
    import json
    df_out = pl.DataFrame({"query_id": queries_df["query_id"], "candidates_json": [json.dumps(c) for c in candidates]})
    out_dir = Path("/data/candidates")
    out_dir.mkdir(exist_ok=True)
    df_out.write_parquet(out_dir / "bm25_candidates.parquet")
    vol.commit()
    print("Saved BM25 candidates.")

@app.function(
    image=gpu_image,
    gpu="A10G", # Cheaper GPU is fine for just running inference encoding
    timeout=7200,
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("huggingface")]
)
def compute_dense_candidates():
    import polars as pl
    import lancedb
    import time
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm
    import os
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    
    print("Loading queries...")
    queries_df = pl.read_parquet("/data/eval_data/eval_queries.parquet")
    query_texts = queries_df["query_text_pipe"].to_list()
    
    print("Loading Dense index...")
    db = lancedb.connect("/data/indexes/gte_modernbert_base_pipe_fp32")
    table = db.open_table("index")
    
    print("Loading Model...")
    model = SentenceTransformer("jayshah5696/er-gte-modernbert-base-pipe-ft", trust_remote_code=True)
    
    print("Batch encoding queries...")
    start = time.time()
    embs = model.encode(query_texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
    
    candidates = []
    for emb in tqdm(embs, desc="Dense Search"):
        try:
            res = table.search(emb).limit(50).to_list()
            candidates.append([{"entity_id": c["entity_id"], "text": c.get("text", "")} for c in res])
        except Exception:
            candidates.append([])
            
    print(f"Dense Search complete in {time.time()-start:.1f}s")
    
    import json
    df_out = pl.DataFrame({"query_id": queries_df["query_id"], "candidates_json": [json.dumps(c) for c in candidates]})
    out_dir = Path("/data/candidates")
    out_dir.mkdir(exist_ok=True)
    df_out.write_parquet(out_dir / "dense_candidates.parquet")
    vol.commit()
    print("Saved Dense candidates.")

@app.local_entrypoint()
def run_precomputes():
    # print("Dispatching BM25 Stage 1 to CPU worker...")
    # bm25_task = compute_bm25_candidates.spawn()
    print("Dispatching Dense Stage 1 to A10G worker...")
    dense_task = compute_dense_candidates.spawn()
    
    # bm25_task.get()
    dense_task.get()
    print("Dense Stage 1 Candidates safely cached to Volume!")
