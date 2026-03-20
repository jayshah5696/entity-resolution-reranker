import modal
from pathlib import Path

app = modal.App("er-cloud-indexer")

# Define the image with necessary dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.6.0",
    "transformers==4.49.0", 
    "sentence-transformers==5.0.0",
    "polars>=0.20.0",
    "lancedb>=0.15.0",
    "pyarrow",
    "huggingface-hub"
)

# The persistent volume where all our eval data will live
vol = modal.Volume.from_name("er-indexes-vol", create_if_missing=True)

@app.local_entrypoint()
def setup_and_upload():
    """Step 1: Upload the small lightweight files from local Mac to Modal Volume"""
    local_eval = Path("../entity-resolution-poc/data/eval/eval_queries.parquet")
    local_bm25 = Path("../entity-resolution-poc/results/indexes/bm25_pipe")
    local_pool = Path("data/pool/base_pool.parquet")
    
    print("Uploading BM25 Index (~144MB)...")
    if local_bm25.exists():
        with vol.batch_upload(force=True) as batch:
            batch.put_directory(local_bm25, "/indexes/bm25_pipe")
            
    print("Uploading Eval Queries (~2.2MB)...")
    if local_eval.exists():
        with vol.batch_upload(force=True) as batch:
            batch.put_file(local_eval, "/eval_data/eval_queries.parquet")
            
    print("Uploading Base Pool for Dense Encoding (~5MB)...")
    if local_pool.exists():
        with vol.batch_upload(force=True) as batch:
            batch.put_file(local_pool, "/pool/base_pool.parquet")
            
    print("Uploads complete! Triggering A100 to build the Dense Index natively in the cloud...")
    
    # Step 2: Trigger the cloud encoding
    build_dense_index.remote()

@app.function(
    image=image,
    gpu="A100", # Use A100 for blazing fast encoding
    timeout=3600,
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("huggingface")]
)
def build_dense_index():
    """Step 2: Run natively on A100 to embed 50K profiles and save 3GB index instantly"""
    import os
    import polars as pl
    import lancedb
    import pyarrow as pa
    from sentence_transformers import SentenceTransformer
    
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    
    print("Loading Base Pool...")
    pool_path = "/data/pool/base_pool.parquet"
    if not os.path.exists(pool_path):
        raise FileNotFoundError(f"Pool not found at {pool_path}. Did the upload fail?")
        
    df = pl.read_parquet(pool_path)
    records = df.to_dicts()
    
    # Serialize to Phase 1 pipe format
    def pipe_serialize(record: dict) -> str:
        fields = ["first_name", "last_name", "company", "email", "country"]
        return " | ".join([str(record.get(f, "")) for f in fields])
        
    texts = [pipe_serialize(r) for r in records]
    
    print("Loading GTE ModernBERT Fine-Tuned Model...")
    # Load the Phase 1 fine-tuned retriever from the hub
    model_id = "jayshah5696/er-gte-modernbert-base-pipe-ft"
    model = SentenceTransformer(model_id, trust_remote_code=True)
    
    print(f"Encoding {len(texts)} profiles on A100...")
    # A100 can easily handle batch_size 1024 or 2048 for base models
    embeddings = model.encode(texts, batch_size=2048, show_progress_bar=True, convert_to_numpy=True)
    
    print("Building LanceDB Index locally first to avoid FUSE IO rename errors...")
    import shutil
    index_dir = "/tmp/gte_modernbert_base_pipe_fp32"
    os.makedirs(index_dir, exist_ok=True)
    
    db = lancedb.connect(index_dir)
    
    # LanceDB requires PyArrow schema
    # Dim is 768 for modernbert-base
    schema = pa.schema([
        pa.field("vector", pa.list_(pa.float32(), 768)),
        pa.field("entity_id", pa.string()),
        pa.field("text", pa.string())
    ])
    
    # Structure data for Lance
    data = []
    for i in range(len(records)):
        data.append({
            "vector": embeddings[i],
            "entity_id": records[i]["entity_id"],
            "text": texts[i]
        })
        
    if "index" in db.table_names():
        db.drop_table("index")
        
    table = db.create_table("index", data=data, schema=schema)
    
    print("Creating IVF_PQ Index for lightning fast search...")
    # Create the vector index natively
    table.create_index(metric="cosine", vector_column_name="vector")
    
    print("Transferring built index to persistent Volume...")
    target_dir = "/data/indexes/gte_modernbert_base_pipe_fp32"
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    shutil.copytree(index_dir, target_dir)
    
    # Sync the volume to ensure it saves permanently!
    vol.commit()
    print("Successfully built and saved 3GB Dense Index to Modal Volume!")
