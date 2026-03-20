import modal
from pathlib import Path

app = modal.App("er-indexes-upload")
vol = modal.Volume.from_name("er-indexes-vol", create_if_missing=True)

@app.local_entrypoint()
def upload():
    # Upload local index directories directly into the Modal Volume
    local_eval = Path("../entity-resolution-poc/data/eval")
    local_bm25 = Path("../entity-resolution-poc/results/indexes/bm25_pipe")
    local_dense = Path("../entity-resolution-poc/results/indexes/gte_modernbert_base_pipe_fp32")
    
    print("Uploading evaluation queries...")
    if local_eval.exists():
        with vol.batch_upload(force=True) as batch:
            batch.put_directory(local_eval, "/eval_data")
            
    print("Uploading BM25 index...")
    if local_bm25.exists():
        with vol.batch_upload(force=True) as batch:
            batch.put_directory(local_bm25, "/indexes/bm25_pipe")
            
    print("Uploading Dense index (this may take a few minutes for 3GB)...")
    if local_dense.exists():
        with vol.batch_upload(force=True) as batch:
            batch.put_directory(local_dense, "/indexes/gte_modernbert_base_pipe_fp32")
            
    print("All required data synced to Modal Volume: 'er-indexes-vol'")
