import modal
app = modal.App("pull-results")
vol = modal.Volume.from_name("er-indexes-vol")

@app.local_entrypoint()
def pull():
    with vol.batch_download() as batch:
        batch.download_file("/results/001_bm25_plus_minilm_reranker.json", "results/001_bm25_plus_minilm_reranker.json")
