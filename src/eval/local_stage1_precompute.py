"""
Local Stage 1 candidate precomputation (BM25 + Dense).

Mirrors modal_stage1_precompute.py but runs entirely on the local machine,
using the Phase 1 LanceDB indexes at the sibling repo path.
"""

import json
import time
from pathlib import Path

import lancedb
import polars as pl
from tqdm import tqdm


PHASE1_DENSE_INDEX = Path(
    "../entity-resolution-poc/results/indexes/gte_modernbert_base_pipe_fp32"
)
PHASE1_BM25_INDEX = Path("../entity-resolution-poc/results/indexes/bm25_pipe")
EVAL_QUERIES = Path("../entity-resolution-poc/data/eval/eval_queries.parquet")
DENSE_OUTPUT = Path("results/dense_candidates.parquet")
BM25_OUTPUT = Path("results/bm25_candidates.parquet")
MODEL_ID = "jayshah5696/er-gte-modernbert-base-pipe-ft"
TOP_K = 50
ENCODE_BATCH = 256


def _sanity_check(
    queries_df: pl.DataFrame, candidates: list[list[dict]], label: str
) -> None:
    """Print entity_id format check and hit rate."""
    sample_cand_id = (
        candidates[0][0]["entity_id"] if candidates and candidates[0] else "N/A"
    )
    sample_query_id = str(queries_df["entity_id"][0])
    print(f"\n  Sanity check ({label}):")
    print(f"    Candidate entity_id sample: {sample_cand_id}")
    print(f"    Query entity_id sample:     {sample_query_id}")
    both_uuid = "-" in sample_cand_id and "-" in sample_query_id
    print(f"    Format match (both UUIDs):  {both_uuid}")
    if not both_uuid:
        print("  WARNING: entity_id formats do not match! Results will be 0.0.")

    hits = 0
    checked = min(100, len(queries_df))
    for i in range(checked):
        eid = str(queries_df["entity_id"][i])
        cand_ids = {c["entity_id"] for c in candidates[i]}
        if eid in cand_ids:
            hits += 1
    print(
        f"    Hit rate (first {checked} queries): {hits}/{checked} = {hits / checked:.1%}"
    )


def _save(queries_df: pl.DataFrame, candidates: list[list[dict]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df_out = pl.DataFrame(
        {
            "query_id": queries_df["query_id"],
            "candidates_json": [json.dumps(c) for c in candidates],
        }
    )
    df_out.write_parquet(path)
    print(f"  Saved {len(df_out)} rows to {path}")


def compute_bm25(queries_df: pl.DataFrame) -> None:
    print("\n=== BM25 Stage 1 ===")
    if not PHASE1_BM25_INDEX.exists():
        raise FileNotFoundError(f"BM25 index not found: {PHASE1_BM25_INDEX}")

    db = lancedb.connect(PHASE1_BM25_INDEX)
    table = db.open_table("index")
    print(f"  {table.count_rows()} rows in BM25 index")

    t0 = time.time()
    candidates: list[list[dict]] = []
    for row in tqdm(queries_df.to_dicts(), desc="BM25 FTS"):
        try:
            res = (
                table.search(row.get("query_text_pipe", ""), query_type="fts")
                .limit(TOP_K)
                .to_list()
            )
            candidates.append(
                [{"entity_id": c["entity_id"], "text": c.get("text", "")} for c in res]
            )
        except Exception:
            candidates.append([])
    print(f"  BM25 search done in {time.time() - t0:.1f}s")

    _sanity_check(queries_df, candidates, "BM25")
    _save(queries_df, candidates, BM25_OUTPUT)


def compute_dense(queries_df: pl.DataFrame) -> None:
    print("\n=== Dense Stage 1 ===")
    if not PHASE1_DENSE_INDEX.exists():
        raise FileNotFoundError(f"Dense index not found: {PHASE1_DENSE_INDEX}")

    from sentence_transformers import SentenceTransformer

    db = lancedb.connect(PHASE1_DENSE_INDEX)
    table = db.open_table("index")
    print(f"  {table.count_rows()} rows in dense index")

    print(f"  Loading encoder: {MODEL_ID}")
    model = SentenceTransformer(MODEL_ID, trust_remote_code=True)

    query_texts = queries_df["query_text_pipe"].to_list()
    print(f"  Encoding {len(query_texts)} queries...")
    t0 = time.time()
    embs = model.encode(
        query_texts,
        batch_size=ENCODE_BATCH,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"  Encoding done in {time.time() - t0:.1f}s")

    t1 = time.time()
    candidates: list[list[dict]] = []
    for emb in tqdm(embs, desc="Dense Search"):
        try:
            res = table.search(emb).limit(TOP_K).to_list()
            candidates.append(
                [{"entity_id": c["entity_id"], "text": c.get("text", "")} for c in res]
            )
        except Exception:
            candidates.append([])
    print(f"  Search done in {time.time() - t1:.1f}s")

    _sanity_check(queries_df, candidates, "Dense")
    _save(queries_df, candidates, DENSE_OUTPUT)


def main() -> None:
    if not EVAL_QUERIES.exists():
        raise FileNotFoundError(f"Eval queries not found: {EVAL_QUERIES}")

    print("Loading eval queries...")
    queries_df = pl.read_parquet(EVAL_QUERIES)
    print(f"  {len(queries_df)} queries loaded")

    compute_bm25(queries_df)
    compute_dense(queries_df)

    print("\nAll Stage 1 candidates precomputed.")


if __name__ == "__main__":
    main()
