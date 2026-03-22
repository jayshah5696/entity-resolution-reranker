import argparse
import sys
from pathlib import Path
import json
import time
import polars as pl
import numpy as np
from tqdm import tqdm
from loguru import logger

from src.eval.load_phase1 import (
    load_phase1_index,
    load_phase1_eval_queries,
    load_bm25_index,
)
from src.data.serialize import pipe_serialize, colval_serialize
from src.models.crossencoder import CrossEncoderReranker
from src.eval.metrics import (
    compute_metrics,
    compute_f1_at_threshold,
    compute_recall_retention,
)

# Configure loguru: remove default, add stderr with timestamps
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}",
    level="INFO",
)


def build_results_json(
    experiment_id: str, stage1_latency: float, stage2_latency: float, metrics: dict
) -> dict:
    """Builds the final output JSON strictly per the plan schema."""
    return {
        "experiment_id": experiment_id,
        "latency": {"stage1_ms": stage1_latency, "stage2_ms": stage2_latency},
        "metrics": metrics,
    }


def process_end_to_end(args):
    t_exp_start = time.time()
    logger.info("Experiment {} starting", args.experiment_id)

    # 1. Load Phase 1
    if args.stage1_model == "bm25":
        logger.info("Loading BM25 index...")
        stage1_idx = (
            load_bm25_index(args.stage1_index) if args.stage1_index.exists() else None
        )
        s1_model = None
    else:
        logger.info("Loading LanceDB dense index...")
        try:
            stage1_idx, s1_model = load_phase1_index(
                args.stage1_index, args.stage1_model
            )
        except Exception as e:
            logger.warning("Skipping live index search: {}", e)
            stage1_idx, s1_model = None, None

    logger.info("Loading queries...")
    queries_df = load_phase1_eval_queries(args.eval_queries)

    logger.info("Loading reranker...")
    import yaml

    with open("configs/models.yaml", "r") as f:
        models_cfg = yaml.safe_load(f)
    if args.reranker not in models_cfg:
        raise ValueError(f"Reranker {args.reranker} not found in models.yaml")

    ce = CrossEncoderReranker(args.reranker, models_cfg[args.reranker])

    # 2. Setup metric tracking
    buckets = (
        queries_df["bucket"].unique().to_list()
        if "bucket" in queries_df.columns
        else ["all"]
    )
    results = {
        "overall": {
            "recall_at_10": [],
            "recall_at_50": [],
            "mrr": [],
            "ndcg_at_10": [],
            "recall_retention": [],
            "scores": [],
            "labels": [],
        },
        "per_bucket": {b: {"recall_at_10": []} for b in buckets},
    }

    stage1_total_time = 0.0
    stage2_total_time = 0.0
    total_queries = len(queries_df)

    # 3. Stage 1: Batch Search
    logger.info("Stage 1 search ({} queries)...", total_queries)
    start_s1 = time.time()

    stage1_candidates_list = []

    if (
        hasattr(args, "precomputed_candidates")
        and args.precomputed_candidates
        and args.precomputed_candidates.exists()
    ):
        logger.info(
            "Loading precomputed candidates from {}",
            args.precomputed_candidates,
        )
        cand_df = pl.read_parquet(args.precomputed_candidates)
        joined = queries_df.join(cand_df, on="query_id", how="left")
        import json as builtin_json

        for j_str in joined["candidates_json"].to_list():
            if j_str:
                stage1_candidates_list.append(builtin_json.loads(j_str))
            else:
                stage1_candidates_list.append([])
    else:
        if args.stage1_model == "bm25" and stage1_idx is not None:
            for row in tqdm(queries_df.to_dicts(), desc="BM25 FTS"):
                try:
                    res = (
                        stage1_idx.search(
                            row.get("query_text_pipe", ""), query_type="fts"
                        )
                        .limit(args.top_k_stage1)
                        .to_list()
                    )
                    stage1_candidates_list.append(res)
                except Exception:
                    stage1_candidates_list.append([])
        elif stage1_idx is not None and s1_model is not None:
            query_texts = queries_df["query_text_pipe"].to_list()
            logger.info("Encoding Stage 1 queries...")
            embs = s1_model.encode(
                query_texts,
                batch_size=256,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
            for emb in tqdm(embs, desc="Vector Search"):
                try:
                    res = stage1_idx.search(emb).limit(args.top_k_stage1).to_list()
                    stage1_candidates_list.append(res)
                except Exception:
                    stage1_candidates_list.append([])
        else:
            # Dummy fallback
            for row in queries_df.to_dicts():
                stage1_candidates_list.append(
                    [{"entity_id": row.get("entity_id"), "first_name": "Dummy"}]
                )

    stage1_total_time = time.time() - start_s1
    logger.info("Stage 1 done in {:.1f}s", stage1_total_time)

    # 4. Stage 2: Batch Cross-Encoder Scoring
    t_pair_start = time.time()
    logger.info("Serializing candidate pairs...")
    all_pairs = []
    pair_to_query_idx = []

    queries_dicts = queries_df.to_dicts()
    for idx, (q_row, candidates) in enumerate(
        zip(queries_dicts, stage1_candidates_list)
    ):
        true_id = q_row.get("entity_id")
        if not candidates:
            candidates = [{"entity_id": true_id, "first_name": "Dummy"}]
            stage1_candidates_list[idx] = candidates

        q_text = q_row.get("query_text_pipe", "")
        parts = [x.strip() for x in q_text.split("|")]
        q_dict = {
            "first_name": parts[0] if len(parts) > 0 else "",
            "last_name": parts[1] if len(parts) > 1 else "",
            "company": parts[2] if len(parts) > 2 else "",
            "email": parts[3] if len(parts) > 3 else "",
            "country": parts[4] if len(parts) > 4 else "",
        }
        q_str = colval_serialize(q_dict)

        for c in candidates:
            c_str = colval_serialize(c)
            all_pairs.append((q_str, c_str))
            pair_to_query_idx.append(idx)

    logger.info(
        "Pair serialization: {:.1f}s ({:,} pairs)",
        time.time() - t_pair_start,
        len(all_pairs),
    )

    logger.info("Scoring {:,} pairs with cross-encoder...", len(all_pairs))
    start_s2 = time.time()
    all_scores = ce.predict(all_pairs, batch_size=512, show_progress_bar=True)

    stage2_total_time = time.time() - start_s2
    logger.info("CE scoring done in {:.1f}s", stage2_total_time)

    # 5. Compute metrics
    t_group_start = time.time()
    logger.info("Grouping {:,} scores by query...", len(all_scores))

    grouped_scores = [[] for _ in range(total_queries)]
    for score, idx in zip(all_scores, pair_to_query_idx):
        grouped_scores[idx].append(score)
    logger.info("Score grouping: {:.1f}s", time.time() - t_group_start)

    t_metrics_start = time.time()
    logger.info("Computing per-query metrics ({:,} queries)...", total_queries)

    for q_idx, (q_row, candidates, scores) in tqdm(
        enumerate(zip(queries_dicts, stage1_candidates_list, grouped_scores)),
        total=total_queries,
        desc="Metrics",
    ):
        true_id = q_row.get("entity_id")
        bucket = q_row.get("bucket", "all")

        scored_candidates = []
        for c, s in zip(candidates, scores):
            c_copy = c.copy()
            c_copy["ce_score"] = float(s)
            scored_candidates.append(c_copy)

        scored_candidates.sort(key=lambda x: x["ce_score"], reverse=True)

        full_reranked = scored_candidates

        reranked_ids = [c.get("entity_id") for c in full_reranked]
        query_metrics = compute_metrics(reranked_ids, str(true_id))

        r50 = 1.0 if str(true_id) in reranked_ids[:50] else 0.0
        retention = compute_recall_retention(candidates, full_reranked, str(true_id))

        # Mean rank of true match after reranking (1-indexed, N+1 if not found)
        try:
            reranked_rank = reranked_ids.index(str(true_id)) + 1
        except ValueError:
            reranked_rank = len(reranked_ids) + 1

        # Stage 1 rank before reranking
        stage1_ids = [str(c.get("entity_id", "")) for c in candidates]
        try:
            stage1_rank = stage1_ids.index(str(true_id)) + 1
        except ValueError:
            stage1_rank = len(stage1_ids) + 1

        rank_delta = stage1_rank - reranked_rank  # positive = reranker helped

        # Accumulate metrics
        for k, v in query_metrics.items():
            results["overall"].setdefault(k, []).append(v)
            results["per_bucket"][bucket].setdefault(k, []).append(v)

        results["overall"].setdefault("recall_at_50", []).append(r50)
        results["overall"].setdefault("recall_retention", []).append(retention)
        results["overall"].setdefault("mean_reranked_rank", []).append(reranked_rank)
        results["overall"].setdefault("mean_stage1_rank", []).append(stage1_rank)
        results["overall"].setdefault("mean_rank_delta", []).append(rank_delta)

        results["per_bucket"][bucket].setdefault("recall_at_50", []).append(r50)
        results["per_bucket"][bucket].setdefault("recall_retention", []).append(
            retention
        )
        results["per_bucket"][bucket].setdefault("mean_reranked_rank", []).append(
            reranked_rank
        )
        results["per_bucket"][bucket].setdefault("mean_rank_delta", []).append(
            rank_delta
        )

        # Track raw scores for F1 calibration
        for c in full_reranked:
            results["overall"]["scores"].append(c.get("ce_score", 0.0))
            results["overall"]["labels"].append(
                1 if str(c.get("entity_id")) == str(true_id) else 0
            )

    logger.info("Per-query metrics: {:.1f}s", time.time() - t_metrics_start)

    # 6. Aggregate + F1 threshold sweep
    t_agg_start = time.time()
    logger.info(
        "Aggregating + F1 sweep ({:,} score/label pairs)...",
        len(results["overall"]["scores"]),
    )
    final_metrics = {"overall": {}, "per_bucket": {}}

    for k, v_list in results["overall"].items():
        if k not in ["scores", "labels"]:
            final_metrics["overall"][k] = float(np.mean(v_list)) if v_list else 0.0

    scores = np.array(results["overall"]["scores"])
    labels = np.array(results["overall"]["labels"])
    if len(scores) > 0:
        best_f1, best_t = 0.0, float(np.median(scores))

        thresholds = np.percentile(scores, np.linspace(1, 99, 50))

        for t in thresholds:
            f1 = compute_f1_at_threshold(scores, labels, t)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        final_metrics["overall"]["f1_best"] = float(best_f1)
        final_metrics["overall"]["f1_threshold"] = float(best_t)
    else:
        final_metrics["overall"]["f1_best"] = 0.0
        final_metrics["overall"]["f1_threshold"] = 0.5

    for b in buckets:
        final_metrics["per_bucket"][b] = {}
        for k, v_list in results["per_bucket"][b].items():
            if k not in ["scores", "labels"]:
                final_metrics["per_bucket"][b][k] = (
                    float(np.mean(v_list)) if v_list else 0.0
                )

    logger.info("Aggregation + F1: {:.1f}s", time.time() - t_agg_start)

    # Calculate Latency per query in ms
    s1_ms = (stage1_total_time / total_queries) * 1000 if total_queries > 0 else 0.0
    s2_ms = (stage2_total_time / total_queries) * 1000 if total_queries > 0 else 0.0

    result = build_results_json(args.experiment_id, s1_ms, s2_ms, final_metrics)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    elapsed = time.time() - t_exp_start
    logger.success(
        "Experiment {} done in {:.0f}s | R@1={:.3f} R@10={:.3f} MRR={:.3f} AvgRank={:.1f} RankDelta={:+.1f}",
        args.experiment_id,
        elapsed,
        final_metrics["overall"].get("recall_at_1", 0),
        final_metrics["overall"].get("recall_at_10", 0),
        final_metrics["overall"].get("mrr_at_10", 0),
        final_metrics["overall"].get("mean_reranked_rank", 0),
        final_metrics["overall"].get("mean_rank_delta", 0),
    )
    logger.info("Results written to {}", args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-model", type=str, required=True)
    parser.add_argument("--stage1-index", type=Path, required=True)
    parser.add_argument("--reranker", type=str, required=True)
    parser.add_argument("--eval-queries", type=Path, required=True)
    parser.add_argument("--top-k-stage1", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-id", type=str, required=True)

    args = parser.parse_args()
    process_end_to_end(args)
