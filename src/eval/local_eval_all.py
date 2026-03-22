"""
Local evaluation runner for all 16 experiments.

Replaces eval_modal.py for running without Modal credits.
Uses precomputed Stage 1 candidates from results/ directory.
"""

import json
import sys
import time
from argparse import Namespace
from pathlib import Path

EVAL_QUERIES_DIR = Path("../entity-resolution-poc/data/eval")
BM25_CANDIDATES = Path("results/bm25_candidates.parquet")
DENSE_CANDIDATES = Path("results/dense_candidates.parquet")
EXPERIMENTS_DIR = Path("experiments")
RESULTS_DIR = Path("results")


def run_experiment(exp_cfg: dict) -> None:
    """Run a single experiment using precomputed Stage 1 candidates."""
    # Lazy imports so we pay the cost once per model, not at script start
    from src.eval.run_reranker import process_end_to_end

    exp_id = exp_cfg["experiment_id"]
    s1_model = exp_cfg["stage1_model"]
    s2_key = exp_cfg["reranker"]
    s2_path = exp_cfg["reranker_path"]

    print(f"\n{'=' * 60}")
    print(f"Experiment {exp_id}: Stage1={s1_model} | Stage2={s2_path}")
    print(f"{'=' * 60}")

    if s1_model == "bm25":
        s1_index = Path("../entity-resolution-poc/results/indexes/bm25_pipe")
        candidates_path = BM25_CANDIDATES
    else:
        s1_index = Path(
            "../entity-resolution-poc/results/indexes/gte_modernbert_base_pipe_fp32"
        )
        candidates_path = DENSE_CANDIDATES

    if not candidates_path.exists():
        print(
            f"  SKIP: {candidates_path} not found. Run local_stage1_precompute.py first."
        )
        return

    # Dynamically patch models.yaml so run_reranker can resolve both
    # stock and fine-tuned model keys
    import yaml

    configs_path = Path("configs/models.yaml")
    with open(configs_path) as f:
        original_cfg = yaml.safe_load(f)

    # Ensure the reranker key maps to the correct HF model path
    patched_cfg = dict(original_cfg)
    patched_cfg[s2_key] = {"hf_id": s2_path}

    with open(configs_path, "w") as f:
        yaml.dump(patched_cfg, f)

    output_name = f"{exp_id}_{s1_model}_plus_{s2_key}.json"
    output_path = RESULTS_DIR / output_name

    try:
        args = Namespace(
            stage1_model=s1_model,
            stage1_index=s1_index,
            reranker=s2_key,
            eval_queries=EVAL_QUERIES_DIR,
            top_k_stage1=50,
            output=output_path,
            experiment_id=exp_id,
            precomputed_candidates=candidates_path,
        )
        process_end_to_end(args)

        # Print summary
        with open(output_path) as f:
            result = json.load(f)
        overall = result.get("metrics", {}).get("overall", {})
        print(
            f"\n  R@1={overall.get('recall_at_1', 0):.3f}  "
            f"R@10={overall.get('recall_at_10', 0):.3f}  "
            f"R@50={overall.get('recall_at_50', 0):.3f}  "
            f"MRR@10={overall.get('mrr_at_10', 0):.3f}  "
            f"F1={overall.get('f1_best', 0):.3f}"
        )

    finally:
        # Restore original models.yaml
        with open(configs_path, "w") as f:
            yaml.dump(original_cfg, f)


def main() -> None:
    # Load all experiment configs
    configs = []
    for f in sorted(EXPERIMENTS_DIR.glob("*/config.json")):
        with open(f) as fp:
            configs.append(json.load(fp))

    # Allow filtering by experiment IDs via CLI args (preserves CLI order)
    if len(sys.argv) > 1:
        requested = sys.argv[1:]
        id_to_cfg = {c["experiment_id"]: c for c in configs}
        configs = [id_to_cfg[eid] for eid in requested if eid in id_to_cfg]

    if not configs:
        print("No experiments to run.")
        return

    print(f"Running {len(configs)} experiments locally...")
    print(f"BM25 candidates: {BM25_CANDIDATES} (exists={BM25_CANDIDATES.exists()})")
    print(f"Dense candidates: {DENSE_CANDIDATES} (exists={DENSE_CANDIDATES.exists()})")

    RESULTS_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    for cfg in configs:
        try:
            run_experiment(cfg)
        except Exception as e:
            print(f"\n  FAILED exp {cfg['experiment_id']}: {e}")
            import traceback

            traceback.print_exc()

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"All experiments complete in {elapsed / 60:.1f} minutes.")
    print(f"Run `uv run python -m src.eval.aggregate` to generate the report.")


if __name__ == "__main__":
    main()
