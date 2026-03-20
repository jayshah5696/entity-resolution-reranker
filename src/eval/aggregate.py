import json
from pathlib import Path
import csv
from datetime import datetime, timezone

BUCKETS = [
    "pristine",
    "missing_firstname",
    "missing_email_company",
    "typo_name",
    "domain_mismatch",
    "swapped_attributes",
]

METRIC_KEYS = [
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "precision_at_5",
    "mrr_at_10",
    "ndcg_at_1",
    "ndcg_at_5",
    "ndcg_at_10",
]

PHASE2_KEYS = [
    "recall_at_50",
    "f1_best",
    "f1_threshold",
    "recall_retention"
]

def flatten_result(data: dict, source_file: str) -> dict:
    row: dict = {
        "experiment_id": data.get("experiment_id", ""),
        "source_file": source_file,
    }

    # Overall metrics
    overall = data.get("metrics", {}).get("overall", {})
    for key in METRIC_KEYS + PHASE2_KEYS:
        row[f"overall_{key}"] = overall.get(key, None)

    # Per-bucket metrics
    per_bucket = data.get("metrics", {}).get("per_bucket", {})
    for bucket in BUCKETS:
        bm = per_bucket.get(bucket, {})
        for key in METRIC_KEYS + PHASE2_KEYS:
            row[f"{bucket}_{key}"] = bm.get(key, None)

    # Latency
    latency = data.get("latency", {})
    row["latency_stage1_ms"] = latency.get("stage1_ms", None)
    row["latency_stage2_ms"] = latency.get("stage2_ms", None)

    return row

def write_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        output_path.write_text("")
        return

    rows_sorted = sorted(rows, key=lambda r: str(r.get("experiment_id", "")))

    base_fields = ["experiment_id", "source_file"]
    overall_fields = [f"overall_{m}" for m in METRIC_KEYS + PHASE2_KEYS]
    bucket_fields = [f"{b}_{m}" for b in BUCKETS for m in METRIC_KEYS + PHASE2_KEYS]
    latency_fields = ["latency_stage1_ms", "latency_stage2_ms"]

    fieldnames = base_fields + overall_fields + bucket_fields + latency_fields

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow(row)

def fmt_metric(val, digits: int = 3) -> str:
    if val is None: return "n/a"
    return f"{val:.{digits}f}"

def fmt_delta(val: float | None) -> str:
    if val is None: return "n/a"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val * 100:.1f}pp"

def write_report(rows: list[dict], output_path: Path) -> None:
    if not rows:
        output_path.write_text("# Experiment Results\n\nNo results found.\n")
        return

    rows_sorted = sorted(rows, key=lambda r: str(r.get("experiment_id", "")))
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    bm25_rows = [r for r in rows_sorted if "001" in str(r.get("experiment_id", ""))]
    bm25_row = bm25_rows[0] if bm25_rows else None

    lines: list[str] = []
    lines.append("# Phase 2 Experiment Results")
    lines.append(f"\nGenerated: {now_str}\n")
    
    # Section 1
    lines.append("## 1. Summary\n")
    lines.append("| Experiment | R@10 Overall | MRR@10 | S1 (ms) | S2 (ms) |")
    lines.append("|------------|:------------:|:------:|:-------:|:-------:|")
    for r in rows_sorted:
        lines.append(f"| {r.get('experiment_id', '')} | {fmt_metric(r.get('overall_recall_at_10'))} | {fmt_metric(r.get('overall_mrr_at_10'))} | {fmt_metric(r.get('latency_stage1_ms'), 1)} | {fmt_metric(r.get('latency_stage2_ms'), 1)} |")
    lines.append("")

    # Section 2
    lines.append("## 2. Per-Bucket Recall@10\n")
    header_cols = " | ".join(BUCKETS)
    lines.append(f"| Experiment | {header_cols} |")
    lines.append(f"|------------|{'|'.join([':---:']*len(BUCKETS))}|")
    for r in rows_sorted:
        b_vals = " | ".join(fmt_metric(r.get(f"{b}_recall_at_10")) for b in BUCKETS)
        lines.append(f"| {r.get('experiment_id', '')} | {b_vals} |")
    lines.append("")

    # Section 3
    lines.append("## 3. Delta vs Baseline (R@10)\n")
    if bm25_row is None:
        lines.append("No baseline (001) found.\n")
    else:
        lines.append(f"| Experiment | {header_cols} | Overall |")
        lines.append(f"|------------|{'|'.join([':---:']*len(BUCKETS))}|:-------:|")
        for r in rows_sorted:
            if r.get("experiment_id") == "001": continue
            d_cols = [fmt_delta(r.get(f"{b}_recall_at_10") - bm25_row.get(f"{b}_recall_at_10")) if r.get(f"{b}_recall_at_10") is not None and bm25_row.get(f"{b}_recall_at_10") is not None else "n/a" for b in BUCKETS]
            o_del = fmt_delta(r.get("overall_recall_at_10") - bm25_row.get("overall_recall_at_10")) if r.get("overall_recall_at_10") is not None and bm25_row.get("overall_recall_at_10") is not None else "n/a"
            lines.append(f"| {r.get('experiment_id')} | {' | '.join(d_cols)} | {o_del} |")
    lines.append("")
    
    # Section 4
    lines.append("## 4. Phase 2 specific metrics\n")
    lines.append("| Experiment | R@50 | F1 Best | F1 Threshold | Retention |")
    lines.append("|------------|:----:|:-------:|:------------:|:---------:|")
    for r in rows_sorted:
        lines.append(f"| {r.get('experiment_id', '')} | {fmt_metric(r.get('overall_recall_at_50'))} | {fmt_metric(r.get('overall_f1_best'))} | {fmt_metric(r.get('overall_f1_threshold'))} | {fmt_metric(r.get('overall_recall_retention'))} |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))

def aggregate_results(results_dir: Path):
    json_files = sorted(results_dir.glob("*.json"))
    if not json_files: return
    
    rows = []
    for jf in json_files:
        try:
            with open(jf) as f: data = json.load(f)
            rows.append(flatten_result(data, jf.name))
        except Exception as e:
            print(f"Failed {jf.name}: {e}")
            
    write_csv(rows, results_dir / "master.csv")
    write_report(rows, results_dir / "report.md")
    print(f"Aggregated {len(rows)} experiments.")

if __name__ == "__main__":
    aggregate_results(Path("results"))
