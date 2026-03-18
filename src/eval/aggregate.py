import json
from pathlib import Path
import csv

def validate_results_schema(results: dict):
    required_top = {"experiment_id", "latency", "metrics"}
    if not required_top.issubset(results.keys()):
        raise ValueError(f"Missing top-level keys. Found: {results.keys()}")
        
    required_metrics = {"overall", "per_bucket"}
    if not required_metrics.issubset(results["metrics"].keys()):
        raise ValueError(f"Missing metrics keys. Found: {results['metrics'].keys()}")

def aggregate_results(results_dir: Path):
    results_dir.mkdir(exist_ok=True)
    json_files = sorted(results_dir.glob("*.json"))
    
    if not json_files:
        print("No result JSONs found in", results_dir)
        return
        
    all_results = []
    for jf in json_files:
        try:
            with open(jf, "r") as f:
                data = json.load(f)
            validate_results_schema(data)
            all_results.append(data)
        except Exception as e:
            print(f"Skipping {jf}: {e}")
            
    if not all_results:
        return
        
    # Write master.csv
    csv_path = results_dir / "master.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment_id", "recall_at_10", "recall_at_50", "mrr", 
            "ndcg_at_10", "f1_best", "f1_threshold", "recall_retention",
            "stage1_ms", "stage2_ms"
        ])
        for r in all_results:
            m = r["metrics"]["overall"]
            l = r["latency"]
            writer.writerow([
                r["experiment_id"],
                m.get("recall_at_10", 0),
                m.get("recall_at_50", 0),
                m.get("mrr", 0),
                m.get("ndcg_at_10", 0),
                m.get("f1_best", 0),
                m.get("f1_threshold", 0),
                m.get("recall_retention", 0),
                l.get("stage1_ms", 0),
                l.get("stage2_ms", 0)
            ])
            
    # Write report.md
    md_path = results_dir / "report.md"
    with open(md_path, "w") as f:
        f.write("# Entity Resolution Phase 2 Results\n\n")
        f.write("| Experiment | R@10 | MRR | nDCG@10 | F1 | Stage 1 (ms) | Stage 2 (ms) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in all_results:
            m = r["metrics"]["overall"]
            l = r["latency"]
            f.write(f"| {r['experiment_id']} | {m.get('recall_at_10', 0):.3f} | {m.get('mrr', 0):.3f} | {m.get('ndcg_at_10', 0):.3f} | {m.get('f1_best', 0):.3f} | {l.get('stage1_ms', 0):.1f} | {l.get('stage2_ms', 0):.1f} |\n")
            
    print(f"Aggregated {len(all_results)} experiments into {csv_path} and {md_path}")

if __name__ == "__main__":
    aggregate_results(Path("results"))
