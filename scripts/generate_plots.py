"""Generate research-quality plots for entity resolution reranker experiments."""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Style
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

COLORS = {
    "bm25": "#5B8FB9",
    "dense": "#B6533A",
    "bm25_light": "#8BBBDB",
    "dense_light": "#D4836F",
    "highlight": "#E8A838",
    "degraded": "#C0392B",
    "neutral": "#7F8C8D",
}

RERANKER_LABELS = {
    "minilm_reranker": "MiniLM",
    "minilm_reranker_ft": "MiniLM-FT",
    "bge_reranker_m3": "BGE-M3",
    "bge_reranker_m3_ft": "BGE-M3-FT",
    "gte_reranker": "GTE",
    "gte_reranker_ft": "GTE-FT",
    "granite_reranker": "Granite",
    "granite_reranker_ft": "Granite-FT",
}

RERANKER_ORDER = list(RERANKER_LABELS.keys())

BUCKETS = [
    "pristine",
    "typo_name",
    "domain_mismatch",
    "swapped_attributes",
    "missing_firstname",
    "missing_email_company",
]

BUCKET_LABELS = {
    "pristine": "Pristine",
    "typo_name": "Typo (Name)",
    "domain_mismatch": "Domain Mismatch",
    "swapped_attributes": "Swapped Attrs",
    "missing_firstname": "Missing First Name",
    "missing_email_company": "Missing Email+Co",
}


def load_results():
    """Load all experiment JSON results."""
    experiments = []
    for f in sorted(os.listdir(RESULTS_DIR)):
        if f.endswith(".json") and f[:3].isdigit() and "int8_baseline" not in f:
            with open(RESULTS_DIR / f) as fp:
                r = json.load(fp)
            o = r["metrics"]["overall"]
            pb = r["metrics"].get("per_bucket", {})

            # Parse stage1 and reranker from filename
            name = f.replace(".json", "")
            parts = name.split("_plus_")
            s1_raw = parts[0].split("_", 1)[1] if len(parts) > 0 else ""
            s2_raw = parts[1] if len(parts) > 1 else ""

            s1_label = "BM25" if "bm25" in s1_raw else "Dense-FT-Int8"

            experiments.append(
                {
                    "exp_id": r["experiment_id"],
                    "s1": s1_label,
                    "s2": s2_raw,
                    "s2_label": RERANKER_LABELS.get(s2_raw, s2_raw),
                    "R@1": o.get("recall_at_1", 0),
                    "R@5": o.get("recall_at_5", 0),
                    "R@10": o.get("recall_at_10", 0),
                    "R@50": o.get("recall_at_50", 0),
                    "MRR@10": o.get("mrr_at_10", 0),
                    "nDCG@10": o.get("ndcg_at_10", 0),
                    "AvgRank": o.get("mean_reranked_rank", 0),
                    "RankDelta": o.get("mean_rank_delta", 0),
                    "per_bucket": pb,
                }
            )
    return experiments


def plot_1_overall_recall_grouped(exps):
    """Grouped bar chart: R@10 by reranker, grouped by Stage1."""
    fig, ax = plt.subplots(figsize=(10, 5))

    rerankers = RERANKER_ORDER
    x = np.arange(len(rerankers))
    w = 0.35

    bm25_vals = []
    dense_vals = []
    for rk in rerankers:
        bm25_v = next(
            (e["R@10"] for e in exps if e["s1"] == "BM25" and e["s2"] == rk), 0
        )
        dense_v = next(
            (e["R@10"] for e in exps if e["s1"] == "Dense-FT-Int8" and e["s2"] == rk), 0
        )
        bm25_vals.append(bm25_v)
        dense_vals.append(dense_v)

    bars1 = ax.bar(
        x - w / 2,
        bm25_vals,
        w,
        label="BM25",
        color=COLORS["bm25"],
        edgecolor="white",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x + w / 2,
        dense_vals,
        w,
        label="Dense-FT-Int8",
        color=COLORS["dense"],
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_ylabel("Recall@10")
    ax.set_title("Recall@10 by Cross-Encoder Reranker")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [RERANKER_LABELS[rk] for rk in rerankers], rotation=30, ha="right"
    )
    ax.set_ylim(0.94, 0.98)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.legend(loc="lower right")

    # Annotate values
    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0005,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color=COLORS["bm25"],
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0005,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color=COLORS["dense"],
        )

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "overall_recall_at_10.png")
    plt.close(fig)
    print(f"  Saved overall_recall_at_10.png")


def plot_2_mrr_grouped(exps):
    """Grouped bar chart: MRR@10 by reranker."""
    fig, ax = plt.subplots(figsize=(10, 5))

    rerankers = RERANKER_ORDER
    x = np.arange(len(rerankers))
    w = 0.35

    bm25_vals = [
        next((e["MRR@10"] for e in exps if e["s1"] == "BM25" and e["s2"] == rk), 0)
        for rk in rerankers
    ]
    dense_vals = [
        next(
            (e["MRR@10"] for e in exps if e["s1"] == "Dense-FT-Int8" and e["s2"] == rk),
            0,
        )
        for rk in rerankers
    ]

    ax.bar(
        x - w / 2,
        bm25_vals,
        w,
        label="BM25",
        color=COLORS["bm25"],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.bar(
        x + w / 2,
        dense_vals,
        w,
        label="Dense-FT-Int8",
        color=COLORS["dense"],
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_ylabel("MRR@10")
    ax.set_title("Mean Reciprocal Rank@10 by Cross-Encoder Reranker")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [RERANKER_LABELS[rk] for rk in rerankers], rotation=30, ha="right"
    )
    ax.set_ylim(0.90, 0.94)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "mrr_at_10.png")
    plt.close(fig)
    print(f"  Saved mrr_at_10.png")


def plot_3_bucket_heatmap(exps):
    """Heatmap: R@10 by bucket for selected experiments."""
    selected = ["001", "003", "009", "011", "013", "015"]
    sel_exps = [e for e in exps if e["exp_id"] in selected]
    sel_exps.sort(key=lambda e: selected.index(e["exp_id"]))

    labels = [f"{e['exp_id']} {e['s1']}+{e['s2_label']}" for e in sel_exps]
    data = []
    for e in sel_exps:
        row = []
        for b in BUCKETS:
            val = e["per_bucket"].get(b, {}).get("recall_at_10", 0)
            row.append(val)
        data.append(row)

    data = np.array(data)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.heatmap(
        data,
        annot=True,
        fmt=".3f",
        xticklabels=[BUCKET_LABELS[b] for b in BUCKETS],
        yticklabels=labels,
        cmap="RdYlGn",
        vmin=0.7,
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        annot_kws={"size": 9},
    )
    ax.set_title("Recall@10 by Query Corruption Type")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "bucket_heatmap_r10.png")
    plt.close(fig)
    print(f"  Saved bucket_heatmap_r10.png")


def plot_4_bucket_heatmap_mrr(exps):
    """Heatmap: MRR@10 by bucket."""
    selected = ["001", "003", "009", "011", "013", "015"]
    sel_exps = [e for e in exps if e["exp_id"] in selected]
    sel_exps.sort(key=lambda e: selected.index(e["exp_id"]))

    labels = [f"{e['exp_id']} {e['s1']}+{e['s2_label']}" for e in sel_exps]
    data = []
    for e in sel_exps:
        row = []
        for b in BUCKETS:
            val = e["per_bucket"].get(b, {}).get("mrr_at_10", 0)
            row.append(val)
        data.append(row)

    data = np.array(data)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.heatmap(
        data,
        annot=True,
        fmt=".3f",
        xticklabels=[BUCKET_LABELS[b] for b in BUCKETS],
        yticklabels=labels,
        cmap="RdYlGn",
        vmin=0.4,
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        annot_kws={"size": 9},
    )
    ax.set_title("MRR@10 by Query Corruption Type")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "bucket_heatmap_mrr.png")
    plt.close(fig)
    print(f"  Saved bucket_heatmap_mrr.png")


def plot_5_rank_delta(exps):
    """Bar chart showing rank delta (reranker impact) -- only for dense experiments."""
    dense_exps = [
        e for e in exps if e["s1"] == "Dense-FT-Int8" and e.get("RankDelta", 0) != 0
    ]
    dense_exps.sort(
        key=lambda e: RERANKER_ORDER.index(e["s2"]) if e["s2"] in RERANKER_ORDER else 99
    )

    if not dense_exps:
        # All rank deltas for dense, including ~0 ones
        dense_exps = [e for e in exps if e["s1"] == "Dense-FT-Int8"]
        dense_exps.sort(
            key=lambda e: (
                RERANKER_ORDER.index(e["s2"]) if e["s2"] in RERANKER_ORDER else 99
            )
        )

    labels = [e["s2_label"] for e in dense_exps]
    deltas = [e.get("RankDelta", 0) for e in dense_exps]
    colors = [
        COLORS["degraded"]
        if d < -0.05
        else COLORS["highlight"]
        if d > 0.05
        else COLORS["neutral"]
        for d in deltas
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(labels, deltas, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Mean Rank Delta (positive = reranker helps)")
    ax.set_title("Reranker Impact on True-Match Ranking (Dense-FT-Int8 Stage 1)")
    ax.axvline(0, color="black", linewidth=0.8)

    for bar, d in zip(bars, deltas):
        x_pos = bar.get_width() - 0.005 if d < 0 else bar.get_width() + 0.005
        ha = "right" if d < 0 else "left"
        ax.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2,
            f"{d:+.3f}",
            ha=ha,
            va="center",
            fontsize=9,
        )

    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "rank_delta.png")
    plt.close(fig)
    print(f"  Saved rank_delta.png")


def plot_6_stage1_comparison(exps):
    """Side-by-side comparison of Stage 1 retrievers (BM25 vs Dense-FT-Int8)."""
    metrics = ["R@1", "R@5", "R@10", "R@50", "MRR@10", "nDCG@10"]
    metric_labels = ["R@1", "R@5", "R@10", "R@50", "MRR@10", "nDCG@10"]

    # Use the best reranker for each stage1 (Granite for BM25, MiniLM for Dense)
    bm25_best = next((e for e in exps if e["exp_id"] == "007"), None)
    dense_best = next((e for e in exps if e["exp_id"] == "009"), None)

    if not bm25_best or not dense_best:
        return

    bm25_vals = [bm25_best[m] for m in metrics]
    dense_vals = [dense_best[m] for m in metrics]

    x = np.arange(len(metrics))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        x - w / 2,
        bm25_vals,
        w,
        label="BM25 + Granite",
        color=COLORS["bm25"],
        edgecolor="white",
    )
    ax.bar(
        x + w / 2,
        dense_vals,
        w,
        label="Dense-FT-Int8 + MiniLM",
        color=COLORS["dense"],
        edgecolor="white",
    )

    ax.set_ylabel("Score")
    ax.set_title("Best Configuration: BM25 vs Dense-FT-Int8 Stage 1")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0.88, 1.0)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.legend()

    # Annotate deltas
    for i, (b, d) in enumerate(zip(bm25_vals, dense_vals)):
        delta = d - b
        if abs(delta) > 0.001:
            ax.annotate(
                f"+{delta:.3f}",
                xy=(x[i] + w / 2, d),
                xytext=(0, 12),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=COLORS["dense"],
                fontweight="bold",
            )

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "stage1_comparison.png")
    plt.close(fig)
    print(f"  Saved stage1_comparison.png")


def plot_7_radar_bucket(exps):
    """Radar chart: per-bucket R@10 for BM25 vs Dense best configs."""
    bm25 = next((e for e in exps if e["exp_id"] == "001"), None)
    dense = next((e for e in exps if e["exp_id"] == "009"), None)

    if not bm25 or not dense:
        return

    categories = [BUCKET_LABELS[b] for b in BUCKETS]
    N = len(categories)

    bm25_vals = [bm25["per_bucket"].get(b, {}).get("recall_at_10", 0) for b in BUCKETS]
    dense_vals = [
        dense["per_bucket"].get(b, {}).get("recall_at_10", 0) for b in BUCKETS
    ]

    # Close the radar
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    bm25_vals += bm25_vals[:1]
    dense_vals += dense_vals[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.fill(angles, bm25_vals, alpha=0.15, color=COLORS["bm25"])
    ax.plot(
        angles,
        bm25_vals,
        "o-",
        color=COLORS["bm25"],
        label="BM25 + MiniLM",
        linewidth=2,
        markersize=5,
    )
    ax.fill(angles, dense_vals, alpha=0.15, color=COLORS["dense"])
    ax.plot(
        angles,
        dense_vals,
        "s-",
        color=COLORS["dense"],
        label="Dense-FT + MiniLM",
        linewidth=2,
        markersize=5,
    )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=9)
    ax.set_ylim(0.7, 1.02)
    ax.set_title("R@10 by Query Corruption Type", pad=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1.15, -0.05))

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "radar_bucket_r10.png")
    plt.close(fig)
    print(f"  Saved radar_bucket_r10.png")


def plot_8_bge_degradation(exps):
    """Show BGE-M3 degradation across buckets compared to best."""
    best = next((e for e in exps if e["exp_id"] == "009"), None)
    bge = next((e for e in exps if e["exp_id"] == "011"), None)
    bge_ft = next((e for e in exps if e["exp_id"] == "012"), None)

    if not all([best, bge, bge_ft]):
        return

    x = np.arange(len(BUCKETS))
    w = 0.25

    best_vals = [best["per_bucket"].get(b, {}).get("recall_at_10", 0) for b in BUCKETS]
    bge_vals = [bge["per_bucket"].get(b, {}).get("recall_at_10", 0) for b in BUCKETS]
    bge_ft_vals = [
        bge_ft["per_bucket"].get(b, {}).get("recall_at_10", 0) for b in BUCKETS
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        x - w,
        best_vals,
        w,
        label="MiniLM (best)",
        color=COLORS["bm25"],
        edgecolor="white",
    )
    ax.bar(
        x,
        bge_vals,
        w,
        label="BGE-M3 (stock)",
        color=COLORS["degraded"],
        edgecolor="white",
    )
    ax.bar(
        x + w,
        bge_ft_vals,
        w,
        label="BGE-M3 (fine-tuned)",
        color=COLORS["highlight"],
        edgecolor="white",
    )

    ax.set_ylabel("Recall@10")
    ax.set_title("BGE-M3 Degrades Rankings vs MiniLM Baseline (Dense-FT-Int8 Stage 1)")
    ax.set_xticks(x)
    ax.set_xticklabels([BUCKET_LABELS[b] for b in BUCKETS], rotation=25, ha="right")
    ax.set_ylim(0.82, 1.005)
    ax.legend()

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "bge_degradation.png")
    plt.close(fig)
    print(f"  Saved bge_degradation.png")


def plot_9_multi_metric(exps):
    """Multi-panel: R@1, R@10, MRR@10, nDCG@10 for all 8 dense experiments."""
    dense_exps = [e for e in exps if e["s1"] == "Dense-FT-Int8"]
    dense_exps.sort(
        key=lambda e: RERANKER_ORDER.index(e["s2"]) if e["s2"] in RERANKER_ORDER else 99
    )

    metrics = ["R@1", "R@10", "MRR@10", "nDCG@10"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, metric in zip(axes.flat, metrics):
        vals = [e[metric] for e in dense_exps]
        labels = [e["s2_label"] for e in dense_exps]
        colors_list = [
            COLORS["degraded"]
            if e["s2"] in ("bge_reranker_m3",)
            else COLORS["highlight"]
            if e["s2"] in ("bge_reranker_m3_ft",)
            else COLORS["bm25"]
            for e in dense_exps
        ]

        bars = ax.bar(labels, vals, color=colors_list, edgecolor="white", linewidth=0.5)
        ax.set_title(metric, fontweight="bold")
        ax.set_ylim(min(vals) - 0.01, max(vals) + 0.005)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.tick_params(axis="x", rotation=35)

        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.0005,
                f"{v:.4f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    fig.suptitle(
        "Dense-FT-Int8 Stage 1: Cross-Encoder Comparison",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "dense_multi_metric.png")
    plt.close(fig)
    print(f"  Saved dense_multi_metric.png")


def main():
    print("Loading results...")
    exps = load_results()
    print(f"  Loaded {len(exps)} experiments")

    print("Generating plots...")
    plot_1_overall_recall_grouped(exps)
    plot_2_mrr_grouped(exps)
    plot_3_bucket_heatmap(exps)
    plot_4_bucket_heatmap_mrr(exps)
    plot_5_rank_delta(exps)
    plot_6_stage1_comparison(exps)
    plot_7_radar_bucket(exps)
    plot_8_bge_degradation(exps)
    plot_9_multi_metric(exps)
    print("\nAll plots saved to results/plots/")


if __name__ == "__main__":
    main()
