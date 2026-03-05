"""
Threshold Analysis for MPNet SNOMED CT Retrieval
-------------------------------------------------
Analyses the per-query adaptive threshold strategy:
for each patient query, only SNOMED findings that score at or above
the mean score of that query's top-K retrieved results are retained.

Outputs:
  - Console: global stats, per-query mean distribution, retention counts
  - src/output_files/neo4j_retrival_output/threshold_analysis.png
"""

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

INPUT_FILE = ROOT / "src" / "output_files" / "neo4j_retrival_output" / "snomed_turn_results.csv"
OUTPUT_PLOT = ROOT / "src" / "output_files" / "neo4j_retrival_output" / "threshold_analysis.png"
CBT_DATA_DIR = ROOT / "src" / "cbt_llm" / "cbt_user_data"

def load_mpnet_scores(path):
    df = pd.read_csv(path)
    mpnet = (
        df[df["Embedding"] == "mpnet"][["Turn", "User Text", "SNOMED Term", "Score"]]
        .drop_duplicates()
        .dropna()
    )
    mpnet["Score"] = mpnet["Score"].astype(float)
    return mpnet


def compute_per_query_stats(mpnet):
    """For each query (Turn), compute mean score and which findings are retained."""
    rows = []
    for turn, group in mpnet.groupby("Turn"):
        scores = group["Score"].values
        mean_score = scores.mean()
        retained = (scores >= mean_score).sum()
        rows.append({
            "Turn": turn,
            "User Text": group["User Text"].iloc[0],
            "N Retrieved": len(scores),
            "Mean Threshold": mean_score,
            "Min Score": scores.min(),
            "Max Score": scores.max(),
            "N Retained": retained,
            "N Dropped": len(scores) - retained,
        })
    return pd.DataFrame(rows)


def print_descriptive_stats(mpnet, per_query):
    scores = mpnet["Score"].values
    thresholds = per_query["Mean Threshold"].values

    print("=" * 55)
    print("  Global Score Statistics — MPNet Cosine Similarity")
    print("=" * 55)
    print(f"  Total findings : {len(scores)}")
    print(f"  Min            : {scores.min():.4f}")
    print(f"  Max            : {scores.max():.4f}")
    print(f"  Mean           : {scores.mean():.4f}")
    print(f"  Median         : {np.median(scores):.4f}")
    print(f"  Std Dev        : {scores.std():.4f}")
    print(f"  Skewness       : {stats.skew(scores):.4f}")
    print()
    print("=" * 55)
    print("  Per-Query Adaptive Threshold Distribution")
    print("  (threshold = mean score of that query's top-K)")
    print("=" * 55)
    print(f"  Min threshold  : {thresholds.min():.4f}")
    print(f"  Max threshold  : {thresholds.max():.4f}")
    print(f"  Mean threshold : {thresholds.mean():.4f}")
    print(f"  Median         : {np.median(thresholds):.4f}")
    print(f"  Std Dev        : {thresholds.std():.4f}")
    print()
    total_retained = per_query["N Retained"].sum()
    total = len(scores)
    print(f"  Total retained : {total_retained} / {total} ({100*total_retained/total:.1f}%)")
    print(f"  Total dropped  : {total - total_retained} / {total} ({100*(total-total_retained)/total:.1f}%)")
    print()


def print_per_query_table(per_query):
    print("=" * 75)
    print("  Per-Query Adaptive Thresholds (sample)")
    print("=" * 75)
    print(f"  {'Turn':>5}  {'Threshold':>10}  {'Retrieved':>10}  {'Retained':>9}  {'Dropped':>8}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*9}  {'-'*8}")
    for _, row in per_query.head(15).iterrows():
        print(f"  {int(row['Turn']):>5}  {row['Mean Threshold']:>10.4f}  "
              f"{int(row['N Retrieved']):>10}  {int(row['N Retained']):>9}  {int(row['N Dropped']):>8}")
    print()


def plot_example_query(output_path):
    """Illustrative example: top-5 retrievals for one query, mean threshold shown."""
    findings = [
        "Psychophysiologic\ninsomnia (disorder)",
        "Asleep\n(finding)",
        "Wakefulness finding\n(finding)",
        "Ready for enhanced\nsleep pattern (finding)",
        "Drowsy\n(finding)",
    ]
    scores = [0.4499, 0.4170, 0.4107, 0.3470, 0.3176]
    mean_score = sum(scores) / len(scores)
    colors = ["#2196F3" if s >= mean_score else "#d62728" for s in scores]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(findings[::-1], scores[::-1], color=colors[::-1],
                   edgecolor="white", height=0.55)

    ax.axvline(mean_score, color="black", linestyle="--", linewidth=1.8,
               label=f"Mean threshold = {mean_score:.4f}")

    for bar, score in zip(bars, scores[::-1]):
        ax.text(score + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}", va="center", fontsize=9)

    ax.set_xlabel("Cosine Similarity Score", fontsize=11)
    ax.set_title(
        'Example: Top-5 Retrieved SNOMED Findings\n'
        'Query: "If I can\'t fall asleep, then I need to get out of bed and stop trying to sleep?"',
        fontsize=11
    )
    ax.legend(fontsize=10)
    ax.set_xlim(0, 0.52)

    # Labels for kept / dropped — placed below x-axis
    ax.text(mean_score + 0.003, -0.12, "KEPT ▶", transform=ax.get_xaxis_transform(),
            fontsize=9, color="#2196F3", fontweight="bold", va="top")
    ax.text(mean_score - 0.003, -0.12, "◀ DROPPED", transform=ax.get_xaxis_transform(),
            fontsize=9, color="#d62728", fontweight="bold", va="top", ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved: {output_path}")
    plt.close()


def count_conversations(data_dir):
    files = glob.glob(str(data_dir / "*.csv"))
    all_ids = []
    for f in files:
        all_ids.extend(pd.read_csv(f)["id"].tolist())
    unique_convos = {"_".join(id_.split("_")[:2]) for id_ in all_ids}
    return len(unique_convos)


def main():
    mpnet = load_mpnet_scores(INPUT_FILE)
    per_query = compute_per_query_stats(mpnet)

    print_descriptive_stats(mpnet, per_query)
    print_per_query_table(per_query)
    plot_example_query(OUTPUT_PLOT)


if __name__ == "__main__":
    main()
