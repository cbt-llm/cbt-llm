"""
Threshold Analysis for MPNet SNOMED CT Retrieval
-------------------------------------------------
Produces statistical evidence for the choice of cosine similarity
threshold (0.35) used to filter retrieved SNOMED CT findings.

Outputs:
  - Console: descriptive stats, histogram, threshold impact table
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

THRESHOLD = 0.35


def load_mpnet_scores(path):
    df = pd.read_csv(path)
    mpnet = (
        df[df["Embedding"] == "mpnet"][["Turn", "User Text", "SNOMED Term", "Score"]]
        .drop_duplicates()
        .dropna()
    )
    mpnet["Score"] = mpnet["Score"].astype(float)
    return mpnet


def print_descriptive_stats(scores):
    print("=" * 55)
    print("  Descriptive Statistics — MPNet Cosine Similarity")
    print("=" * 55)
    print(f"  N              : {len(scores)}")
    print(f"  Min            : {scores.min():.4f}")
    print(f"  Max            : {scores.max():.4f}")
    print(f"  Mean           : {scores.mean():.4f}")
    print(f"  Median         : {np.median(scores):.4f}")
    print(f"  Std Dev        : {scores.std():.4f}")
    print(f"  Skewness       : {stats.skew(scores):.4f}")
    print(f"  Kurtosis       : {stats.kurtosis(scores):.4f}")
    print()
    print("  Percentiles:")
    for p in [10, 25, 50, 75, 90, 95]:
        print(f"    p{p:2d}           : {np.percentile(scores, p):.4f}")
    print()


def print_histogram(scores):
    bins = np.arange(0.15, 0.75, 0.05)
    counts, edges = np.histogram(scores, bins=bins)
    print("=" * 55)
    print("  Score Distribution (bin width = 0.05)")
    print("=" * 55)
    for i, c in enumerate(counts):
        marker = " ← threshold" if abs(edges[i] - THRESHOLD) < 0.001 else ""
        bar = "█" * (c // 4)
        print(f"  {edges[i]:.2f}–{edges[i+1]:.2f} | {bar:30s} {c:3d}{marker}")
    print()


def print_threshold_table(scores):
    total = len(scores)
    print("=" * 55)
    print("  Findings retained at candidate thresholds")
    print("=" * 55)
    print(f"  {'Threshold':>10}  {'Kept':>6}  {'%':>6}  {'Dropped':>8}  {'%':>6}")
    print(f"  {'-'*10}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*6}")
    for t in [0.25, 0.30, 0.33, 0.35, 0.38, 0.40, 0.45, 0.50]:
        kept = int((scores >= t).sum())
        dropped = total - kept
        marker = " ◄" if t == THRESHOLD else ""
        print(f"  {t:>10.2f}  {kept:>6}  {100*kept/total:>5.1f}%  {dropped:>8}  {100*dropped/total:>5.1f}%{marker}")
    print()


def plot_distribution(scores, output_path, n_conversations, n_turns):
    n_retained = int((scores >= THRESHOLD).sum())
    n_total = len(scores)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(
        "MPNet Cosine Similarity Distribution — SNOMED CT Retrieval",
        fontsize=13, fontweight="bold"
    )
    fig.text(
        0.5, 0.93,
        (f"{n_conversations} CBT conversations  ·  {n_turns} patient turns  ·  "
         f"{n_total} SNOMED findings retrieved  ·  {n_retained} retained (threshold = {THRESHOLD})"),
        ha="center", fontsize=9, color="#555555"
    )

    bins = np.arange(0.15, 0.75, 0.05)
    counts, edges = np.histogram(scores, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2

    # --- Left: Histogram with threshold line ---
    ax1 = axes[0]
    bar_colors = ["#d62728" if e < THRESHOLD else "#2196F3" for e in edges[:-1]]
    ax1.bar(centers, counts, width=0.045, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax1.axvline(THRESHOLD, color="black", linestyle="--", linewidth=1.5,
                label=f"Threshold = {THRESHOLD}")
    ax1.axvline(np.mean(scores), color="orange", linestyle=":", linewidth=1.5,
                label=f"Mean = {np.mean(scores):.3f}")
    ax1.axvline(np.median(scores), color="green", linestyle=":", linewidth=1.5,
                label=f"Median = {np.median(scores):.3f}")
    ax1.set_xlabel("Cosine Similarity Score", fontsize=11)
    ax1.set_ylabel("Number of Findings", fontsize=11)
    ax1.set_title("Score Distribution with Threshold", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Annotate dropped vs kept
    ax1.fill_betweenx([0, max(counts) * 1.05], 0.15, THRESHOLD,
                      alpha=0.08, color="red", label="dropped")
    ax1.fill_betweenx([0, max(counts) * 1.05], THRESHOLD, 0.75,
                      alpha=0.08, color="blue", label="kept")
    ax1.text(0.22, max(counts) * 0.9, "Dropped\n(below threshold)",
             color="#d62728", fontsize=8, ha="center")
    ax1.text(0.55, max(counts) * 0.9, "Kept\n(above threshold)",
             color="#2196F3", fontsize=8, ha="center")

    # --- Right: Cumulative % retained at each threshold ---
    ax2 = axes[1]
    thresholds = np.arange(0.15, 0.72, 0.01)
    total = len(scores)
    retained_pct = [(scores >= t).sum() / total * 100 for t in thresholds]

    ax2.plot(thresholds, retained_pct, color="#2196F3", linewidth=2)
    ax2.axvline(THRESHOLD, color="black", linestyle="--", linewidth=1.5,
                label=f"Threshold = {THRESHOLD}")
    threshold_pct = (scores >= THRESHOLD).sum() / total * 100
    ax2.scatter([THRESHOLD], [threshold_pct], color="black", zorder=5, s=60)
    ax2.annotate(f"{threshold_pct:.1f}% retained",
                 xy=(THRESHOLD, threshold_pct),
                 xytext=(THRESHOLD + 0.04, threshold_pct + 5),
                 fontsize=9, arrowprops=dict(arrowstyle="->", color="black"))
    ax2.set_xlabel("Cosine Similarity Threshold", fontsize=11)
    ax2.set_ylabel("% Findings Retained", fontsize=11)
    ax2.set_title("Retention Rate vs. Threshold", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 105)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.text(0.97, 0.97,
             f"{n_conversations} conversations\n{n_turns} patient turns\n{n_total} retrieved\n{n_retained} retained",
             transform=ax2.transAxes, fontsize=8, color="#555555",
             va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"))

    plt.tight_layout(rect=[0, 0, 1, 0.91])
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
    scores = mpnet["Score"].values
    n_turns = mpnet["Turn"].nunique()
    n_conversations = count_conversations(CBT_DATA_DIR)

    print_descriptive_stats(scores)
    print_histogram(scores)
    print_threshold_table(scores)

    # Normality test
    stat, p = stats.shapiro(scores[:50])  # Shapiro-Wilk on sample
    print("=" * 55)
    print("  Shapiro-Wilk normality test (n=50 sample)")
    print("=" * 55)
    print(f"  W = {stat:.4f},  p = {p:.4f}")
    print(f"  Distribution is {'normal' if p > 0.05 else 'non-normal'} (α=0.05)")
    print()

    plot_distribution(scores, OUTPUT_PLOT, n_conversations, n_turns)


if __name__ == "__main__":
    main()
