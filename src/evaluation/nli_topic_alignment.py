"""
NLI Topic Alignment for CBT-LLM sessions.

For each turn in a transcript, the selected LLM response is scored using
bidirectional NLI entailment against the transcript's own core issue.

Hypothesis: "This response addresses [core_issue]"
Premise:    the selected LLM response

Score = entailment probability (0-1)
  High = model is addressing the core issue
  Low  = model has drifted from the core issue

Distraction injection turns are marked with red dashed lines + triangle markers.
One plot per transcript. Works across transcripts with different core issues
since each transcript is scored against its own core issue.
"""

import glob
import json
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, LinearSegmentedColormap
from sentence_transformers import CrossEncoder

# ── Model ─────────────────────────────────────────────────────────────────────

NLI_MODEL  = "cross-encoder/nli-deberta-v3-small"
ENTAIL_IDX = 1  # deberta label order: contradiction=0, entailment=1, neutral=2

print("Loading NLI model...")
nli = CrossEncoder(NLI_MODEL)
print("Model ready.")

# ── Scoring ───────────────────────────────────────────────────────────────────

def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def nli_score(response, core_issue):
    """Entailment probability that response addresses the core issue."""
    if not response or not response.strip():
        return np.nan
    hypothesis = f"This response addresses {core_issue.lower()}."
    scores     = nli.predict([(response, hypothesis)])
    return float(_softmax(scores[0])[ENTAIL_IDX])

# ── File processing ───────────────────────────────────────────────────────────

def process_file(filepath):
    with open(filepath) as f:
        data = json.load(f)

    meta       = data["metadata"]
    core_issue = meta["core_issue"]
    model_name = meta["llm_response"]

    turns, scores, dist_flags = [], [], []

    for td in data["transcript"]:
        turn_num       = td["turn"]
        response       = td.get("llm_response", {}).get("response", "")
        is_distraction = td.get("patient", {}).get("distraction_injected", False)

        turns.append(turn_num)
        scores.append(nli_score(response, core_issue))
        dist_flags.append(is_distraction)

    dist_turns = [t for t, d in zip(turns, dist_flags) if d]

    mean_clean = np.nanmean([s for s, d in zip(scores, dist_flags) if not d])
    mean_dist  = np.nanmean([s for s, d in zip(scores, dist_flags) if d])

    print(f"    core_issue     : {core_issue}")
    print(f"    mean (clean)   : {mean_clean:.3f}")
    print(f"    mean (distract): {mean_dist:.3f}")
    print(f"    delta          : {mean_clean - mean_dist:.3f}")

    return {
        "turns":      np.array(turns),
        "scores":     np.array(scores),
        "dist_flags": np.array(dist_flags),
        "dist_turns": dist_turns,
        "core_issue": core_issue,
        "model":      model_name,
        "mean_clean": mean_clean,
        "mean_dist":  mean_dist,
        "delta":      mean_clean - mean_dist,
    }

# ── Plotting ──────────────────────────────────────────────────────────────────

from matplotlib.colors import Normalize, LinearSegmentedColormap

MODEL_CMAP = {
    "gemma":   LinearSegmentedColormap.from_list(
                   "gemma_purples", cm.Purples(np.linspace(0.35, 1.0, 256))),
    "mistral": LinearSegmentedColormap.from_list(
                   "mistral_blues",  cm.Blues(np.linspace(0.35, 1.0, 256))),
}
MODEL_LABEL = {
    "gemma":   "Gemma 3-12B",
    "mistral": "Mistral 7B",
}

def _model_key(model_name):
    return "gemma" if "gemma" in model_name.lower() else "mistral"

def plot_model(model_key, results_list, out_dir):
    """
    One plot per model — X = transcript index, Y = mean NLI alignment score.
    Gradient line coloured by transcript index, same style as overlay_user_sentiment.py.
    """
    results_list = sorted(results_list, key=lambda r: r["file_id"])

    x    = np.array([int(r["file_id"]) for r in results_list], dtype=float)
    y    = np.array([np.nanmean(r["scores"]) for r in results_list], dtype=float)
    lbls = [r["core_issue"] for r in results_list]

    cmap  = MODEL_CMAP[model_key]
    norm  = Normalize(vmin=x.min(), vmax=x.max())
    label = MODEL_LABEL[model_key]

    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"{label}: NLI Topic Alignment Across Transcripts",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Transcript Index", fontsize=13)
    ax.set_ylabel("Mean NLI Topic Alignment Score", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{int(i)}" for i in x], fontsize=10)
    ax.set_ylim(0.0, max(y) * 1.3 if y.max() > 0 else 1.0)

    # Segment-by-segment gradient line (same pattern as overlay_user_sentiment.py)
    for i in range(len(x) - 1):
        ax.plot(x[i:i+2], y[i:i+2],
                color=cmap(norm(x[i])),
                linewidth=3, solid_capstyle="round", zorder=2)

    # Scatter dots coloured by index
    for i, (xi, yi) in enumerate(zip(x, y)):
        ax.scatter(xi, yi, color=cmap(norm(xi)), s=80, zorder=4,
                   edgecolors="white", linewidths=1.2)

    # Core issue labels below each point
    for xi, yi, lbl in zip(x, y, lbls):
        ax.text(xi, yi - max(y) * 0.06, lbl,
                ha="center", va="top", fontsize=8.5,
                color="#546E7A", wrap=True)

    # Start (○) and end (★) markers — white fill, black edge (same as reference)
    ax.scatter(x[0],  y[0],  color="white", edgecolors="black",
               marker="o", s=120, zorder=6, label="First Transcript")
    ax.scatter(x[-1], y[-1], color="white", edgecolors="black",
               marker="*", s=250, zorder=6, label="Last Transcript")

    ax.legend(fontsize=11, frameon=True, loc="upper right")

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Transcript Index", shrink=0.85)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"nli_alignment_{model_key}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

def plot_transcripts(results_by_file, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Group by model
    by_model = {}
    for file_id, results in results_by_file.items():
        for r in results:
            r["file_id"] = file_id
            key = _model_key(r["model"])
            by_model.setdefault(key, []).append(r)

    for model_key, results_list in sorted(by_model.items()):
        plot_model(model_key, results_list, out_dir)

# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(results_by_file):
    print("\n" + "=" * 65)
    print(f"{'File':<6} {'Model':<18} {'Core Issue':<30} {'Mean':>6} {'Delta':>7}")
    print("=" * 65)
    for file_id, results in sorted(results_by_file.items()):
        for r in results:
            model_short = _model_key(r["model"]).capitalize()
            print(f"{file_id:<6} {model_short:<18} {r['core_issue']:<30} "
                  f"{np.nanmean(r['scores']):>6.3f} {r['delta']:>7.3f}")
    print("=" * 65)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    plot_out_dir = "./output/plots/nli_topic_alignment"

    patterns = {
        "gemma":   "./output/gemma_realcbt/realcbt_cbt_mcot_file_*.json",
        "mistral": "./output/mistral_realcbt/realcbt_cbt_mcot_file_*.json",
    }

    results_by_file = {}

    for model_key, pattern in patterns.items():
        for filepath in sorted(glob.glob(pattern)):
            file_id = filepath.split("file_")[-1].replace(".json", "")
            print(f"\nProcessing {model_key} file_{file_id} ...")
            result = process_file(filepath)
            results_by_file.setdefault(file_id, []).append(result)

    print_summary(results_by_file)

    print("\nGenerating plots...")
    plot_transcripts(results_by_file, plot_out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
