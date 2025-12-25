"""
plot_cbt_eval.py

Generate therapist-side CBT evaluation plots from evaluation/{model}/summary.csv

Plots per model (Appendix):
1. CBT Protocol Adherence (Radar)
2. CBT Quality (Baseline vs CBT-guided)

These plots align with the paper’s therapist-side evaluation:
- Model-level aggregation first
- Cross-model aggregation handled separately for main figures

Usage:
python src/cbt_llm/plot_cbt_eval.py --models gpt gemma mistral
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EVAL_ROOT = Path("evaluation")
PLOTS_OUT = EVAL_ROOT / "plots_cbt"
PLOTS_OUT.mkdir(parents=True, exist_ok=True)

MODEL_DISPLAY = {
    "gpt": "GPT-4o-mini",
    "gemma": "Gemma-2-9B",
    "mistral": "Mistral-7B-Instruct",
    "qwen": "Qwen-3-4B",
    "deepseek": "DeepSeek-R1-8B",
}

PROTOCOL_COLS = {
    "Validation & Reflection": "judge_avg_validate_and_reflect_quality",
    "Socratic Questioning": "judge_avg_socratic_questioning_quality",
    "Cognitive Reframing": "judge_avg_cognitive_reframing_quality",
}

def model_title(model: str) -> str:
    return MODEL_DISPLAY.get(model, model.upper())

def normalize_filename(path: str) -> str:
    return Path(path).name.lower()

def is_cbt_file(path: str) -> bool:
    """
    Strictly identify CBT-guided conversations.
    Matches experimental naming: cbt_*.json
    """
    return normalize_filename(path).startswith("cbt_")

def load_summary(model: str) -> pd.DataFrame:
    path = EVAL_ROOT / model / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary.csv for model: {model}")
    return pd.read_csv(path)


def plot_protocol_radar(model: str, df: pd.DataFrame):
    baseline = df[~df["file"].apply(is_cbt_file)]
    cbt = df[df["file"].apply(is_cbt_file)]

    labels = list(PROTOCOL_COLS.keys())

    baseline_vals = [baseline[col].mean() for col in PROTOCOL_COLS.values()]
    cbt_vals = [cbt[col].mean() for col in PROTOCOL_COLS.values()]

    baseline_vals += baseline_vals[:1]
    cbt_vals += cbt_vals[:1]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(
        figsize=(7.5, 6.5),
        subplot_kw=dict(polar=True)
    )

    line_b, = ax.plot(angles, baseline_vals, linewidth=2, label="Baseline")
    ax.fill(angles, baseline_vals, alpha=0.15)

    line_c, = ax.plot(angles, cbt_vals, linewidth=2, label="CBT-guided")
    ax.fill(angles, cbt_vals, alpha=0.15)

    ax.set_thetagrids(
        np.degrees(angles[:-1]),
        labels,
        fontsize=11
    )
    ax.tick_params(pad=14)
    ax.set_ylim(0, 5)

    ax.set_title(
        f"{model_title(model)}: CBT Protocol Adherence",
        fontsize=14,
        pad=32,
    )

    fig.legend(
        handles=[line_b, line_c],
        labels=["Baseline", "CBT-guided"],
        loc="center right",
        bbox_to_anchor=(0.98, 0.55),
        frameon=True,
        fontsize=11,
    )

    fig.subplots_adjust(left=0.08, right=0.78, top=0.85, bottom=0.08)

    plt.savefig(PLOTS_OUT / f"{model}_cbt_protocol_adherence.png", dpi=200)
    plt.close()


def plot_overall_quality(model: str, df: pd.DataFrame):
    baseline = df[~df["file"].apply(is_cbt_file)]["judge_avg_avg_score"]
    cbt = df[df["file"].apply(is_cbt_file)]["judge_avg_avg_score"]

    plt.figure(figsize=(5.5, 4.5))
    plt.bar(
        ["Baseline", "CBT-guided"],
        [baseline.mean(), cbt.mean()],
    )

    plt.ylabel("Average CBT Quality Score")
    plt.ylim(0, 5)
    plt.title(f"{model_title(model)}: CBT Quality")

    plt.tight_layout()
    plt.savefig(PLOTS_OUT / f"{model}_overall_cbt_quality.png", dpi=200)
    plt.close()


def plot_delta_cbt_quality_across_models(models):
    """
    Plot Δ-CBT Quality = mean(CBT-guided) - mean(Baseline) per model.
    This is the main paper figure.
    """
    model_names = []
    deltas = []

    for model in models:
        df = load_summary(model)

        baseline = df[~df["file"].apply(is_cbt_file)]["judge_avg_avg_score"]
        cbt = df[df["file"].apply(is_cbt_file)]["judge_avg_avg_score"]

        if baseline.empty or cbt.empty:
            continue

        delta = cbt.mean() - baseline.mean()

        model_names.append(model_title(model))
        deltas.append(delta)

    x = np.arange(len(model_names))

    plt.figure(figsize=(7.0, 4.5))
    plt.bar(x, deltas)

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(x, model_names)
    plt.ylabel("Difference in CBT Quality (CBT-guided - Baseline)")


    plt.title("Effect of CBT-Guided Prompting on Therapist Quality")

    plt.tight_layout()
    plt.savefig(
        PLOTS_OUT / "difference_cbt_quality_across_models.png",
        dpi=200
    )
    plt.close()



def main(models):
    for model in models:
        df = load_summary(model)

        required = {"file", "judge_avg_avg_score"} | set(PROTOCOL_COLS.values())
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{model} summary.csv missing columns: {missing}")

        plot_protocol_radar(model, df)
        plot_overall_quality(model, df)
    
    plot_delta_cbt_quality_across_models(models)

    print("Evaluation plots generated.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        nargs="+",
        required=True,
    )
    args = ap.parse_args()

    main(args.models)
