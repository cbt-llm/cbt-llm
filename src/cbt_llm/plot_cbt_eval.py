"""
Generate CBT evaluation plots from evaluating_benchmark outputs.

Plots (per model):
1. CBT Coverage (Bar Chart)
2. CBT Bench (Radar Chart)
3. CBT Depth Over Turns (Line Plot)

"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



ROOT = Path(__file__).resolve().parents[2] 
RESULTS_ROOT = ROOT / "output" / "results"
PLOTS_OUT = ROOT / "output" / "reports" / "figures"
PLOTS_OUT.mkdir(parents=True, exist_ok=True)

MODELS = {"gemma": "gemma-2-9b", "mistral": "mistral-7b-instruct"}

PROTOCOL_COLS = [
    "validation_execution",
    "socratic_execution",
    "reframing_execution",
]

LABEL_MAP = {
    "validation_execution": "Validation",
    "socratic_execution": "Socratic Questioning",
    "reframing_execution": "Reframing Unhelpful Thoughts",
}

def load_summary(model: str) -> pd.DataFrame:
    path = RESULTS_ROOT / model / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary.csv for {model}: {path}")
    return pd.read_csv(path)

def load_turns(model: str, stem: str) -> pd.DataFrame:
    path = RESULTS_ROOT / model / f"{stem}.turns.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing turns file: {path}")
    return pd.read_csv(path)

def is_cbt(stem: str) -> bool:
    return stem.startswith("cbt_")


def plot_cbt_coverage(model: str, summary: pd.DataFrame):
    coverage = {"Baseline": [], "CBT": []}

    for _, row in summary.iterrows():
        active = sum(
            row[f"{p}_avg"] > 0
            for p in [
                "validation_execution",
                "socratic_execution",
                "reframing_execution",
            ]
        )
        score = active / 3.0
        group = "CBT" if is_cbt(row["stem"]) else "Baseline"
        coverage[group].append(score)

    means = {k: np.mean(v) for k, v in coverage.items()}

    plt.figure(figsize=(4, 4))
    plt.bar(means.keys(), means.values())
    plt.ylim(0, 1.0)
    plt.ylabel("CBT Coverage")
    plt.title(f"{MODELS[model].upper()}: CBT Coverage")
    plt.tight_layout()
    plt.savefig(PLOTS_OUT / f"{model}_cbt_coverage.png", dpi=200)
    plt.close()


def plot_radar(model: str, summary: pd.DataFrame):
    categories = ["Validation", "Socratic Questioning", "Reframing", "Continuity"]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    def aggregate(group: str):
        rows = summary[
            summary["stem"].apply(lambda s: is_cbt(s) if group == "CBT" else not is_cbt(s))
        ]
        vals = [
            rows["validation_execution_avg"].mean(),
            rows["socratic_execution_avg"].mean(),
            rows["reframing_execution_avg"].mean(),
            1.0 - rows["refusal_rate"].mean(),   # continuity proxy
        ]
        return vals + vals[:1]

    baseline_vals = aggregate("Baseline")
    cbt_vals = aggregate("CBT")

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))

    ax.plot(angles, baseline_vals, label="Baseline")
    ax.fill(angles, baseline_vals, alpha=0.1)

    ax.plot(angles, cbt_vals, label="CBT")
    ax.fill(angles, cbt_vals, alpha=0.1)

    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_ylim(0, 1)
    ax.set_title(f"{model.upper()}: CBT Bench ")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(PLOTS_OUT / f"{model}_cbt_radar.png", dpi=200)
    plt.close()

def plot_depth_over_turns(model: str, summary: pd.DataFrame):
    depth = {"Baseline": {}, "CBT": {}}

    for _, row in summary.iterrows():
        stem = row["stem"]
        group = "CBT" if is_cbt(stem) else "Baseline"
        df = load_turns(model, stem)

        df["cbt_depth"] = df[PROTOCOL_COLS].mean(axis=1)

        for idx, val in enumerate(df["cbt_depth"]):
            depth[group].setdefault(idx, []).append(val)

    plt.figure(figsize=(6, 4))

    for group in ["Baseline", "CBT"]:
        xs = sorted(depth[group].keys())
        ys = [np.mean(depth[group][x]) for x in xs]
        plt.plot(xs, ys, marker="o", label=group)

    plt.xlabel("Therapist Turn Index")
    plt.ylabel("CBT Depth")
    plt.ylim(0, 1)
    plt.title(f"{MODELS[model].upper()}: CBT Depth Over Turns")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_OUT / f"{model}_cbt_depth_over_turns.png", dpi=200)
    plt.close()


def main():
    for model in MODELS:
        summary = load_summary(model)

        plot_cbt_coverage(model, summary)
        plot_radar(model, summary)
        plot_depth_over_turns(model, summary)

        print(f"[OK] Plots generated for {model}")

if __name__ == "__main__":
    main()
