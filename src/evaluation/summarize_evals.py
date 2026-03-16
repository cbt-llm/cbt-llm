from pathlib import Path
import json
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------
# SETTINGS
# -----------------------------

JUDGES = {
    "gpt-5.1": "GPT-5.1",
    "qwen3-14b": "Qwen3-14B"
}

MODE_LABELS = {
    "baseline": "Baseline",
    "cbt": "CBT CoT",
    "cbt_mcot": "CBT MCoT"
}

MODE_COLORS = {
    "baseline": "#1f77b4",
    "cbt": "#2ca02c",
    "cbt_mcot": "#ff7f0e"
}

JUDGE_LINESTYLE = {
    "gpt-5.1": "-",
    "qwen3-14b": "--"
}

MODEL_TITLES = {
    "gpt": "GPT-OSS-20B",
    "gemma": "Gemma3-12B",
    "deepseek": "DeepSeek-R1-8B",
    "mistral": "Mistral-7B"
}

PROTOCOL_DIMS = [
    "validate_and_reflect",
    "socratic_questioning",
    "cognitive_restructuring"
]


# -----------------------------
# HELPERS
# -----------------------------

def read_json(path: Path):
    with open(path) as f:
        return json.load(f)


def normalize(score):
    return score / 5


def detect_mode(filename):

    name = filename.lower()

    if name.startswith("baseline"):
        return "baseline"
    if name.startswith("cbt_mcot"):
        return "cbt_mcot"
    if name.startswith("cbt"):
        return "cbt"

    return None


# -----------------------------
# LOAD EVALUATIONS
# -----------------------------

def load_model_evals(base_eval_dir: Path, model: str):

    results = {}

    for judge in JUDGES:

        judge_dir = base_eval_dir / judge / model

        if not judge_dir.exists():
            continue

        if judge not in results:
            results[judge] = {}

        for f in sorted(judge_dir.glob("*_eval.json")):

            mode = detect_mode(f.name)

            if mode is None:
                continue

            if mode not in results[judge]:
                results[judge][mode] = {p: [] for p in PROTOCOL_DIMS}

            data = read_json(f)

            for turn_eval in data.get("turn_evals", []):

                judgment = turn_eval["judgment"]

                for dim in PROTOCOL_DIMS:

                    score = judgment["protocol_scores"][dim]["score"]

                    results[judge][mode][dim].append(
                        normalize(score)
                    )

    return results


# -----------------------------
# RADAR PLOT
# -----------------------------

def plot_radar(results, model, output_dir):

    categories = PROTOCOL_DIMS
    labels = [c.replace("_", " ").title() for c in categories]

    N = len(categories)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)

    for judge, modes in results.items():

        for mode, dim_scores in modes.items():

            values = [
                np.mean(dim_scores[p]) for p in categories
            ]

            values += values[:1]

            ax.plot(
                angles,
                values,
                linewidth=2,
                linestyle=JUDGE_LINESTYLE[judge],
                color=MODE_COLORS[mode],
                label=f"{MODE_LABELS[mode]} ({JUDGES[judge]})"
            )

            ax.fill(
                angles,
                values,
                alpha=0.15,
                color=MODE_COLORS[mode]
            )

    ax.set_thetagrids(
        np.degrees(angles[:-1]),
        labels=labels
    )

    ax.tick_params(axis="x", pad=14)

    ax.set_ylim(0,1)
    ax.set_rlabel_position(30)

    title = MODEL_TITLES.get(model, model)

    ax.set_title(
        f"CBT Protocol Scores: {title}",
        fontsize=14,
        y=1.08
    )

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.1,1.1),
        frameon=True
    )

    plt.subplots_adjust(right=0.75)

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        output_dir / f"protocol_radar_plot_{model}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# -----------------------------
# TABLE SUMMARY
# -----------------------------

def build_summary_table(results):

    rows = []

    for judge, modes in results.items():

        for mode, dim_scores in modes.items():

            avg = np.mean([
                np.mean(dim_scores[p]) for p in PROTOCOL_DIMS
            ])

            rows.append({
                "judge": JUDGES[judge],
                "mode": MODE_LABELS[mode],
                "protocol_mean": avg
            })

    df = pd.DataFrame(rows)

    return df


# -----------------------------
# MAIN
# -----------------------------

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--eval-root",
        default="evals",
        help="root directory containing judge folders"
    )

    parser.add_argument(
        "--model",
        default="all",
        help="model name or 'all'"
    )

    parser.add_argument(
        "--output-dir",
        default="evals_summary"
    )

    args = parser.parse_args()

    eval_root = Path(args.eval_root)

    models = list(MODEL_TITLES.keys())

    if args.model != "all":
        models = [args.model]

    for model in models:

        print(f"\nProcessing model: {model}")

        results = load_model_evals(eval_root, model)

        if not results:
            print("No data found.")
            continue

        out_dir = Path(args.output_dir)

        plot_radar(results, model, out_dir)

        df = build_summary_table(results)

        df.to_csv(
            out_dir / f"protocol_summary_{model}.csv",
            index=False
        )

        print(df)