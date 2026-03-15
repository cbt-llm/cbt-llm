from pathlib import Path
import json
from typing import Dict, List
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)

def normalize(score: float) -> float:
    """Convert 0–5 scale to 0–1 scale."""
    return score / 5

def summarize_evals(eval_dir: Path, output_dir: Path):
    protocol_dimensions: List[str] = [
        "validate_and_reflect",
        "socratic_questioning",
        "cognitive_restructuring"
    ]

    modes: Dict[str, Dict] = {}

    # Read all evaluation files
    for f in sorted(eval_dir.glob("*_eval.json")):
        if not f.is_file():
            continue

        name = f.name
        if name.startswith("baseline"):
            mode = "baseline"
        elif name.startswith("cbt_mcot"):
            mode = "cbt_mcot"
        elif name.startswith("cbt"):
            mode = "cbt"
        else:
            continue

        if mode not in modes:
            modes[mode] = {
                "protocol_scores": {p: [] for p in protocol_dimensions},
                "protocol_effectiveness": [],
                "cbt_best_practices": []
            }

        data = read_json(f)
        for turn_eval in data.get("turn_evals", []):
            judgment = turn_eval["judgment"]

            for dim in protocol_dimensions:
                modes[mode]["protocol_scores"][dim].append(
                    normalize(judgment["protocol_scores"][dim]["score"])
                )

            modes[mode]["protocol_effectiveness"].append(
                normalize(judgment["protocol_effectiveness"]["effectiveness"])
            )

            bp = judgment["cbt_best_practices"]
            vals = [
                bp["therapeutic_relationship"],
                bp["collaboration"],
                bp["goal_oriented"],
                bp["present_focused"],
                bp["educative"],
                bp["guided_discovery"],
            ]
            modes[mode]["cbt_best_practices"].append(
                normalize(sum(vals)/len(vals))
            )

    if not modes:
        print(f"[WARNING] No evaluation files found in {eval_dir}")
        return

    # RADAR PLOT: Protocol dimensions only
    categories = protocol_dimensions
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(7,7))
    ax = plt.subplot(111, polar=True)

    for mode, data in modes.items():
        values = [sum(data["protocol_scores"][p])/len(data["protocol_scores"][p])
                for p in categories]
        values += values[:1]
        ax.plot(angles, values, label=mode, linewidth=2)
        ax.fill(angles, values, alpha=0.25)

    # Fix label positions
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    for label, angle in zip(ax.get_xticklabels(), angles):
        label.set_horizontalalignment('center')
        label.set_rotation(np.degrees(angle))
        label.set_rotation_mode('anchor')

    # Set radial limits and grid
    ax.set_rlabel_position(30)
    ax.set_ylim(0, 1)
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    ax.set_title(f"CBT Protocol Radar Plot: {eval_dir.name}", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / "protocol_radar_plot.png", dpi=300)
    plt.close()

    # TABLE: Protocol effectiveness and CBT best practices
    table_rows = []
    for mode, data in modes.items():
        eff = sum(data["protocol_effectiveness"])/len(data["protocol_effectiveness"])
        bp_avg = sum(data["cbt_best_practices"])/len(data["cbt_best_practices"])
        table_rows.append({
            "mode": mode,
            "protocol_effectiveness": eff,
            "cbt_best_practices": bp_avg
        })

    df = pd.DataFrame(table_rows)
    df = df.set_index("mode")
    df.to_csv(output_dir / "effectiveness_best_practices.csv")
    print("Saved effectiveness & best practices table to:", output_dir / "effectiveness_best_practices.csv")
    print(df)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--laaj-model", required=True, help="Top-level model folder (gpt-5.1, qwen3:14b, etc.)")
    parser.add_argument("--model", required=True, help="Sub-model folder (gemma, deepseek, etc.) or 'all'")
    parser.add_argument("--output-dir", default="evals_summary", help="Directory to save CSV and plots")
    args = parser.parse_args()

    laaj_dir = Path("evals") / args.laaj_model
    output_dir_base = Path(args.output_dir) / args.laaj_model

    if args.model.lower() == "all":
        submodels = [d.name for d in laaj_dir.iterdir() if d.is_dir()]
    else:
        submodels = [args.model]

    for model_name in submodels:
        eval_dir = laaj_dir / model_name
        output_dir = output_dir_base / model_name
        print(f"\nProcessing {args.laaj_model}/{model_name} ...")
        summarize_evals(eval_dir, output_dir)