from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLOTS_ROOT = PROJECT_ROOT / "evaluation" / "user_sentiment_plots"

MODEL_DISPLAY = {
    "gpt": "GPT-4o-mini",
    "gemma": "Gemma-2-9B",
    "mistral": "Mistral-7B-Instruct",
    "qwen": "Qwen-3-4B",
    "deepseek": "DeepSeek-R1-8B",
}

MODE_LABELS = {
    "baseline": "Baseline",
    "cot": "CoT",
    "mcot": "MCoT",
}


def model_title(model: str) -> str:
    return MODEL_DISPLAY.get(model, model.upper())


def load_image_safe(path: Path):
    if not path.exists():
        return None
    return mpimg.imread(path)


def make_side_by_side_figure(model_dir: Path, model: str, kind: str):
    """
    kind:
      - "cumulative"
      - "turn_by_turn"
    Expected filenames:
      baseline_cumulative_sentiment.png
      cot_cumulative_sentiment.png
      mcot_cumulative_sentiment.png

      baseline_turn_by_turn_sentiment.png
      cot_turn_by_turn_sentiment.png
      mcot_turn_by_turn_sentiment.png
    """
    if kind == "cumulative":
        file_map = {
            "baseline": model_dir / "baseline_cumulative_sentiment.png",
            "cot": model_dir / "cot_cumulative_sentiment.png",
            "mcot": model_dir / "mcot_cumulative_sentiment.png",
        }
        out_path = model_dir / "side_by_side_cumulative_sentiment.png"
        suptitle = f"{model_title(model)} — Cumulative User Sentiment"
    elif kind == "turn_by_turn":
        file_map = {
            "baseline": model_dir / "baseline_turn_by_turn_sentiment.png",
            "cot": model_dir / "cot_turn_by_turn_sentiment.png",
            "mcot": model_dir / "mcot_turn_by_turn_sentiment.png",
        }
        out_path = model_dir / "side_by_side_turn_by_turn_sentiment.png"
        suptitle = f"{model_title(model)} — User Sentiment by Turn"
    else:
        raise ValueError("kind must be 'cumulative' or 'turn_by_turn'")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    modes = ["baseline", "cot", "mcot"]

    for ax, mode in zip(axes, modes):
        img = load_image_safe(file_map[mode])

        if img is None:
            ax.text(0.5, 0.5, f"Missing plot:\n{file_map[mode].name}", ha="center", va="center", fontsize=11)
            ax.set_title(MODE_LABELS[mode], fontsize=13)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        ax.imshow(img)
        ax.set_title(MODE_LABELS[mode], fontsize=13)
        ax.axis("off")

    fig.suptitle(suptitle, fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="Model folder under evaluation/user_sentiment_plots/, e.g. deepseek, gpt, gemma, mistral",
    )
    parser.add_argument(
        "--kind",
        choices=["cumulative", "turn_by_turn", "both"],
        default="both",
        help="Which combined figure to create",
    )
    args = parser.parse_args()

    model = args.model
    model_dir = PLOTS_ROOT / model

    if not model_dir.exists():
        raise FileNotFoundError(f"Missing model plot directory: {model_dir}")

    if args.kind in {"cumulative", "both"}:
        make_side_by_side_figure(model_dir, model, "cumulative")

    if args.kind in {"turn_by_turn", "both"}:
        make_side_by_side_figure(model_dir, model, "turn_by_turn")


if __name__ == "__main__":
    main()