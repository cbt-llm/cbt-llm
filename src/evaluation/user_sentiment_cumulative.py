from pathlib import Path
import argparse
import json
import re

import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "output"
SAVE_ROOT = PROJECT_ROOT / "evaluation" / "vader_sentiment_plots"

MODEL_DISPLAY = {
    "gpt": "GPT-OSS-20B",
    "gemma": "Gemma3-12B",
    "mistral": "Mistral-7B-Instruct",
    "deepseek": "DeepSeek-R1-8B",
}
GRID_ORDER = ["mistral", "deepseek", "gemma", "gpt"]  # panel layout for the 2x2

# therapist reasoning mode -> (label, color)
MODES = {
    "baseline": ("Baseline", "#4C72B0"),
    "cot": ("CBT-CoT", "#55A868"),
    "mcot": ("CBT-MCoT", "#C44E52"),
}

# swap this single function to change the sentiment scorer later
_AZ = SentimentIntensityAnalyzer()
def score_text(text: str) -> float:
    return _AZ.polarity_scores(text)["compound"]


def mode_of(name):
    n = name.lower()
    if n.startswith("baseline_"):
        return "baseline"
    if n.startswith("cbt_mcot_") or n.startswith("mcot_"):
        return "mcot"
    if n.startswith("cbt_") or n.startswith("cot_"):
        return "cot"
    return None


def case_id(name):
    m = re.search(r"(\d+)(?=\.json$)", name)
    return int(m.group(1)) if m else None


def patient_rows(fp):
    """(turn, query) per patient utterance. turn = file's own field (0-indexed)."""
    obj = json.loads(fp.read_text(encoding="utf-8"))
    out = []
    for it in obj.get("transcript", []):
        if not isinstance(it, dict):
            continue
        q = str((it.get("patient") or {}).get("query", "")).strip()
        if q and it.get("turn") is not None:
            out.append((int(it["turn"]), q))
    return out


def build(model, max_turns):
    rows = []
    for fp in (OUTPUT_ROOT / model).glob("*.json"):
        mode, cid = mode_of(fp.name), case_id(fp.name)
        if mode is None or cid is None:
            continue
        for turn, q in patient_rows(fp):
            disp = turn + 1               # transcript 0-9 -> User Turn 1-10
            if max_turns and disp > max_turns:
                continue
            rows.append((mode, cid, disp, score_text(q)))
    df = pd.DataFrame(rows, columns=["mode", "case", "turn", "sent"])
    if df.empty:
        raise ValueError(f"No patient turns for '{model}'")

    shared = set.intersection(*(set(g["case"]) for _, g in df.groupby("mode")))
    df = df[df["case"].isin(shared)].sort_values(["mode", "case", "turn"])

    df["cum"] = (df.groupby(["mode", "case"])["sent"]
                 .expanding().mean().reset_index(level=[0, 1], drop=True))

    per_turn = (df.groupby(["mode", "turn"], as_index=False)
                .agg(mean=("sent", "mean"), sem=("sent", "sem"), n=("sent", "count")))
    cumulative = (df.groupby(["mode", "turn"], as_index=False)
                  .agg(mean=("cum", "mean"), sem=("cum", "sem"), n=("cum", "count")))
    per_turn["sem"] = per_turn["sem"].fillna(0.0)
    cumulative["sem"] = cumulative["sem"].fillna(0.0)
    return per_turn, cumulative, len(shared)


def _draw(ax, summary, show_legend):
    for mode, (label, color) in MODES.items():
        d = summary[summary["mode"] == mode]
        if d.empty:
            continue
        ax.plot(d["turn"], d["mean"], marker="o", ms=4, color=color, label=label)
        ax.fill_between(d["turn"], d["mean"] - d["sem"], d["mean"] + d["sem"],
                        color=color, alpha=0.15)
    ax.axhline(0, ls="--", alpha=0.4, lw=0.8)
    if show_legend:
        ax.legend(fontsize=8, loc="lower right")


def _ylim(summaries):
    """Shared y-range across panels, padded, so heights are comparable."""
    lo = min((s["mean"] - s["sem"]).min() for s in summaries)
    hi = max((s["mean"] + s["sem"]).max() for s in summaries)
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def plot_single(summary, ylabel, title, fp):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    _draw(ax, summary, show_legend=True)
    ax.set_xlabel("User Turn")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(fp, dpi=200)
    plt.close(fig)


def plot_grid(model_summaries, key, ylabel, fp):
    """model_summaries: {model: (per_turn_df, cumulative_df)}; key=0 per-turn, 1 cumulative."""
    models = [m for m in GRID_ORDER if m in model_summaries]
    panels = [model_summaries[m][key] for m in models]
    ylo, yhi = _ylim(panels)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, m in zip(axes.flat, models):
        _draw(ax, model_summaries[m][key], show_legend=(ax is axes.flat[0]))
        ax.set_title(MODEL_DISPLAY.get(m, m), fontsize=11)
        ax.set_ylim(ylo, yhi)
    for ax in axes.flat[len(models):]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("User Turn")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(fp, dpi=200)
    plt.close(fig)


def run(model, max_turns):
    out = SAVE_ROOT / model
    out.mkdir(parents=True, exist_ok=True)
    per_turn, cumulative, n = build(model, max_turns)
    name = MODEL_DISPLAY.get(model, model)
    per_turn.to_csv(out / f"{model}_per_turn.csv", index=False)
    cumulative.to_csv(out / f"{model}_cumulative.csv", index=False)
    plot_single(per_turn, "VADER Compound Sentiment",
                f"{name} \u2014 Turn-by-Turn User Sentiment",
                out / f"{model}_per_turn.png")
    plot_single(cumulative, "Running Average Sentiment",
                f"{name} \u2014 Cumulative User Sentiment",
                out / f"{model}_cumulative.png")
    print(f"{model}: {n} shared cases")
    return per_turn, cumulative


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="one model, 'all' (per-model files), or 'grid' (2x2 shared-axis figure)")
    p.add_argument("--max-turns", type=int, default=10, help="0 = no cap")
    args = p.parse_args()
    max_turns = args.max_turns or None

    if args.model == "grid":
        SAVE_ROOT.mkdir(parents=True, exist_ok=True)
        summaries = {}
        for m in GRID_ORDER:
            if (OUTPUT_ROOT / m).is_dir():
                pt, cum, n = build(m, max_turns)
                summaries[m] = (pt, cum)
                print(f"{m}: {n} shared cases")
        plot_grid(summaries, 0, "VADER Compound Sentiment",
                  SAVE_ROOT / "grid_per_turn.png")
        plot_grid(summaries, 1, "Running Average Sentiment",
                  SAVE_ROOT / "grid_cumulative.png")
        print(f"grid saved to {SAVE_ROOT/'grid_per_turn.png'}, {SAVE_ROOT/'grid_cumulative.png'}")
        return

    models = (sorted(d.name for d in OUTPUT_ROOT.iterdir() if d.is_dir())
              if args.model == "all" else [args.model])
    for m in models:
        run(m, max_turns)


if __name__ == "__main__":
    main()