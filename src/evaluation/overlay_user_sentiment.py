from pathlib import Path
import argparse
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# =========================
# CONFIG
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "output"
SAVE_ROOT = PROJECT_ROOT / "evaluation" / "user_sentiment_plots"
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

MODEL_DISPLAY = {
    "gpt": "GPT-4o-mini",
    "gemma": "Gemma-2-9B",
    "mistral": "Mistral-7B-Instruct",
    "qwen": "Qwen-3-4B",
    "deepseek": "DeepSeek-R1-8B",
}

MODE_LABELS = {
    "baseline": "Baseline",
    "cot": "CBT-CoT",
    "mcot": "CBT-MCoT",
}

COLOR_MAP = {
    "baseline": "#4C72B0",  # blue
    "cot": "#55A868",       # green
    "mcot": "#C44E52",      # red
}


# =========================
# HELPERS
# =========================

def model_title(model: str):
    return MODEL_DISPLAY.get(model, model.upper())


def classify_mode_from_filename(filename: str):
    name = filename.lower()

    if name.startswith("baseline_"):
        return "baseline"
    if name.startswith("cbt_mcot_"):
        return "mcot"
    if name.startswith("cbt_"):
        return "cot"

    return None


def transcript_sort_key(fp: Path):
    m = re.search(r"(\d+)(?=\.json$)", fp.name)
    if m:
        return int(m.group(1))
    return 9999


def load_transcript_files(model: str):

    model_dir = OUTPUT_ROOT / model

    if not model_dir.exists():
        raise FileNotFoundError(f"Missing output directory: {model_dir}")

    grouped = {
        "baseline": [],
        "cot": [],
        "mcot": [],
    }

    for fp in sorted(model_dir.glob("*.json"), key=transcript_sort_key):

        mode = classify_mode_from_filename(fp.name)

        if mode is not None:
            grouped[mode].append(fp)

    return grouped


def extract_messages(fp: Path):

    with fp.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, dict) and "transcript" in obj:
        return obj["transcript"]

    return []


def extract_user_turn_sentiments(messages, analyzer):

    rows = []

    for item in messages:

        patient_block = item.get("patient", {})

        if not isinstance(patient_block, dict):
            continue

        role = str(patient_block.get("role", "")).lower()

        if role not in {"patient", "user"}:
            continue

        text = patient_block.get("query", "").strip()

        if not text:
            continue

        turn_idx = item.get("turn", len(rows))

        score = analyzer.polarity_scores(text)["compound"]

        rows.append({
            "turn": int(turn_idx) + 1,
            "sentiment": float(score),
        })

    return rows


def build_raw_df(files, analyzer, mode):

    rows = []

    for fp in files:

        messages = extract_messages(fp)

        turn_rows = extract_user_turn_sentiments(messages, analyzer)

        for row in turn_rows:

            rows.append({
                "mode": mode,
                "transcript": fp.stem,
                "turn": row["turn"],
                "sentiment": row["sentiment"],
            })

    return pd.DataFrame(rows)


def add_cumulative_sentiment(df):

    if df.empty:
        df["cumulative_sentiment"] = []
        return df

    df = df.sort_values(["transcript", "turn"])

    df["cumulative_sentiment"] = (
        df.groupby("transcript")["sentiment"]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )

    return df


def summarize_turn(df):

    if df.empty:
        return pd.DataFrame()

    out = (
        df.groupby("turn", as_index=False)
        .agg(
            mean_sentiment=("sentiment", "mean"),
            sem=("sentiment", "sem"),
        )
        .sort_values("turn")
    )

    out["sem"] = out["sem"].fillna(0)

    return out


def summarize_cumulative(df):

    if df.empty:
        return pd.DataFrame()

    out = (
        df.groupby("turn", as_index=False)
        .agg(
            mean_cumulative_sentiment=("cumulative_sentiment", "mean"),
            sem=("cumulative_sentiment", "sem"),
        )
        .sort_values("turn")
    )

    out["sem"] = out["sem"].fillna(0)

    return out


# =========================
# PLOTTING
# =========================

def plot_combined_cumulative(all_summaries, out_dir, model):

    plt.figure(figsize=(7.5, 5))

    for mode in ["baseline", "cot", "mcot"]:

        df = all_summaries.get(mode)

        if df is None or df.empty:
            continue

        x = df["turn"].to_numpy()
        y = df["mean_cumulative_sentiment"].to_numpy()
        sem = df["sem"].to_numpy()

        color = COLOR_MAP[mode]

        plt.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            color=color,
            label=MODE_LABELS[mode],
        )

        plt.fill_between(
            x,
            y - sem,
            y + sem,
            color=color,
            alpha=0.15,
        )

    plt.axhline(0, linestyle="--", alpha=0.4)

    plt.xlabel("User Turn", fontsize=16)
    plt.ylabel("Running Average Sentiment", fontsize=16)

    plt.legend(
        loc="lower right",   # force consistent placement
        frameon=True,
        fontsize=16
    )

    plt.title("", fontsize=12)

    plt.tight_layout()

    plt.savefig(
        out_dir / "combined_cumulative_sentiment.png",
        dpi=300,
    )

    plt.close()


# =========================
# MAIN
# =========================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        help="model folder in output/",
    )

    args = parser.parse_args()

    model = args.model

    out_dir = SAVE_ROOT / model
    out_dir.mkdir(parents=True, exist_ok=True)

    analyzer = SentimentIntensityAnalyzer()

    grouped_files = load_transcript_files(model)

    cumulative_summaries = {}

    for mode in ["baseline", "cot", "mcot"]:

        files = grouped_files[mode]

        raw_df = build_raw_df(files, analyzer, mode)

        raw_with_cum = add_cumulative_sentiment(raw_df)

        cumulative_summary = summarize_cumulative(raw_with_cum)

        cumulative_summary.to_csv(
            out_dir / f"{mode}_cumulative_summary.csv",
            index=False,
        )

        cumulative_summaries[mode] = cumulative_summary

    plot_combined_cumulative(
        cumulative_summaries,
        out_dir,
        model,
    )

    print(f"\nSaved plots to: {out_dir}")


if __name__ == "__main__":
    main()