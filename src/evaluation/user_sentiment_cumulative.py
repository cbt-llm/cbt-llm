from pathlib import Path
import argparse
import json
import math
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
    "cot": "CoT",
    "mcot": "MCoT",
}


# =========================
# HELPERS
# =========================

def model_title(model: str) -> str:
    return MODEL_DISPLAY.get(model, model.upper())


def classify_mode_from_filename(filename: str) -> str | None:
    name = filename.lower()

    if name.startswith("baseline_"):
        return "baseline"
    if name.startswith("cbt_mcot_"):
        return "mcot"
    if name.startswith("cbt_"):
        return "cot"
    return None

def transcript_sort_key(fp: Path):
    """
    Sort by trailing integer if present.
    """
    m = re.search(r"(\d+)(?=\.json$)", fp.name)
    if m:
        return (fp.stem, int(m.group(1)))
    return (fp.stem, 10**9)


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

    if isinstance(obj, dict) and "transcript" in obj and isinstance(obj["transcript"], list):
        return obj["transcript"]

    print(f"[WARN] Unexpected format in {fp.name}")
    if isinstance(obj, dict):
        print("Top-level keys:", list(obj.keys()))
    return []

def extract_patient_turn_sentiments(messages, analyzer, fp_name=None):
    """
    Extract patient/user sentiment from nested transcript format.

    Expected turn structure:
    {
        "turn": 0,
        "patient": {
            "role": "patient",
            "query": "..."
        },
        "llm_response": {
            "role": "agent" or "cbt_agent",
            "response": "..."
        }
    }
    """
    rows = []

    if not isinstance(messages, list):
        print(f"[WARN] messages is not a list for {fp_name}")
        return rows

    for item in messages:
        if not isinstance(item, dict):
            continue

        patient_block = item.get("patient", {})
        if not isinstance(patient_block, dict):
            continue

        role = str(patient_block.get("role", "")).strip().lower()
        if role not in {"patient", "user"}:
            continue

        text = str(patient_block.get("query", "")).strip()
        if not text:
            continue

        turn_idx = item.get("turn", None)
        if turn_idx is None:
            turn_idx = len(rows)

        score = analyzer.polarity_scores(text)["compound"]

        rows.append({
            "turn": int(turn_idx) + 1,   # makes plotting start at 1 instead of 0
            "text": text,
            "sentiment": float(score),
        })

    if fp_name is not None:
        print(f"{fp_name}: extracted {len(rows)} patient turns")

    return rows


def build_raw_df(files, analyzer, mode: str):
    all_rows = []

    for fp in files:
        messages = extract_messages(fp)
        turn_rows = extract_patient_turn_sentiments(messages, analyzer, fp.name)

        for row in turn_rows:
            all_rows.append({
                "mode": mode,
                "transcript": fp.stem,
                "turn": row["turn"],
                "sentiment": row["sentiment"],
            })

    if not all_rows:
        print(f"[WARN] No patient turns extracted for mode={mode}")
        return pd.DataFrame(columns=["mode", "transcript", "turn", "sentiment"])

    df = pd.DataFrame(all_rows)
    print(f"Mode={mode}: extracted {len(df)} total patient-turn rows")
    return df


def add_cumulative_sentiment(raw_df: pd.DataFrame):
    """
    Adds cumulative average sentiment within each transcript:
      cumulative_sentiment at turn t = mean(sentiment of turns 1..t)
    """
    if raw_df.empty:
        out = raw_df.copy()
        out["cumulative_sentiment"] = pd.Series(dtype=float)
        return out

    raw_df = raw_df.sort_values(["transcript", "turn"]).copy()

    raw_df["cumulative_sentiment"] = (
        raw_df.groupby("transcript")["sentiment"]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )

    return raw_df


def summarize_turn_by_turn(df_with_cum: pd.DataFrame):
    """
    Mean of exact turn t across transcripts.
    """
    if df_with_cum.empty:
        return pd.DataFrame(columns=["turn", "mean_sentiment", "sem", "n"])

    out = (
        df_with_cum.groupby("turn", as_index=False)
        .agg(
            mean_sentiment=("sentiment", "mean"),
            sem=("sentiment", "sem"),
            n=("sentiment", "count"),
        )
        .sort_values("turn")
    )

    out["sem"] = out["sem"].fillna(0.0)
    return out


def summarize_cumulative(df_with_cum: pd.DataFrame):
    """
    Mean of cumulative sentiment at turn t across transcripts.
    This is the running trend you want.
    """
    if df_with_cum.empty:
        return pd.DataFrame(columns=["turn", "mean_cumulative_sentiment", "sem", "n"])

    out = (
        df_with_cum.groupby("turn", as_index=False)
        .agg(
            mean_cumulative_sentiment=("cumulative_sentiment", "mean"),
            sem=("cumulative_sentiment", "sem"),
            n=("cumulative_sentiment", "count"),
        )
        .sort_values("turn")
    )

    out["sem"] = out["sem"].fillna(0.0)
    return out


def summarize_by_transcript(df_with_cum: pd.DataFrame):
    """
    One row per transcript:
      - average exact sentiment across all turns
      - final cumulative value (same as average across all turns)
      - num turns
    """
    if df_with_cum.empty:
        return pd.DataFrame(columns=[
            "transcript",
            "avg_sentiment",
            "final_cumulative_sentiment",
            "num_turns",
        ])

    grouped = df_with_cum.groupby("transcript", as_index=False)

    avg_df = grouped.agg(
        avg_sentiment=("sentiment", "mean"),
        num_turns=("turn", "max"),
    )

    final_cum = (
        df_with_cum.sort_values(["transcript", "turn"])
        .groupby("transcript", as_index=False)
        .tail(1)[["transcript", "cumulative_sentiment"]]
        .rename(columns={"cumulative_sentiment": "final_cumulative_sentiment"})
    )

    out = avg_df.merge(final_cum, on="transcript", how="left")
    return out.sort_values("transcript")


def build_overall_summary(model: str, mode: str, transcript_summary: pd.DataFrame, raw_df: pd.DataFrame):
    if transcript_summary.empty or raw_df.empty:
        return {
            "model": model,
            "mode": mode,
            "num_transcripts": 0,
            "avg_sentiment_across_transcripts": np.nan,
            "std_sentiment_across_transcripts": np.nan,
            "avg_sentiment_across_all_turns": np.nan,
            "num_patient_turns": 0,
        }

    return {
        "model": model,
        "mode": mode,
        "num_transcripts": int(len(transcript_summary)),
        "avg_sentiment_across_transcripts": float(transcript_summary["avg_sentiment"].mean()),
        "std_sentiment_across_transcripts": float(
            transcript_summary["avg_sentiment"].std(ddof=1)
        ) if len(transcript_summary) > 1 else 0.0,
        "avg_sentiment_across_all_turns": float(raw_df["sentiment"].mean()),
        "num_patient_turns": int(len(raw_df)),
    }


# =========================
# PLOTTING
# =========================

def plot_turn_by_turn(mode: str, summary_df: pd.DataFrame, out_dir: Path, model: str):
    plt.figure(figsize=(7, 4.5))

    if summary_df.empty:
        plt.text(0.5, 0.5, f"No data for {MODE_LABELS[mode]}", ha="center", va="center")
        plt.title(f"{model_title(model)} — {MODE_LABELS[mode]} User Sentiment by Turn")
        plt.tight_layout()
        plt.savefig(out_dir / f"{mode}_turn_by_turn_sentiment.png", dpi=200)
        plt.close()
        return

    x = summary_df["turn"].to_numpy()
    y = summary_df["mean_sentiment"].to_numpy()
    sem = summary_df["sem"].to_numpy()

    plt.plot(x, y, marker="o", label=MODE_LABELS[mode])
    plt.fill_between(x, y - sem, y + sem, alpha=0.2)

    plt.axhline(0, linestyle="--", alpha=0.4)
    plt.xlabel("Patient Turn")
    plt.ylabel("VADER Compound Sentiment")
    plt.title(f"{model_title(model)} — {MODE_LABELS[mode]} User Sentiment by Turn")
    plt.tight_layout()
    plt.savefig(out_dir / f"{mode}_turn_by_turn_sentiment.png", dpi=200)
    plt.close()


def plot_cumulative(mode: str, summary_df: pd.DataFrame, out_dir: Path, model: str):
    plt.figure(figsize=(7, 4.5))

    if summary_df.empty:
        plt.text(0.5, 0.5, f"No data for {MODE_LABELS[mode]}", ha="center", va="center")
        plt.title(f"{model_title(model)} — {MODE_LABELS[mode]} Cumulative User Sentiment")
        plt.tight_layout()
        plt.savefig(out_dir / f"{mode}_cumulative_sentiment.png", dpi=200)
        plt.close()
        return

    x = summary_df["turn"].to_numpy()
    y = summary_df["mean_cumulative_sentiment"].to_numpy()
    sem = summary_df["sem"].to_numpy()

    plt.plot(x, y, marker="o", label=MODE_LABELS[mode])
    plt.fill_between(x, y - sem, y + sem, alpha=0.2)

    plt.axhline(0, linestyle="--", alpha=0.4)
    plt.xlabel("Patient Turn")
    plt.ylabel("Running Average VADER Sentiment")
    plt.title(f"{model_title(model)} — {MODE_LABELS[mode]} Cumulative User Sentiment")
    plt.tight_layout()
    plt.savefig(out_dir / f"{mode}_cumulative_sentiment.png", dpi=200)
    plt.close()


def plot_combined_cumulative(all_mode_summaries: dict, out_dir: Path, model: str):
    plt.figure(figsize=(7.5, 5))

    has_any = False
    for mode in ["baseline", "cot", "mcot"]:
        df = all_mode_summaries.get(mode)
        if df is None or df.empty:
            continue

        has_any = True
        x = df["turn"].to_numpy()
        y = df["mean_cumulative_sentiment"].to_numpy()
        sem = df["sem"].to_numpy()

        plt.plot(x, y, marker="o", label=MODE_LABELS[mode])
        plt.fill_between(x, y - sem, y + sem, alpha=0.15)

    if not has_any:
        plt.text(0.5, 0.5, "No data found", ha="center", va="center")

    plt.axhline(0, linestyle="--", alpha=0.4)
    plt.xlabel("Patient Turn")
    plt.ylabel("Running Average VADER Sentiment")
    plt.title(f"{model_title(model)} — Cumulative User Sentiment Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "combined_cumulative_sentiment.png", dpi=200)
    plt.close()


def plot_combined_turn_by_turn(all_mode_summaries: dict, out_dir: Path, model: str):
    plt.figure(figsize=(7.5, 5))

    has_any = False
    for mode in ["baseline", "cot", "mcot"]:
        df = all_mode_summaries.get(mode)
        if df is None or df.empty:
            continue

        has_any = True
        x = df["turn"].to_numpy()
        y = df["mean_sentiment"].to_numpy()
        sem = df["sem"].to_numpy()

        plt.plot(x, y, marker="o", label=MODE_LABELS[mode])
        plt.fill_between(x, y - sem, y + sem, alpha=0.15)

    if not has_any:
        plt.text(0.5, 0.5, "No data found", ha="center", va="center")

    plt.axhline(0, linestyle="--", alpha=0.4)
    plt.xlabel("Patient Turn")
    plt.ylabel("VADER Compound Sentiment")
    plt.title(f"{model_title(model)} — Turn-by-Turn User Sentiment Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "combined_turn_by_turn_sentiment.png", dpi=200)
    plt.close()


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="Model folder inside output/, e.g. deepseek, gpt, gemma, mistral"
    )
    args = parser.parse_args()

    model = args.model
    out_dir = SAVE_ROOT / model
    out_dir.mkdir(parents=True, exist_ok=True)

    print("PROJECT_ROOT =", PROJECT_ROOT)
    print("OUTPUT_ROOT  =", OUTPUT_ROOT)
    print("MODEL DIR    =", OUTPUT_ROOT / model)

    analyzer = SentimentIntensityAnalyzer()
    grouped_files = load_transcript_files(model)

    overall_rows = []
    cumulative_summaries = {}
    turn_summaries = {}

    for mode in ["baseline", "cot", "mcot"]:
        files = grouped_files[mode]

        raw_df = build_raw_df(files, analyzer, mode)
        raw_with_cum_df = add_cumulative_sentiment(raw_df)

        turn_summary = summarize_turn_by_turn(raw_with_cum_df)
        cumulative_summary = summarize_cumulative(raw_with_cum_df)
        transcript_summary = summarize_by_transcript(raw_with_cum_df)

        raw_df.to_csv(out_dir / f"{mode}_raw_turn_sentiment.csv", index=False)
        raw_with_cum_df.to_csv(out_dir / f"{mode}_raw_turn_sentiment_with_cumulative.csv", index=False)
        turn_summary.to_csv(out_dir / f"{mode}_turn_by_turn_summary.csv", index=False)
        cumulative_summary.to_csv(out_dir / f"{mode}_cumulative_summary.csv", index=False)
        transcript_summary.to_csv(out_dir / f"{mode}_transcript_summary.csv", index=False)

        plot_turn_by_turn(mode, turn_summary, out_dir, model)
        plot_cumulative(mode, cumulative_summary, out_dir, model)

        turn_summaries[mode] = turn_summary
        cumulative_summaries[mode] = cumulative_summary

        overall_rows.append(
            build_overall_summary(
                model=model,
                mode=mode,
                transcript_summary=transcript_summary,
                raw_df=raw_df,
            )
        )

    plot_combined_cumulative(cumulative_summaries, out_dir, model)
    plot_combined_turn_by_turn(turn_summaries, out_dir, model)

    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(out_dir / "overall_summary_by_mode.csv", index=False)

    print(f"\nSaved outputs to: {out_dir}")

    print("Generated per mode:")
    print("  - *_raw_turn_sentiment.csv")
    print("  - *_raw_turn_sentiment_with_cumulative.csv")
    print("  - *_turn_by_turn_summary.csv")
    print("  - *_cumulative_summary.csv")
    print("  - *_transcript_summary.csv")
    print("  - *_turn_by_turn_sentiment.png")
    print("  - *_cumulative_sentiment.png")
    print("Generated combined:")
    print("  - combined_turn_by_turn_sentiment.png")
    print("  - combined_cumulative_sentiment.png")
    print("  - overall_summary_by_mode.csv")


if __name__ == "__main__":
    main()