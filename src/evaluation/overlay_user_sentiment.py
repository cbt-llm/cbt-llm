from pathlib import Path
import argparse
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "output"
SAVE_ROOT = PROJECT_ROOT / "evaluation" / "user_sentiment_plots"
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

# Download NRC VAD v2.1 lexicon directory (https://saifmohammad.com/WebPages/nrc-vad.html) and please place the unzipped folder at: project/external_libs/
NRC_VAD_DIR = PROJECT_ROOT / "external_libs" / "NRC-VAD-Lexicon-v2.1"

MODE_LABELS = {
    "baseline": "Baseline",
    "cot": "CBT-CoT",
    "mcot": "CBT-MCoT",
}

CIRCUMPLEX_CMAPS = {
    "baseline": cm.Purples,
    "cot": cm.Greens,
    "mcot": cm.Blues,
}

CIRCUMPLEX_LEGEND_COLORS = {
    "baseline": cm.Purples(0.75),
    "cot": cm.Greens(0.75),
    "mcot": cm.Blues(0.75),
}


def load_nrc_vad(lexicon_dir: Path) -> dict:
    unigrams_dir = lexicon_dir / "Unigrams"

    def _load_dim(filename: str) -> dict:
        fp = (unigrams_dir / filename).resolve()
        df = pd.read_csv(fp, sep="\t", header=0)
        df.columns = [c.strip().lower() for c in df.columns]
        word_col = df.columns[0]
        score_col = df.columns[1]
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
        df = df.dropna(subset=[score_col])
        print(f"  {filename}: {len(df)} entries, "
              f"score range [{df[score_col].min():.3f}, {df[score_col].max():.3f}]")
        return df.set_index(word_col)[score_col].to_dict()

    valence = _load_dim("unigrams-valence-NRC-VAD-Lexicon-v2.1.txt")
    arousal = _load_dim("unigrams-arousal-NRC-VAD-Lexicon-v2.1.txt")

    common = set(valence) & set(arousal)
    vad_dict = {w: {"valence": valence[w], "arousal": arousal[w]} for w in common}
    print(f"NRC VAD lexicon loaded: {len(vad_dict):,} unigrams\n")
    return vad_dict

def coverage_report(messages, vad_dict):
    total_tokens, matched_tokens = 0, 0
    for item in messages:
        text = item.get("patient", {}).get("query", "")
        tokens = re.findall(r"\b[a-z]+\b", text.lower())
        total_tokens += len(tokens)
        matched_tokens += sum(1 for t in tokens if t in vad_dict)
    if total_tokens:
        print(f"Coverage: {matched_tokens}/{total_tokens} = {matched_tokens/total_tokens:.1%}")


def classify_mode_from_filename(filename: str):
    name = filename.lower()
    if name.startswith("baseline_"):
        return "baseline"
    if name.startswith("cbt_mcot_"):
        return "mcot"
    if name.startswith("cbt_"):
        return "cot"
    return None


def transcript_sort_key(fp: Path) -> int:
    m = re.search(r"(\d+)(?=\.json$)", fp.name)
    return int(m.group(1)) if m else 9999


def load_transcript_files(model: str) -> dict:
    model_dir = OUTPUT_ROOT / model
    if not model_dir.exists():
        raise FileNotFoundError(f"Missing output directory: {model_dir}")
    grouped = {"baseline": [], "cot": [], "mcot": []}
    for fp in sorted(model_dir.glob("*.json"), key=transcript_sort_key):
        mode = classify_mode_from_filename(fp.name)
        if mode is not None:
            grouped[mode].append(fp)
    return grouped


def extract_messages(fp: Path) -> list:
    with fp.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "transcript" in obj:
        return obj["transcript"]
    return []


def extract_user_turn_sentiments(messages: list, vad_dict: dict) -> list:
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
        tokens = re.findall(r"\b[a-z]+\b", text.lower())
        matched = [vad_dict[t] for t in tokens if t in vad_dict]
        if not matched:
            continue
        rows.append({
            "turn": int(turn_idx) + 1,
            "valence": float(np.mean([m["valence"] for m in matched])),
            "arousal": float(np.mean([m["arousal"] for m in matched])),
        })
    return rows


def build_raw_df(files: list, vad_dict: dict, mode: str) -> pd.DataFrame:
    rows = []
    for fp in files:
        messages = extract_messages(fp)
        turn_rows = extract_user_turn_sentiments(messages, vad_dict)
        for row in turn_rows:
            rows.append({
                "mode": mode,
                "transcript": fp.stem,
                "turn": row["turn"],
                "valence": row["valence"],
                "arousal": row["arousal"],
            })
    return pd.DataFrame(rows)


def add_cumulative_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df["cumulative_valence"] = pd.Series(dtype=float)
        df["cumulative_arousal"] = pd.Series(dtype=float)
        return df
    df = df.sort_values(["transcript", "turn"])
    for dim in ["valence", "arousal"]:
        df[f"cumulative_{dim}"] = (
            df.groupby("transcript")[dim]
            .expanding()
            .mean()
            .reset_index(level=0, drop=True)
        )
    return df


def summarize_cumulative(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = (
        df.groupby("turn", as_index=False)
        .agg(
            mean_cumulative_valence=("cumulative_valence", "mean"),
            sem_valence=("cumulative_valence", "sem"),
            mean_cumulative_arousal=("cumulative_arousal", "mean"),
            sem_arousal=("cumulative_arousal", "sem"),
        )
        .sort_values("turn")
    )
    out[["sem_valence", "sem_arousal"]] = out[["sem_valence", "sem_arousal"]].fillna(0)
    return out

def plot_va_circumplex(all_summaries: dict, out_dir: Path, model: str):
    fig, ax = plt.subplots(figsize=(7, 6))

    for mode in ["baseline", "cot", "mcot"]:
        df = all_summaries.get(mode)
        if df is None or df.empty:
            continue

        x = df["mean_cumulative_valence"].to_numpy()
        y = df["mean_cumulative_arousal"].to_numpy()
        turns = df["turn"].to_numpy()
        cmap = CIRCUMPLEX_CMAPS[mode]
        norm = Normalize(vmin=turns.min(), vmax=turns.max())

        for i in range(len(x) - 1):
            ax.plot(
                x[i:i+2], y[i:i+2],
                color=cmap(0.35 + 0.65 * norm(turns[i])),
                linewidth=3,
                solid_capstyle="round",
            )

        mid_color = CIRCUMPLEX_LEGEND_COLORS[mode]
        ax.scatter(x[0], y[0], color="white", edgecolors=mid_color,
                   marker="o", s=120, linewidths=2, zorder=6)
        ax.scatter(x[-1], y[-1], color="white", edgecolors=mid_color,
                   marker="*", s=250, linewidths=1.5, zorder=6)

    ax.set_xlabel("Valence", fontsize=14)
    ax.set_ylabel("Arousal", fontsize=14)

    legend_elements = [
        Line2D([0], [0], color=CIRCUMPLEX_LEGEND_COLORS[mode],
               linewidth=3, label=MODE_LABELS[mode])
        for mode in ["baseline", "cot", "mcot"]
    ] + [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="gray", markersize=9, label="Start"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="white",
               markeredgecolor="gray", markersize=12, label="End"),
    ]

    ax.set_title("Synthetic Transcript: User Valence-Arousal Trajectory")
    ax.legend(handles=legend_elements, fontsize=12, frameon=True, loc="best")
    ax.annotate("Dark → early turns   Bright → late turns",
                xy=(0.02, 0.02), xycoords="axes fraction",
                fontsize=9, color="gray")

    plt.tight_layout()
    plt.savefig(out_dir / f"{model}_va_circumplex.png", dpi=300, bbox_inches="tight")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model folder name inside output/")
    args = parser.parse_args()

    model = args.model
    out_dir = SAVE_ROOT / model
    out_dir.mkdir(parents=True, exist_ok=True)

    vad_dict = load_nrc_vad(NRC_VAD_DIR)
    grouped_files = load_transcript_files(model)

    cumulative_summaries = {}

    for mode in ["baseline", "cot", "mcot"]:
        files = grouped_files[mode]
        raw_df = build_raw_df(files, vad_dict, mode)
        raw_with_cum = add_cumulative_sentiment(raw_df)
        cumulative_summary = summarize_cumulative(raw_with_cum)
        cumulative_summary.to_csv(out_dir / f"{model}_{mode}_cumulative_summary.csv", index=False)
        cumulative_summaries[mode] = cumulative_summary

    plot_va_circumplex(cumulative_summaries, out_dir, model)

    print(f"\nSaved to: {out_dir}")
    print("  - va_circumplex.png")
    print("  - {mode}_cumulative_summary.csv  (for each mode)")

    for mode in ["baseline", "cot", "mcot"]:
        files = grouped_files[mode]
        if files:
            sample_messages = extract_messages(files[0])
            print(f"[{mode}] ", end="")
            coverage_report(sample_messages, vad_dict)


if __name__ == "__main__":
    main()