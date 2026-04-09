from pathlib import Path
import argparse
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LinearSegmentedColormap


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "output"
SAVE_ROOT = PROJECT_ROOT / "evaluation" / "user_sentiment_plots"
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

NRC_VAD_DIR = PROJECT_ROOT / "external_libs" / "NRC-VAD-Lexicon-v2.1"

CIRCUMPLEX_CMAP = LinearSegmentedColormap.from_list(
    "truncated_purples",
    cm.Purples(np.linspace(0.35, 1.0, 256))
)


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


def load_transcript_with_metadata(fp: Path) -> tuple[list, str]:
    with fp.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    messages = obj.get("transcript", []) if isinstance(obj, dict) else []
    core_issue = obj.get("metadata", {}).get("core_issue", "unknown") if isinstance(obj, dict) else "unknown"
    return messages, core_issue


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
    df = all_summaries.get("mcot")
    if df is None or df.empty:
        print("No mcot data found.")
        return

    x = df["mean_cumulative_valence"].to_numpy()
    y = df["mean_cumulative_arousal"].to_numpy()
    turns = df["turn"].to_numpy()

    norm = Normalize(vmin=turns.min(), vmax=turns.max())
    cmap = CIRCUMPLEX_CMAP

    fig, ax = plt.subplots(figsize=(7, 6))

    for i in range(len(x) - 1):
        ax.plot(x[i:i+2], y[i:i+2],
                color=cmap(norm(turns[i])),
                linewidth=3, solid_capstyle="round")

    ax.scatter(x[0], y[0], color="white", edgecolors="black",
               marker="o", s=120, zorder=6, label="Conversation Start")
    ax.scatter(x[-1], y[-1], color="white", edgecolors="black",
               marker="*", s=250, zorder=6, label="Conversation End")

    pad = 0.01
    ax.set_xlim(x.min() - pad, x.max() + pad)
    ax.set_ylim(y.min() - pad, y.max() + pad)
    ax.invert_yaxis()

    ax.set_title("CBT-MCoT: User Valence-Arousal Trajectory")
    ax.set_xlabel("Valence", fontsize=14)
    ax.set_ylabel("Arousal", fontsize=14)
    ax.legend(fontsize=12, frameon=True, loc="upper right")

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="User Turn", shrink=0.85)

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

    for mode in ["mcot"]:
        files = grouped_files[mode]
        raw_df = build_raw_df(files, vad_dict, mode)
        raw_with_cum = add_cumulative_sentiment(raw_df)
        cumulative_summary = summarize_cumulative(raw_with_cum)
        cumulative_summary.to_csv(out_dir / f"{model}_mcot_cumulative_summary.csv", index=False)
        cumulative_summaries[mode] = cumulative_summary

    plot_va_circumplex(cumulative_summaries, out_dir, model)

    print(f"\nSaved to: {out_dir}")
    print(f"  - {model}_va_circumplex.png")
    print(f"  - {model}_mcot_cumulative_summary.csv")

    files = grouped_files["mcot"]
    if files:
        sample_messages = extract_messages(files[0])
        print("[mcot] ", end="")
        coverage_report(sample_messages, vad_dict)


if __name__ == "__main__":
    main()