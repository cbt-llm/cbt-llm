"""
Overlay VA circumplex: RealCBT + GPT-MCoT + Mistral-MCoT + Gemma-MCoT
"""
from pathlib import Path
import re
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "output"
DATA_ROOT = PROJECT_ROOT / "data" / "external" / "RealCBT" / "RealCBT_Dataset"
NRC_VAD_DIR = PROJECT_ROOT / "external_libs" / "NRC-VAD-Lexicon-v2.1"
SAVE_ROOT = PROJECT_ROOT / "evaluation" / "overlay_all_sentiment"
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

SKIP_FIRST_N_UTTERANCES = 2
MIN_SESSION_COVERAGE = 20

MODELS = ["gpt", "mistral", "gemma"]

# Per-condition visual style: (color, label, linestyle)
STYLES = {
    "realcbt":  ("#4C72B0", "RealCBT",      "-"),
    "gpt":      ("#DD8452", "GPT-MCoT",     "-"),
    "mistral":  ("#55A868", "Mistral-MCoT", "-"),
    "gemma":    ("#C44E52", "Gemma-MCoT",   "-"),
}


# ---------------------------------------------------------------------------
# NRC VAD
# ---------------------------------------------------------------------------

def load_nrc_vad(lexicon_dir: Path) -> dict:
    unigrams_dir = lexicon_dir / "Unigrams"

    def _load_dim(filename):
        fp = (unigrams_dir / filename).resolve()
        if not fp.exists():
            raise FileNotFoundError(f"Not found: {fp}")
        df = pd.read_csv(fp, sep="\t", header=0)
        df.columns = [c.strip().lower() for c in df.columns]
        word_col, score_col = df.columns[0], df.columns[1]
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
        df = df.dropna(subset=[score_col])
        return df.set_index(word_col)[score_col].to_dict()

    valence = _load_dim("unigrams-valence-NRC-VAD-Lexicon-v2.1.txt")
    arousal = _load_dim("unigrams-arousal-NRC-VAD-Lexicon-v2.1.txt")
    common = set(valence) & set(arousal)
    vad_dict = {w: {"valence": valence[w], "arousal": arousal[w]} for w in common}
    print(f"NRC VAD loaded: {len(vad_dict):,} unigrams")
    return vad_dict


def score_tokens(text: str, vad_dict: dict):
    tokens = re.findall(r"\b[a-z]+\b", text.lower())
    matched = [vad_dict[t] for t in tokens if t in vad_dict]
    if not matched:
        return None
    return {
        "valence": float(np.mean([m["valence"] for m in matched])),
        "arousal": float(np.mean([m["arousal"] for m in matched])),
    }


# ---------------------------------------------------------------------------
# RealCBT loader
# ---------------------------------------------------------------------------

_CLIENT_RE = re.compile(r"^Client[：:]\s*(.+)", re.IGNORECASE)


def parse_client_utterances(fp: Path) -> list[str]:
    utterances, current = [], []
    with fp.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                if current:
                    utterances.append(" ".join(current))
                    current = []
                continue
            m = _CLIENT_RE.match(line)
            if m:
                if current:
                    utterances.append(" ".join(current))
                current = [m.group(1).strip()]
            elif re.match(r"^(Counselor|Therapist)[：:]", line, re.IGNORECASE):
                if current:
                    utterances.append(" ".join(current))
                    current = []
            else:
                if current:
                    current.append(line)
    if current:
        utterances.append(" ".join(current))
    return [u for u in utterances if u]


def build_realcbt_df(data_dir: Path, vad_dict: dict) -> pd.DataFrame:
    rows = []
    files = sorted(data_dir.glob("*.txt"), key=lambda f: int(re.search(r"\d+", f.stem).group()))
    for fp in files:
        utterances = parse_client_utterances(fp)[SKIP_FIRST_N_UTTERANCES:]
        for i, text in enumerate(utterances):
            scored = score_tokens(text, vad_dict)
            if scored:
                rows.append({"session": fp.stem, "turn": i + 1, **scored})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LLM transcript loader
# ---------------------------------------------------------------------------

def classify_mode(filename: str):
    name = filename.lower()
    if name.startswith("cbt_mcot_"):
        return "mcot"
    return None


def build_model_df(model: str, vad_dict: dict) -> pd.DataFrame:
    model_dir = OUTPUT_ROOT / model
    rows = []
    files = sorted(
        (fp for fp in model_dir.glob("*.json") if classify_mode(fp.name) == "mcot"),
        key=lambda fp: int(re.search(r"(\d+)(?=\.json$)", fp.name).group(1))
    )
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        messages = obj.get("transcript", []) if isinstance(obj, dict) else []
        for item in messages:
            patient = item.get("patient", {})
            if not isinstance(patient, dict):
                continue
            if str(patient.get("role", "")).lower() not in {"patient", "user"}:
                continue
            text = patient.get("query", "").strip()
            if not text:
                continue
            turn_idx = item.get("turn", len(rows))
            scored = score_tokens(text, vad_dict)
            if scored:
                rows.append({"session": fp.stem, "turn": int(turn_idx) + 1, **scored})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cumulative mean + summary
# ---------------------------------------------------------------------------

def add_cumulative(df: pd.DataFrame, session_col: str = "session") -> pd.DataFrame:
    df = df.sort_values([session_col, "turn"])
    for dim in ["valence", "arousal"]:
        df[f"cumulative_{dim}"] = (
            df.groupby(session_col)[dim]
            .expanding()
            .mean()
            .reset_index(level=0, drop=True)
        )
    return df


def summarize(df: pd.DataFrame, session_col: str = "session") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = (
        df.groupby("turn", as_index=False)
        .agg(
            mean_cumulative_valence=("cumulative_valence", "mean"),
            sem_valence=("cumulative_valence", "sem"),
            mean_cumulative_arousal=("cumulative_arousal", "mean"),
            sem_arousal=("cumulative_arousal", "sem"),
            n=(session_col, "count"),
        )
        .sort_values("turn")
    )
    out[["sem_valence", "sem_arousal"]] = out[["sem_valence", "sem_arousal"]].fillna(0)
    return out


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_overlay(summaries: dict[str, pd.DataFrame], out_path: Path):
    """
    summaries: keys are condition names (realcbt, gpt, mistral, gemma)
    Each df has columns: turn, mean_cumulative_valence, mean_cumulative_arousal, n
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for cond, df in summaries.items():
        if df.empty:
            print(f"  [skip] {cond}: no data")
            continue

        if "n" in df.columns:
            df = df[df["n"] >= MIN_SESSION_COVERAGE].copy()

        x = df["mean_cumulative_valence"].to_numpy()
        y = df["mean_cumulative_arousal"].to_numpy()

        color, label, ls = STYLES[cond]
        ax.plot(x, y, color=color, label=label, linewidth=2.5,
                linestyle=ls, solid_capstyle="round")

        # start marker
        ax.scatter(x[0], y[0], color=color, edgecolors="white",
                   marker="o", s=100, zorder=6, linewidths=1.5)
        # end marker
        ax.scatter(x[-1], y[-1], color=color, edgecolors="white",
                   marker="*", s=220, zorder=6, linewidths=1.5)

    ax.set_xlabel("Valence (cumulative mean)", fontsize=13)
    ax.set_ylabel("Arousal (cumulative mean)", fontsize=13)
    ax.set_title("User/Client Valence–Arousal Trajectory\n(RealCBT vs. LLM MCoT models)", fontsize=13)
    ax.legend(fontsize=11, frameon=True, loc="best")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    vad_dict = load_nrc_vad(NRC_VAD_DIR)

    summaries = {}

    # RealCBT
    print("\nLoading RealCBT...")
    raw = build_realcbt_df(DATA_ROOT, vad_dict)
    if raw.empty:
        print("  WARNING: no RealCBT data found — check DATA_ROOT")
    else:
        summaries["realcbt"] = summarize(add_cumulative(raw, "session"), "session")
        print(f"  {raw['session'].nunique()} sessions, {len(raw)} scored utterances")

    # LLM models
    for model in MODELS:
        print(f"\nLoading {model} MCoT...")
        raw = build_model_df(model, vad_dict)
        if raw.empty:
            print(f"  WARNING: no data for {model}")
        else:
            summaries[model] = summarize(add_cumulative(raw, "session"), "session")
            print(f"  {raw['session'].nunique()} transcripts, {len(raw)} scored turns")

    # Save CSVs
    for cond, df in summaries.items():
        df.to_csv(SAVE_ROOT / f"{cond}_summary.csv", index=False)

    # Plot
    plot_overlay(summaries, SAVE_ROOT / "overlay_va_circumplex.png")


if __name__ == "__main__":
    main()
