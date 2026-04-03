from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "external" / "RealCBT" / "RealCBT_Dataset"
SAVE_ROOT = PROJECT_ROOT / "evaluation" / "realcbt_sentiment_plots"
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

SKIP_FIRST_N_UTTERANCES = 2
MIN_SESSION_COVERAGE = 20

# Download NRC VAD v2.1 lexicon directory (https://saifmohammad.com/WebPages/nrc-vad.html) and please place the unzipped folder at: project/external_libs/
NRC_VAD_DIR = PROJECT_ROOT / "external_libs" / "NRC-VAD-Lexicon-v2.1"

CONDITION_LABEL = "RealCBT"
CONDITION_COLOR = "#4C72B0"
CIRCUMPLEX_CMAP = cm.plasma

def load_nrc_vad(lexicon_dir: Path) -> dict:
    unigrams_dir = lexicon_dir / "Unigrams"

    def _load_dim(filename: str) -> dict:
        fp = (unigrams_dir / filename).resolve()
        if not fp.exists():
            raise FileNotFoundError(f"Not found: {fp}")
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

_CLIENT_RE = re.compile(r"^Client[：:]\s*(.+)", re.IGNORECASE)


def parse_client_utterances(fp: Path) -> list[str]:
    utterances = []
    current = []

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

def score_utterances(utterances: list[str], vad_dict: dict) -> list[dict]:
    rows = []
    for i, text in enumerate(utterances):
        tokens = re.findall(r"\b[a-z]+\b", text.lower())
        matched = [vad_dict[t] for t in tokens if t in vad_dict]
        if not matched:
            continue
        rows.append({
            "turn": i + 1,
            "valence": float(np.mean([m["valence"] for m in matched])),
            "arousal": float(np.mean([m["arousal"] for m in matched])),
        })
    return rows


def coverage_report(utterances: list[str], vad_dict: dict, label: str = ""):
    total = sum(len(re.findall(r"\b[a-z]+\b", u.lower())) for u in utterances)
    matched = sum(
        sum(1 for t in re.findall(r"\b[a-z]+\b", u.lower()) if t in vad_dict)
        for u in utterances
    )
    tag = f"[{label}] " if label else ""
    print(f"{tag}Coverage: {matched}/{total} = {matched/total:.1%}" if total else f"{tag}No tokens.")


def build_raw_df(data_dir: Path, vad_dict: dict) -> pd.DataFrame:
    rows = []
    files = sorted(data_dir.glob("*.txt"), key=lambda f: int(re.search(r"\d+", f.stem).group()))

    all_utterances = []

    for fp in files:
        utterances = parse_client_utterances(fp)
        all_utterances.extend(utterances)

        utterances_trimmed = utterances[SKIP_FIRST_N_UTTERANCES:]
        scored = score_utterances(utterances_trimmed, vad_dict)

        for row in scored:
            rows.append({
                "file": fp.stem,
                "turn": row["turn"],
                "valence": row["valence"],
                "arousal": row["arousal"],
            })

    coverage_report(all_utterances, vad_dict, label="RealCBT full dataset")
    return pd.DataFrame(rows)


def add_cumulative(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values(["file", "turn"])
    for dim in ["valence", "arousal"]:
        df[f"cumulative_{dim}"] = (
            df.groupby("file")[dim]
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
            n=("cumulative_valence", "count"),
        )
        .sort_values("turn")
    )
    out[["sem_valence", "sem_arousal"]] = out[["sem_valence", "sem_arousal"]].fillna(0)
    return out

def _plot_dim(summary, y_col, sem_col, y_label, out_path):
    df = summary[summary["n"] >= MIN_SESSION_COVERAGE].copy()
    plt.figure(figsize=(7.5, 5))
    x = df["turn"].to_numpy()
    y = df[y_col].to_numpy()
    sem = df[sem_col].to_numpy()
    plt.plot(x, y, marker="o", linewidth=2, color=CONDITION_COLOR, label=CONDITION_LABEL)
    plt.fill_between(x, y - sem, y + sem, color=CONDITION_COLOR, alpha=0.15)
    plt.xlabel("Client Utterance Number", fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.legend(loc="lower right", frameon=True, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_grid(summary, out_path):
    df = summary[summary["n"] >= MIN_SESSION_COVERAGE].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    dims = [
        ("mean_cumulative_valence", "sem_valence", "Running Average Valence"),
        ("mean_cumulative_arousal", "sem_arousal", "Running Average Arousal"),
    ]
    for ax, (y_col, sem_col, y_label) in zip(axes, dims):
        x = df["turn"].to_numpy()
        y = df[y_col].to_numpy()
        sem = df[sem_col].to_numpy()
        ax.plot(x, y, marker="o", linewidth=2, color=CONDITION_COLOR, label=CONDITION_LABEL)
        ax.fill_between(x, y - sem, y + sem, color=CONDITION_COLOR, alpha=0.15)
        ax.set_xlabel("Client Utterance Number", fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        ax.legend(loc="lower right", frameon=True, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_va_circumplex(summary, out_path):
    df = summary[summary["n"] >= MIN_SESSION_COVERAGE].copy()

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
               marker="o", s=120, zorder=6, label="Start")
    ax.scatter(x[-1], y[-1], color="white", edgecolors="black",
               marker="*", s=250, zorder=6, label="End")

    pad = 0.01
    ax.set_title("RealCBT Dataset: User Valence-Arousal Trajectory")
    ax.set_xlim(x.min() - pad, x.max() + pad)
    ax.set_ylim(y.min() - pad, y.max() + pad)

    ax.set_xlabel("Valence", fontsize=14)
    ax.set_ylabel("Arousal", fontsize=14)
    ax.legend(fontsize=12, frameon=True, loc="upper left")

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Client Utterance", shrink=0.85)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def main():
    print(f"DATA_ROOT : {DATA_ROOT}")
    print(f"NRC_VAD_DIR: {NRC_VAD_DIR}\n")

    vad_dict = load_nrc_vad(NRC_VAD_DIR)
    raw_df = build_raw_df(DATA_ROOT, vad_dict)

    if raw_df.empty:
        print("No data found — check DATA_ROOT and file format.")
        return

    raw_with_cum = add_cumulative(raw_df)
    summary = summarize_cumulative(raw_with_cum)

    summary.to_csv(SAVE_ROOT / "realcbt_cumulative_summary.csv", index=False)

    plot_va_circumplex(summary, out_path=SAVE_ROOT / "realcbt_va_circumplex.png")

    print(f"\nSaved to: {SAVE_ROOT}")
    print("  - realcbt_cumulative_valence.png")
    print("  - realcbt_cumulative_arousal.png")
    print("  - realcbt_valence_arousal_grid.png")
    print("  - realcbt_va_circumplex.png")
    print("  - realcbt_cumulative_summary.csv")

    print(f"\nFiles processed : {raw_df['file'].nunique()}")
    print(f"Total scored utterances: {len(raw_df)}")
    print(f"Mean valence : {raw_df['valence'].mean():.3f}")
    print(f"Mean arousal : {raw_df['arousal'].mean():.3f}")


if __name__ == "__main__":
    main()