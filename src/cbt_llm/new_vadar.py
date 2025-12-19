import os, json, glob, argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER_BACKEND = "vaderSentiment"
except Exception:
    SentimentIntensityAnalyzer = None
    _VADER_BACKEND = "missing"


ROLE_MAP = {
    "patient": "human",
    "client": "human",
    "human": "human",
    "user": "human",

    "therapist": "therapist",
    "counselor": "therapist",
    "assistant": "therapist",
    "model": "therapist",
}

def norm_role(s):
    if not isinstance(s, str):
        return "unknown"
    return ROLE_MAP.get(s.strip().lower(), s.strip().lower())

def get_text(m):
    """Extract text from various message shapes."""
    if isinstance(m, str):
        return m
    if not isinstance(m, dict):
        return None
    if isinstance(m.get("content"), str):
        return m["content"]
    if isinstance(m.get("text"), str):
        return m["text"]

    c = m.get("content")
    if isinstance(c, list):
        parts = []
        for p in c:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
        return "\n".join(parts) if parts else None

    for k in ("utterance", "value", "message", "msg"):
        if isinstance(m.get(k), str):
            return m[k]
    return None

def read_any_json(path: Path):
    if path.suffix == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return rows
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def to_messages(obj):
    """
    Return list of messages from:
      - list directly
      - dict wrapper with keys including your NEW key: "transcript"
    """
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("transcript", "messages", "turns", "dialogue", "conversation", "chat", "utterances"):
            if isinstance(obj.get(key), list):
                return obj[key]
    return [obj]

def ensure_analyzer():
    global _VADER_BACKEND
    if SentimentIntensityAnalyzer is not None:
        return SentimentIntensityAnalyzer()

    try:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer as NLTK_SIA
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
        _VADER_BACKEND = "nltk"
        return NLTK_SIA()
    except Exception as e:
        raise RuntimeError(
            "VADER analyzer not available. Install one of:\n"
            "  pip install vaderSentiment\n"
            "or\n"
            "  pip install nltk && python -c \"import nltk; nltk.download('vader_lexicon')\"\n"
            f"Original error: {e}"
        )

def build_dataframe(input_dir: Path, pattern: str):
    files = sorted(glob.glob(str(input_dir / "**" / pattern), recursive=True))
    if not files:
        raise RuntimeError(f"No files matched pattern='{pattern}' under {input_dir}")

    analyzer = ensure_analyzer()
    all_rows = []

    for fp in files:
        fp_path = Path(fp)
        session_id = fp_path.stem
        try:
            raw = read_any_json(fp_path)
            msgs = to_messages(raw)

            turn_idx = 0
            for m in msgs:
                text = get_text(m)
                if not (isinstance(text, str) and text.strip()):
                    continue

                role_raw = ""
                if isinstance(m, dict):
                    role_raw = m.get("role") or m.get("speaker") or m.get("author") or ""
                speaker = norm_role(role_raw)

                turn_idx += 1
                scores = analyzer.polarity_scores(text.strip())
                all_rows.append({
                    "session_id": session_id,
                    "turn_id": turn_idx,
                    "speaker": speaker,
                    "text": text.strip(),
                    "compound": scores["compound"],
                    "pos": scores["pos"],
                    "neu": scores["neu"],
                    "neg": scores["neg"],
                    "source_file": str(fp_path),
                })
        except Exception as e:
            print(f"[WARN] Skipping {fp_path.name}: {e}")

    if not all_rows:
        raise RuntimeError("No messages parsed. Check files/schema/pattern.")

    return pd.DataFrame(all_rows)

def per_session_progress(hum: pd.DataFrame):
    rows = []

    for session_id, g in hum.groupby("session_id"):
        s = g["compound"].dropna()
        if len(s) < 2:
            continue

        n = max(1, int(len(s) * 0.2))
        mean_early = float(s.iloc[:n].mean())
        mean_late = float(s.iloc[-n:].mean())

        rows.append({
            "session_id": session_id,
            "mean_early": mean_early,
            "mean_late": mean_late,
            "delta": mean_late - mean_early,
            "n_turns": int(len(s)),
        })

    return pd.DataFrame(rows)



def aggregate_trajectory(hum: pd.DataFrame, bins=20):
    """Normalize each session to 0..1 progress and average across sessions."""
    hum = hum.copy()
    hum["idx_in_session"] = hum.groupby("session_id").cumcount()
    hum["len_session"] = hum.groupby("session_id")["compound"].transform("size")
    hum = hum[hum["len_session"] >= 3]  # ignore ultra-short sessions
    hum["pos01"] = (hum["idx_in_session"] / (hum["len_session"] - 1)).clip(0, 1)
    hum["bin"] = (hum["pos01"] * (bins - 1)).round().astype(int)
    agg = hum.groupby("bin")["compound"].agg(["mean", "count"]).reset_index()
    agg["progress_pct"] = (agg["bin"] / (bins - 1)) * 100.0
    return agg

def plot_per_session_patient(hum: pd.DataFrame, out_dir: Path):
    for session_id, g in hum.groupby("session_id"):
        g = g.sort_values("turn_id")
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(1, len(g) + 1), g["compound"].values, linewidth=2)
        plt.axhline(0, linestyle="--", alpha=0.6)
        plt.title(f"Patient Sentiment (VADER compound) - {session_id}")
        plt.xlabel("Patient turn #")
        plt.ylabel("VADER compound")
        plt.tight_layout()
        plt.savefig(out_dir / f"{session_id}_patient_sentiment.png", dpi=150)
        plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", help="Folder containing transcript JSON files (e.g., output/)")
    ap.add_argument("--out", default="vader_outputs", help="Output folder")
    ap.add_argument(
        "--pattern",
        default="*transcript*.json",
        help="Glob pattern (recursive) used to pick files. Default only reads transcript files.",
    )
    args = ap.parse_args()

    input_dir = Path(os.path.expanduser(args.input_dir)).resolve()
    out_dir = Path(os.path.expanduser(args.out)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataframe(input_dir, pattern=args.pattern)

    # Only patient/human messages
    hum = df[df["speaker"] == "human"].copy()
    if hum.empty:
        raise RuntimeError("No patient/human turns found—check ROLE_MAP or file pattern.")

    # Save raw scored messages
    df.to_csv(out_dir / "all_turns_with_vader.csv", index=False)
    hum.to_csv(out_dir / "patient_turns_with_vader.csv", index=False)

    # Per-session progress
    progress = per_session_progress(hum)
    progress.to_csv(out_dir / "session_progress_summary.csv", index=False)

    plot_per_session_patient(hum, out_dir)

    # Plot 1: Progress across sessions (Δ)
    prog_plot = progress.dropna(subset=["delta"]).sort_values("delta")
    if not prog_plot.empty:
        plt.figure(figsize=(10, 5))
        plt.barh(prog_plot["session_id"].astype(str), prog_plot["delta"])
        plt.axvline(0, linestyle="--", alpha=0.6)
        plt.title("Per-Session Progress (Δ compound: last 20% − first 20%)")
        plt.xlabel("Δ VADER compound")
        plt.tight_layout()
        plt.savefig(out_dir / "progress_across_sessions.png", dpi=150)
        plt.close()

    # Plot 2: Average normalized trajectory
    traj = aggregate_trajectory(hum, bins=20)
    if not traj.empty:
        plt.figure(figsize=(10, 4))
        plt.plot(traj["progress_pct"], traj["mean"], linewidth=2)
        plt.axhline(0, linestyle="--", alpha=0.6)
        plt.title("Average Patient Sentiment Trajectory (All Sessions, normalized)")
        plt.xlabel("Conversation progress (%)")
        plt.ylabel("Mean VADER compound")
        plt.tight_layout()
        plt.savefig(out_dir / "avg_normalized_trajectory.png", dpi=150)
        plt.close()

    overall_delta = progress["delta"].dropna()
    print(f"VADER backend: {_VADER_BACKEND}")
    if not overall_delta.empty:
        print(
            f"Sessions: {progress['session_id'].nunique()} | "
            f"Mean Δ: {overall_delta.mean():.3f} | "
            f"Median Δ: {overall_delta.median():.3f}"
        )
    print("Done. Outputs in:", out_dir)

if __name__ == "__main__":
    main()
