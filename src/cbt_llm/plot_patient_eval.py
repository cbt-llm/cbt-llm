"""
plot_patient_eval.py

Primary patient-level evaluation plots for CBT experiments.

Produces (per model):

1. Patient Distress Trajectory (Baseline vs CBT)
   - Mean patient sentiment trajectory across conversation
   - With uncertainty band (SEM)
   - One plot per model

2. Protocol-Based Patient Effect Plot
   - Mean change in patient sentiment on the next patient turn
   - Weighted by continuous protocol execution scores
   - Bar chart with uncertainty (NO dots, NO lines)

Assumptions:
- Transcripts live in: output/{model}/*.json
- Judge summaries live in: evaluation/{model}/summary.csv
- Uses VADER compound score as proxy for distress
"""

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# =========================
# CONFIG
# =========================

OUTPUT_ROOT = Path("output")
EVAL_ROOT = Path("evaluation")
PLOTS_OUT = EVAL_ROOT / "patient_plots"
PLOTS_OUT.mkdir(parents=True, exist_ok=True)

MODEL_DISPLAY = {
    "gpt": "GPT-4o-mini",
    "gemma": "Gemma-2-9B",
    "mistral": "Mistral-7B-Instruct",
    "qwen": "Qwen-3-4B",
    "deepseek": "DeepSeek-R1-8B",
}

BINS = 20  # normalized trajectory bins


# =========================
# HELPERS
# =========================

def model_title(model: str) -> str:
    return MODEL_DISPLAY.get(model, model.upper())

def is_cbt_file(fp: Path) -> bool:
    return fp.name.lower().startswith("cbt_")

def load_transcripts(model: str):
    base = OUTPUT_ROOT / model
    if not base.exists():
        raise FileNotFoundError(f"Missing output directory: {base}")
    return sorted(base.glob("*.json"))

def extract_messages(fp: Path):
    obj = json.loads(fp.read_text())
    return obj.get("transcript", [])

def patient_sentiment(messages, analyzer):
    rows = []
    turn = 0
    for m in messages:
        if m.get("role") != "patient":
            continue
        text = (m.get("content") or "").strip()
        if not text:
            continue
        turn += 1
        s = analyzer.polarity_scores(text)
        rows.append((turn, s["compound"]))
    return rows

def normalize_and_bin(values):
    if len(values) < 3:
        return None
    arr = np.array(values)
    x = np.linspace(0, 1, len(arr))
    bins = np.floor(x * (BINS - 1)).astype(int)
    out = pd.DataFrame({"bin": bins, "val": arr})
    return out.groupby("bin")["val"].mean()


# =========================
# PLOT 1: PATIENT DISTRESS TRAJECTORY
# =========================

def plot_patient_trajectory(model: str, files):
    analyzer = SentimentIntensityAnalyzer()
    baseline, cbt = [], []

    for fp in files:
        msgs = extract_messages(fp)
        vals = patient_sentiment(msgs, analyzer)
        if not vals:
            continue

        _, scores = zip(*vals)
        traj = normalize_and_bin(scores)
        if traj is None:
            continue

        if is_cbt_file(fp):
            cbt.append(traj)
        else:
            baseline.append(traj)

    def aggregate(trajs):
        df = pd.concat(trajs, axis=1)
        mean = df.mean(axis=1).values
        sem = df.sem(axis=1).fillna(0).values
        return mean, sem

    plt.figure(figsize=(6, 4))

    if baseline:
        m, _ = aggregate(baseline)
        x = np.linspace(0, 100, len(m))
        plt.plot(x, m, label="Baseline")

    if cbt:
        m, _ = aggregate(cbt)
        x = np.linspace(0, 100, len(m))
        plt.plot(x, m, label="CBT-guided")

    plt.axhline(0, linestyle="--", alpha=0.4)
    plt.xlabel("Conversation Progress (%)")
    plt.ylabel("Patient Sentiment Score (VADER Compound)")
    plt.title(f"{model_title(model)}: Patient Sentiment Trajectory")
    plt.legend()
    plt.tight_layout()

    plt.savefig(PLOTS_OUT / f"{model}_patient_trajectory.png", dpi=200)
    plt.close()


# =========================
# PLOT 2: PROTOCOL-BASED PATIENT EFFECT
# =========================

def plot_protocol_effect(model: str, files):
    """
    Correct turn-level protocol-based patient effect.

    For each therapist turn:
      change = patient_sentiment(next_patient) - patient_sentiment(previous_patient)
      weight = protocol quality score for that therapist turn (1..5)

    Then compute weighted mean change per protocol.
    """
    analyzer = SentimentIntensityAnalyzer()

    weighted = {
        "Validation": [],
        "Socratic": [],
        "Reframing": [],
    }

    for fp in files:
        if not is_cbt_file(fp):
            continue

        msgs = extract_messages(fp)

        patient_by_msg_idx = {}
        for idx, m in enumerate(msgs):
            if m.get("role") == "patient":
                text = (m.get("content") or "").strip()
                if not text:
                    continue
                patient_by_msg_idx[idx] = analyzer.polarity_scores(text)["compound"]

        if len(patient_by_msg_idx) < 2:
            continue

        judge_fp = (EVAL_ROOT / model / f"{fp.stem}.judge.jsonl")
        if not judge_fp.exists():
            continue

        judge_rows = []
        with judge_fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    judge_rows.append(json.loads(line))
                except Exception:
                    continue

        for jr in judge_rows:
            t_idx = jr.get("_therapist_turn_idx")
            if not isinstance(t_idx, int):
                continue

            prev_patient_idx = t_idx - 1
            next_patient_idx = t_idx + 1

            if prev_patient_idx not in patient_by_msg_idx:
                continue
            if next_patient_idx not in patient_by_msg_idx:
                continue

            change = patient_by_msg_idx[next_patient_idx] - patient_by_msg_idx[prev_patient_idx]

            v = jr.get("validate_and_reflect_quality")
            s = jr.get("socratic_questioning_quality")
            r = jr.get("cognitive_reframing_quality")

            if isinstance(v, (int, float)):
                weighted["Validation"].append((float(v), float(change)))
            if isinstance(s, (int, float)):
                weighted["Socratic"].append((float(s), float(change)))
            if isinstance(r, (int, float)):
                weighted["Reframing"].append((float(r), float(change)))

    labels, means = [], []

    for label in ["Validation", "Socratic", "Reframing"]:
        vals = weighted[label]
        if not vals:
            continue

        w, d = zip(*vals)
        w = np.asarray(w, dtype=float)
        d = np.asarray(d, dtype=float)

        mean = float(np.sum(w * d) / np.sum(w)) if np.sum(w) > 0 else float(np.mean(d))

        labels.append(label)
        means.append(mean)

    plt.figure(figsize=(6, 4))
    plt.bar(labels, means)
    plt.axhline(0, linestyle="--", alpha=0.4)
    plt.ylabel("Change in Patient Sentiment on Next Turn")
    plt.title(f"{model_title(model)}: Effect of CBT Protocols on Patient Sentiment")
    plt.tight_layout()
    plt.savefig(PLOTS_OUT / f"{model}_protocol_patient_effect.png", dpi=200)
    plt.close()


def plot_protocol_effect_across_models(models):
    """
    Aggregate protocol-based patient sentiment change across models.

    For each model:
      - Compute weighted mean patient sentiment change per protocol
      - Then average across protocols to get a single model-level effect

    Produces one bar per model.
    """
    analyzer = SentimentIntensityAnalyzer()

    model_names = []
    model_effects = []

    for model in models:
        files = load_transcripts(model)

        weighted = {
            "Validation": [],
            "Socratic": [],
            "Reframing": [],
        }

        for fp in files:
            if not is_cbt_file(fp):
                continue

            msgs = extract_messages(fp)

            patient_by_msg_idx = {}
            for idx, m in enumerate(msgs):
                if m.get("role") == "patient":
                    text = (m.get("content") or "").strip()
                    if not text:
                        continue
                    patient_by_msg_idx[idx] = analyzer.polarity_scores(text)["compound"]

            if len(patient_by_msg_idx) < 2:
                continue

            judge_fp = (EVAL_ROOT / model / f"{fp.stem}.judge.jsonl")
            if not judge_fp.exists():
                continue

            with judge_fp.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        jr = json.loads(line)
                    except Exception:
                        continue

                    t_idx = jr.get("_therapist_turn_idx")
                    if not isinstance(t_idx, int):
                        continue

                    prev_idx = t_idx - 1
                    next_idx = t_idx + 1

                    if prev_idx not in patient_by_msg_idx or next_idx not in patient_by_msg_idx:
                        continue

                    change = (
                        patient_by_msg_idx[next_idx]
                        - patient_by_msg_idx[prev_idx]
                    )

                    v = jr.get("validate_and_reflect_quality")
                    s = jr.get("socratic_questioning_quality")
                    r = jr.get("cognitive_reframing_quality")

                    if isinstance(v, (int, float)):
                        weighted["Validation"].append((float(v), change))
                    if isinstance(s, (int, float)):
                        weighted["Socratic"].append((float(s), change))
                    if isinstance(r, (int, float)):
                        weighted["Reframing"].append((float(r), change))

        protocol_means = []

        for label in ["Validation", "Socratic", "Reframing"]:
            vals = weighted[label]
            if not vals:
                continue

            w, d = zip(*vals)
            w = np.asarray(w, dtype=float)
            d = np.asarray(d, dtype=float)

            mean = np.sum(w * d) / np.sum(w) if np.sum(w) > 0 else np.mean(d)
            protocol_means.append(mean)

        if protocol_means:
            model_names.append(model_title(model))
            model_effects.append(float(np.mean(protocol_means)))

    plt.figure(figsize=(7, 4.5))
    plt.bar(model_names, model_effects)
    plt.axhline(0, linestyle="--", alpha=0.4)
    plt.ylabel("Average Change in Patient Sentiment")
    plt.title("Effect of CBT-Guided Therapist Behavior on Patient Sentiment")
    plt.tight_layout()

    plt.savefig(
        PLOTS_OUT / "patient_effect_across_models.png",
        dpi=200
    )
    plt.close()


def main(models):
    for model in models:
        files = load_transcripts(model)
        if not files:
            print(f"[WARN] No transcripts for {model}")
            continue

        plot_patient_trajectory(model, files)
        plot_protocol_effect(model, files)
    
    plot_protocol_effect_across_models(models)

    print(f"Plots saved to: {PLOTS_OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model folders inside output/ (e.g. gpt gemma mistral)",
    )
    args = ap.parse_args()
    main(args.models)
