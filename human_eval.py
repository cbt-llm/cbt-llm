from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import krippendorff
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

STUDY1_DIR = Path("human_eval_study_1")
STUDY2_DIR = Path("human_eval_study_2")

MODELS = {
    "mistral": ("Mistral-7B", "MCoT"),
    "deepseek": ("DeepSeek-R1-8B", "CoT"),
    "gemma": ("Gemma3-12B", "MCoT"),
    "gpt": ("GPT-OSS-20B", "CoT"),
}
MODEL_ORDER = ["Mistral-7B", "DeepSeek-R1-8B", "Gemma3-12B", "GPT-OSS-20B"]
SIGNIFICANCE_PAIRS = [
    ("Gemma3-12B", "GPT-OSS-20B"),
    ("Gemma3-12B", "DeepSeek-R1-8B"),
    ("Mistral-7B", "GPT-OSS-20B"),
]

PROTOCOLS = ["V", "SQ", "CR", "O/N"]
CONFIDENCE_MAX = 4
LIKERT = {
    "very inappropriate": 1,
    "inappropriate": 2,
    "inppropriate": 2,
    "neutral": 3,
    "appropriate": 4,
    "very appropriate": 5,
}
S1_PROTOCOL = { 
    "validation & reflection": "V",
    "socratic questioning": "SQ",
    "cognitive restructuring": "CR",
    "neither/other": "O/N",
}

S2_PRINCIPLE = {
    "socratic questioning": "SQ",
    "validation & reflection": "V",
    "alternative perspective": "CR",
    "cognitive restructuring": "CR",
}


# --- response accumulator ------------------------------------------------------
class ModelStats:
    """Confidence mass per protocol plus Study-1 ratings for one model."""

    def __init__(self, mode: str):
        self.mode = mode
        self.mass = dict.fromkeys(PROTOCOLS, 0.0)
        self.scores: list[int] = []           # Study-1 Likert effectiveness
        self.rater_scores: list[list] = []     # per-rater Likert vectors (alpha)
        self.rater_protocols: list[list] = []  # per-rater protocol vectors (alpha)

    def adherence(self) -> dict[str, float]:
        total = sum(self.mass.values())
        return {p: 100.0 * self.mass[p] / total for p in PROTOCOLS}

    def effectiveness(self) -> tuple[float, float]:
        if not self.scores:
            return float("nan"), float("nan")
        return float(np.mean(self.scores)), float(np.var(self.scores))


# --- parsing -----------------------------------------------------------------
def model_of(path: Path) -> tuple[str, str]:
    name = path.name.lower()
    for token, (label, mode) in MODELS.items():
        if token in name:
            return label, mode
    raise ValueError(f"no known model token in filename: {path.name}")


def read_qualtrics(path: Path) -> pd.DataFrame:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, question_text, data = rows[0], rows[1], rows[3:]
    width = len(header)
    data = [(row + [""] * width)[:width] for row in data]
    df = pd.DataFrame(data, columns=header)
    df.attrs["question_text"] = dict(zip(header, question_text))
    return df


def parse_protocols(cell: str) -> list[str]:
    if pd.isna(cell):
        return []
    tokens = (t.strip().lower().replace(" and ", " & ") for t in str(cell).split(","))
    return [S1_PROTOCOL[t] for t in tokens if t in S1_PROTOCOL]


def add_study1(stats: ModelStats, df: pd.DataFrame) -> None:
    protocol_cols = [c for c in df.columns if c.endswith("a")]
    score_cols = [c for c in df.columns if c.endswith("b_1")]
    for _, row in df.iterrows():
        turn_scores, turn_protocols = [], []
        for protocol_col, score_col in zip(protocol_cols, score_cols):
            protocols = parse_protocols(row[protocol_col])
            if not protocols:
                continue
            for protocol in protocols:
                stats.mass[protocol] += CONFIDENCE_MAX
            label = str(row[score_col]).strip().lower()
            if label in LIKERT:
                stats.scores.append(LIKERT[label])
                turn_scores.append(LIKERT[label])
            turn_protocols.append(protocols[0])
        if len(turn_scores) > 1:
            stats.rater_scores.append(turn_scores)
        if len(turn_protocols) > 1:
            stats.rater_protocols.append([PROTOCOLS.index(p) for p in turn_protocols])


def add_study2(stats: ModelStats, df: pd.DataFrame) -> None:
    question_text = df.attrs["question_text"]
    for col in df.columns:
        text = question_text.get(col, "")
        if "Rate your confidence" not in text or " - " not in text:
            continue
        principle = text.rsplit(" - ", 1)[-1].strip().lower()
        protocol = S2_PRINCIPLE.get(principle)
        if protocol is None:
            print(f"unmapped Study 2 principle: {principle!r}", file=sys.stderr)
            continue
        confidence = pd.to_numeric(df[col], errors="coerce").dropna().clip(0, CONFIDENCE_MAX)
        stats.mass[protocol] += confidence.sum()


def collect(study1_dir: Path, study2_dir: Path) -> dict[str, ModelStats]:
    stats: dict[str, ModelStats] = {}
    for directory, add in [(study1_dir, add_study1), (study2_dir, add_study2)]:
        for path in sorted(directory.glob("*.csv")):
            label, mode = model_of(path)
            add(stats.setdefault(label, ModelStats(mode)), read_qualtrics(path))
    return stats


def krippendorff_alpha(rater_vectors: list[list], level: str) -> float:
    if not rater_vectors:
        return float("nan")
    width = max(len(v) for v in rater_vectors)
    padded = [v + [np.nan] * (width - len(v)) for v in rater_vectors]
    return krippendorff.alpha(reliability_data=np.array(padded).T,
                              level_of_measurement=level)


def adherence_table(stats: dict[str, ModelStats]) -> pd.DataFrame:
    order = [m for m in MODEL_ORDER if m in stats]
    order += [m for m in stats if m not in order]
    records = []
    for label in order:
        s = stats[label]
        pct = s.adherence()
        mean, var = s.effectiveness()
        records.append({"Model": label, "Mode": s.mode,
                        **{p: round(pct[p], 1) for p in PROTOCOLS},
                        "Mean": round(mean, 2), "Var": round(var, 2)})
    return pd.DataFrame.from_records(records,
                                     columns=["Model", "Mode", *PROTOCOLS, "Mean", "Var"])


def report(stats: dict[str, ModelStats]) -> None:
    print(adherence_table(stats).to_string(index=False))

    eff = krippendorff_alpha([v for s in stats.values() for v in s.rater_scores],
                             "ordinal")
    adh = krippendorff_alpha([v for s in stats.values() for v in s.rater_protocols],
                             "nominal")
    print(f"\nKrippendorff's alpha  effectiveness={eff:.2f}  adherence={adh:.2f}")

    for a, b in SIGNIFICANCE_PAIRS:
        if stats.get(a) and stats.get(b) and stats[a].scores and stats[b].scores:
            _, p = mannwhitneyu(stats[a].scores, stats[b].scores,
                                alternative="two-sided")
            print(f"{a} vs {b}: p={p:.4f}")


PREFERENCE = ["Keep", "Other", "Neither"]  # Study 2 Part B categories


def classify_preference(value: str) -> str | None:
    choice = value.strip().lower()
    if choice == "keep the same":
        return "Keep"
    if choice == "neither":
        return "Neither"
    return "Other" if choice else None  # any rewrite counts as Other; blank is skipped


def preference_table(study2_dir: Path) -> pd.DataFrame:
    """Table VI: Study-2 expert preference (Keep/Other/Neither) and agreement (kappa)."""
    counts: dict[str, Counter] = defaultdict(Counter)
    choices: dict[str, list[list]] = defaultdict(list)  # per-rater category codes
    for path in sorted(study2_dir.glob("*.csv")):
        label, _ = model_of(path)
        df = read_qualtrics(path)
        cols = [c for c in df.columns if c.startswith("Conversation") and c.endswith("b")]
        for _, row in df.iterrows():
            vector = []
            for col in cols:
                category = classify_preference(str(row[col]))
                vector.append(np.nan if category is None else PREFERENCE.index(category))
                if category is not None:
                    counts[label][category] += 1
            if any(pd.notna(v) for v in vector):
                choices[label].append(vector)

    order = [m for m in MODEL_ORDER if m in counts]
    order += [m for m in counts if m not in order]
    records = []
    for label in order:
        total = sum(counts[label].values())
        record = {"Model": label}
        record |= {f"{c} (%)": round(100 * counts[label][c] / total, 1) for c in PREFERENCE}
        record["κ"] = round(krippendorff_alpha(choices[label], "nominal"), 3)
        records.append(record)
    return pd.DataFrame.from_records(
        records, columns=["Model", *(f"{c} (%)" for c in PREFERENCE), "κ"])

def main() -> None:
    study1 = Path(sys.argv[1]) if len(sys.argv) > 1 else STUDY1_DIR
    study2 = Path(sys.argv[2]) if len(sys.argv) > 2 else STUDY2_DIR
    report(collect(study1, study2))
    print()
    print(preference_table(study2).to_string(index=False))

if __name__ == "__main__":
    main()