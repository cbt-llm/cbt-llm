"""
Fleiss' Kappa Calculator
========================
Calculates Fleiss' Kappa for inter-rater reliability.

CSV format expected:
  - First column: Rater label (e.g. Rater_1)
  - Remaining columns: scores per turn/subject (0-4)
  - First row: header

Usage:
  python fleiss_kappa.py path/to/your_file.csv
  python fleiss_kappa.py  (runs on all 3 convo files as a demo)
"""

import csv
import sys
from pathlib import Path


def load_csv(filepath):
    """Load CSV into a 2D list of integers (raters x subjects)."""
    with open(filepath, newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Skip header row, skip first column (rater label)
    data = []
    for row in rows[1:]:
        scores = []
        for val in row[1:]:
            try:
                scores.append(int(val))
            except ValueError:
                scores.append(0)  # treat non-numeric as 0
        if scores:
            data.append(scores)
    return data


def fleiss_kappa(data):
    """
    Compute Fleiss' Kappa.

    Parameters
    ----------
    data : list of lists
        Shape (N_subjects x N_raters) — rows are subjects, cols are raters.
        Each cell is a category score.

    Returns
    -------
    kappa   : float
    P_bar   : float  (observed agreement)
    P_e_bar : float  (expected agreement by chance)
    P_i     : list   (per-subject agreement)
    p_j     : dict   (category proportions)
    """
    # Transpose if needed: we want rows = subjects, cols = raters
    N = len(data)          # number of subjects
    n = len(data[0])       # number of raters
    categories = sorted(set(v for row in data for v in row))
    k = len(categories)

    # Count how many raters assigned each category to each subject
    # nij[i][c] = number of raters who assigned category c to subject i
    nij = []
    for row in data:
        counts = {c: 0 for c in categories}
        for v in row:
            if v in counts:
                counts[v] += 1
        nij.append(counts)

    # Per-subject agreement P_i
    P_i = []
    for i in range(N):
        s = sum(nij[i][c] * (nij[i][c] - 1) for c in categories)
        P_i.append(s / (n * (n - 1)))

    # Overall observed agreement
    P_bar = sum(P_i) / N

    # Category proportions p_j
    p_j = {}
    for c in categories:
        total = sum(nij[i][c] for i in range(N))
        p_j[c] = total / (N * n)

    # Expected agreement by chance
    P_e_bar = sum(p_j[c] ** 2 for c in categories)

    # Fleiss' Kappa
    if P_e_bar == 1.0:
        kappa = 1.0
    else:
        kappa = (P_bar - P_e_bar) / (1 - P_e_bar)

    return kappa, P_bar, P_e_bar, P_i, p_j


def interpret(kappa):
    if kappa < 0:
        return "Less than chance"
    elif kappa <= 0.20:
        return "Slight"
    elif kappa <= 0.40:
        return "Fair"
    elif kappa <= 0.60:
        return "Moderate"
    elif kappa <= 0.80:
        return "Substantial"
    else:
        return "Almost perfect"


def run(filepath):
    path = Path(filepath)
    if not path.exists():
        print(f"  File not found: {filepath}")
        return

    data = load_csv(filepath)

    # data from CSV is raters x subjects — transpose to subjects x raters
    subjects_x_raters = list(map(list, zip(*data)))

    kappa, P_bar, P_e_bar, P_i, p_j = fleiss_kappa(subjects_x_raters)

    print(f"\n{'='*55}")
    print(f"  File     : {path.name}")
    print(f"  Subjects : {len(subjects_x_raters)}  |  Raters: {len(data)}")
    print(f"{'='*55}")
    print(f"  Fleiss Kappa (κ)      : {kappa:.4f}")
    print(f"  Observed agreement    : {P_bar:.4f}")
    print(f"  Expected agreement    : {P_e_bar:.4f}")
    print(f"  Interpretation        : {interpret(kappa)}")
    print(f"\n  Category proportions:")
    for c, p in sorted(p_j.items()):
        bar = '█' * int(p * 30)
        print(f"    [{c}]  {p:.4f}  {bar}")
    print(f"\n  Per-subject agreement (P_i):")
    for i, pi in enumerate(P_i):
        print(f"    Turn {i+1:>2} : {pi:.4f}")


# ── Entry point ────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run on files passed as arguments
        for f in sys.argv[1:]:
            run(f)
    else:
        # Default: run on all 6 output CSVs if present
        default_files = [
            "socratic_questioning.csv",
            "validation_reflection.csv",
            "alternative_perspective.csv",
            "convo2_socratic_questioning.csv",
            "convo2_validation_reflection.csv",
            "convo2_alternative_perspective.csv",
            "convo3_socratic_questioning.csv",
            "convo3_validation_reflection.csv",
            "convo3_alternative_perspective.csv",
        ]
        for f in default_files:
            run(f)