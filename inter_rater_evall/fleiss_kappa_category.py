"""
Protocol Kappa Calculator
=========================
Calculates percentage breakdown and Fleiss' Kappa for protocol CSV files
with categorical values: 'Keep the same', 'Other protocol', 'Neither'

CSV format expected:
  - First column: Rater label (e.g. Rater_1)
  - Remaining columns: category per turn
  - First row: header

Usage:
  python protocol_kappa.py                          (runs on default files)
  python protocol_kappa.py file1.csv file2.csv ...  (runs on specified files)

Output:
  Prints summary table + saves protocol_summary.csv
"""

import csv
import sys
from pathlib import Path

CATEGORIES = ['Keep the same', 'Other protocol', 'Neither']


def load_csv(filepath):
    """Load CSV, skip header row and first (rater label) column."""
    with open(filepath, newline='') as f:
        rows = list(csv.reader(f))
    data = []
    for row in rows[1:]:
        data.append(row[1:])
    return data


def get_percentages(data):
    """Count how often each category appears and return as percentages."""
    counts = {c: 0 for c in CATEGORIES}
    total = 0
    for row in data:
        for val in row:
            val = val.strip()
            if val in counts:
                counts[val] += 1
            else:
                counts['Neither'] += 1  # treat unknowns as Neither
            total += 1
    pcts = {c: round(counts[c] / total * 100, 1) for c in CATEGORIES}
    return pcts, counts, total


def fleiss_kappa(data):
    """
    Compute Fleiss' Kappa for categorical data.

    data: list of lists (raters x subjects/turns)
    Transposes to subjects x raters internally.

    Returns kappa, P_bar, P_e
    """
    # Transpose: rows = subjects (turns), cols = raters
    subjects = list(map(list, zip(*data)))
    N = len(subjects)   # number of turns/subjects
    n = len(subjects[0])  # number of raters

    # Count category assignments per subject
    nij = []
    for subj in subjects:
        counts = {c: 0 for c in CATEGORIES}
        for v in subj:
            v = v.strip()
            if v in counts:
                counts[v] += 1
            else:
                counts['Neither'] += 1
        nij.append(counts)

    # Per-subject agreement P_i
    P_i = []
    for i in range(N):
        s = sum(nij[i][c] * (nij[i][c] - 1) for c in CATEGORIES)
        P_i.append(s / (n * (n - 1)))

    # Overall observed agreement
    P_bar = sum(P_i) / N

    # Category proportions p_j
    p_j = {}
    for c in CATEGORIES:
        total = sum(nij[i][c] for i in range(N))
        p_j[c] = total / (N * n)

    # Expected agreement by chance
    P_e = sum(p_j[c] ** 2 for c in CATEGORIES)

    # Fleiss' Kappa
    kappa = (P_bar - P_e) / (1 - P_e) if P_e != 1.0 else 1.0

    return round(kappa, 4), round(P_bar, 4), round(P_e, 4), P_i, p_j


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


def run(files):
    results = []

    # Print header
    print(f"\n{'Model':<12} {'Keep the same':>15} {'Other protocol':>16} {'Neither':>10} {'Fleiss κ':>10} {'Interpretation':<20}")
    print("-" * 85)

    for label, filepath in files:
        path = Path(filepath)
        if not path.exists():
            print(f"  File not found: {filepath}")
            continue

        data = load_csv(filepath)
        pcts, counts, total = get_percentages(data)
        kappa, P_bar, P_e, P_i, p_j = fleiss_kappa(data)
        interp = interpret(kappa)

        print(
            f"{label:<12}"
            f"{str(pcts['Keep the same']) + '%':>15}"
            f"{str(pcts['Other protocol']) + '%':>16}"
            f"{str(pcts['Neither']) + '%':>10}"
            f"{kappa:>10}"
            f"  {interp:<20}"
        )

        # Detailed breakdown
        print(f"  {'Subjects:':<14} {len(list(zip(*data)))}  |  Raters: {len(data)}  |  Total ratings: {total}")
        print(f"  {'P_bar:':<14} {P_bar}   P_e: {P_e}")
        print(f"  Per-turn agreement (P_i):")
        for i, pi in enumerate(P_i):
            print(f"    Turn {i+1:>2}: {pi:.4f}")
        print()

        results.append({
            'Model': label,
            'Keep the same (%)': pcts['Keep the same'],
            'Other protocol (%)': pcts['Other protocol'],
            'Neither (%)': pcts['Neither'],
            'Fleiss Kappa': kappa,
            'Interpretation': interp,
            'P_bar': P_bar,
            'P_e': P_e,
            'Total ratings': total,
            'Subjects': len(list(zip(*data))),
            'Raters': len(data),
        })

    # Save summary CSV
    if results:
        out_path = 'protocol_summary.csv'
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Summary saved to: {out_path}")

    return results


# ── Entry point ────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Files passed as arguments — use filename (without extension) as label
        files = [(Path(f).stem, f) for f in sys.argv[1:]]
    else:
        # Default: run on the 3 protocol files
        files = [
            ('GPT',     'convo1_gpt_protocol.csv'),
            ('Gemma',   'convo2_gemma_protocol.csv'),
            ('Mistral', 'convo3_mistral_protocol.csv'),
        ]

    run(files)