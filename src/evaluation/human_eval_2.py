#!/usr/bin/env python3
"""
Analyze Qualtrics convo CSVs for:
  1. A discrepancy table (per file): model's protocol_used vs. what annotators
     actually chose in column `b`, with counts and percentages.
  2. Krippendorff's alpha (per file) on the annotator-chosen protocol per turn
     across N raters (unit = turn, value = protocol category).

Annotator `b` values are normalized to a protocol category:
  - "Keep the same"  -> the model's own protocol_used (they endorsed it)
  - "Neither"        -> "neither" (separate category; treated as disagreement)
  - "" (blank)       -> missing
  - free-text        -> inferred via the same heuristic used for the model response

Usage:
    python analyze_protocol_discrepancy.py convo_1_gpt.csv convo_2_gemma.csv convo_3_mistral.csv
    python analyze_protocol_discrepancy.py --dir ./data --out-dir ./results

Outputs (per input file, written to --out-dir or alongside input):
    <stem>_discrepancy.csv   # rows = model protocol, cols = annotator choice
    <stem>_alpha.txt         # Krippendorff alpha + N raters + N turns
    A combined printout on stdout.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import krippendorff  # pip install krippendorff
import numpy as np

CONV_B_COL_RE = re.compile(r"^Conversation (\d+)\.(\d+)b$")

PROTOCOLS = ["socratic_questioning", "validate_and_reflect", "alternative_perspective"]

ALT_PERSPECTIVE_PATTERNS = [
    r"\bif a (?:close |good |trusted )?friend were in your shoes\b",
    r"\bif a (?:close |good |trusted )?friend were watching you\b",
    r"\bwhat (?:might|would) a (?:close |good |trusted )?friend say\b",
    r"\bwhat (?:might|would) (?:you say|a friend tell)\b.*\bin your shoes\b",
    r"\bwhat do you think a (?:close |good |trusted )?friend\b",
    r"\ba (?:close |good |trusted )?friend might (?:see|say|tell|think)\b",
    r"\bwhat (?:might|would) (?:they|she|he) say if (?:they|she|he) knew\b",
    r"\bwhat do you think (?:your (?:girlfriend|partner|mom|dad|friend)|they) might say\b",
    r"\bwhat do you think (?:might be|is) going through (?:their|her|his) mind\b",
    r"\bwhat (?:might|would) (?:your )?(?:girlfriend|partner|they) say\b",
    r"\bimagine (?:a |your )?(?:friend|loved one)\b",
]


def infer_protocol(text: str) -> str:
    """Heuristic: alternative_perspective > socratic_questioning > validate_and_reflect."""
    t = text.lower()

    for pat in ALT_PERSPECTIVE_PATTERNS:
        if re.search(pat, t):
            return "alternative_perspective"

    if "?" in t:
        openers = (
            r"\b(?:what|how|why|can you tell me|could you|what makes|what goes "
            r"through|what specifically|what do you|what might|when|where)\b"
        )
        for seg in t.split("?")[:-1]:
            if re.search(openers, seg[-200:]):
                return "socratic_questioning"

    return "validate_and_reflect"


def extract_model_response(cell_text: str) -> str:
    m = re.search(
        r"Model Response\s*\d*\s*:(.*?)(?:\n\s*\n|Which principle|Quick guide|$)",
        cell_text, flags=re.DOTALL | re.IGNORECASE,
    )
    return m.group(1).strip() if m else cell_text


def classify_annotator_choice(
    raw: str, model_protocol: str
) -> Optional[str]:
    """Convert a `b` cell value into a protocol label (or None for missing).

    Returns one of: socratic_questioning, validate_and_reflect,
    alternative_perspective, neither, or None (missing).
    """
    v = (raw or "").strip()
    if not v:
        return None
    vl = v.lower()
    # Exact canonical Qualtrics choices
    if vl in {"keep the same", "keep same", "same"}:
        return model_protocol
    if vl == "neither":
        return "neither"
    # Otherwise, it's pasted response text — infer protocol
    return infer_protocol(v)


def read_csv_rows(path: Path) -> List[List[str]]:
    with path.open(newline="") as f:
        return list(csv.reader(f))


def analyze_file(csv_path: Path, out_dir: Path) -> Dict:
    rows = read_csv_rows(csv_path)
    if len(rows) < 4:
        raise ValueError(f"{csv_path}: expected header + 2 metadata rows + 1+ response rows")

    header = rows[0]
    question_row = rows[1]
    data_rows = rows[3:]  # annotator response rows

    # For each turn: find a_1 column (to infer model protocol) and b column
    turns: List[Tuple[int, int, str, int]] = []  # (conv, turn, model_protocol, b_col_idx)
    for i, col in enumerate(header):
        m = CONV_B_COL_RE.match(col.strip())
        if not m:
            continue
        conv, turn = int(m.group(1)), int(m.group(2))
        a1_col = f"Conversation {conv}.{turn}a_1"
        if a1_col not in header:
            print(f"[warn] {csv_path.name}: missing {a1_col}, skipping turn {turn}",
                  file=sys.stderr)
            continue
        a1_idx = header.index(a1_col)
        model_proto = infer_protocol(extract_model_response(question_row[a1_idx]))
        turns.append((conv, turn, model_proto, i))

    n_annotators = len(data_rows)
    n_turns = len(turns)

    # Build: for each turn, list of annotator-chosen protocols (len == n_annotators, may have None)
    per_turn_choices: List[List[Optional[str]]] = []
    for conv, turn_num, model_proto, b_idx in turns:
        choices = []
        for ar in data_rows:
            raw = ar[b_idx] if b_idx < len(ar) else ""
            choices.append(classify_annotator_choice(raw, model_proto))
        per_turn_choices.append(choices)

    # ---- Table 2: discrepancy (counts + percentages) ----
    # Rows = model protocol, Cols = annotator choice (incl. "neither" and "missing")
    cat_order = PROTOCOLS + ["neither", "missing"]
    matrix = {p: Counter() for p in PROTOCOLS}
    for (conv, turn_num, model_proto, _), choices in zip(turns, per_turn_choices):
        for c in choices:
            key = c if c is not None else "missing"
            matrix[model_proto][key] += 1

    # Write discrepancy CSV
    disc_path = out_dir / f"{csv_path.stem}_discrepancy.csv"
    with disc_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_protocol \\ annotator_choice"] + cat_order + ["row_total"])
        for p in PROTOCOLS:
            row_total = sum(matrix[p].values())
            counts = [matrix[p].get(c, 0) for c in cat_order]
            if row_total:
                pcts = [f"{(v / row_total * 100):.1f}%" for v in counts]
                cells = [f"{v} ({pct})" for v, pct in zip(counts, pcts)]
            else:
                cells = [str(v) for v in counts]
            w.writerow([p] + cells + [row_total])
        # Column totals
        col_totals = [sum(matrix[p].get(c, 0) for p in PROTOCOLS) for c in cat_order]
        grand = sum(col_totals)
        w.writerow(["col_total"] + col_totals + [grand])

    # ---- Krippendorff's alpha (nominal) on annotator protocol choice per turn ----
    # krippendorff expects a 2D array: shape (n_raters, n_units).
    # Values are categorical; use string labels -> int codes. Missing = np.nan.
    label_to_code: Dict[str, int] = {}
    def code(v: Optional[str]) -> float:
        if v is None:
            return np.nan
        if v not in label_to_code:
            label_to_code[v] = len(label_to_code)
        return float(label_to_code[v])

    # Build matrix: rows=raters, cols=turns
    rel = np.full((n_annotators, n_turns), np.nan)
    for t_i, choices in enumerate(per_turn_choices):
        for r_i, c in enumerate(choices):
            rel[r_i, t_i] = code(c)

    alpha: Optional[float] = None
    alpha_note = ""
    try:
        # Require at least 2 raters and 2 units with any data
        if n_annotators >= 2 and n_turns >= 2 and np.any(~np.isnan(rel)):
            alpha = krippendorff.alpha(
                reliability_data=rel, level_of_measurement="nominal"
            )
        else:
            alpha_note = "insufficient data"
    except Exception as e:
        alpha_note = f"error: {e}"

    # Write alpha text file
    alpha_path = out_dir / f"{csv_path.stem}_alpha.txt"
    with alpha_path.open("w") as f:
        f.write(f"File: {csv_path.name}\n")
        f.write(f"N annotators (raters): {n_annotators}\n")
        f.write(f"N turns (units):       {n_turns}\n")
        f.write(f"Categories observed:   {sorted(label_to_code.keys())}\n")
        missing_cells = int(np.isnan(rel).sum())
        f.write(f"Missing cells:         {missing_cells} / {n_annotators * n_turns}\n")
        if alpha is not None:
            f.write(f"Krippendorff's alpha (nominal): {alpha:.4f}\n")
        else:
            f.write(f"Krippendorff's alpha: N/A ({alpha_note})\n")

    return {
        "file": csv_path.name,
        "n_annotators": n_annotators,
        "n_turns": n_turns,
        "alpha": alpha,
        "alpha_note": alpha_note,
        "matrix": matrix,
        "categories": cat_order,
        "discrepancy_csv": disc_path,
        "alpha_txt": alpha_path,
    }


def print_discrepancy(result: Dict) -> None:
    print(f"\n{'=' * 70}")
    print(f"File: {result['file']}  "
          f"(raters={result['n_annotators']}, turns={result['n_turns']})")
    print(f"{'=' * 70}")
    cats = result["categories"]
    matrix = result["matrix"]
    # Header
    col_w = 12
    name_w = max(len(p) for p in PROTOCOLS) + 2
    header = "model \\ chosen".ljust(name_w) + "".join(c[:col_w-1].ljust(col_w) for c in cats) + "total"
    print(header)
    print("-" * len(header))
    for p in PROTOCOLS:
        row_total = sum(matrix[p].values())
        cells = []
        for c in cats:
            v = matrix[p].get(c, 0)
            pct = (v / row_total * 100) if row_total else 0
            cells.append(f"{v} ({pct:.0f}%)".ljust(col_w))
        print(p.ljust(name_w) + "".join(cells) + str(row_total))
    if result["alpha"] is not None:
        print(f"\nKrippendorff's α (nominal, per-turn protocol): {result['alpha']:.4f}")
    else:
        print(f"\nKrippendorff's α: N/A ({result['alpha_note']})")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("csv_files", nargs="*", type=Path)
    ap.add_argument("--dir", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    files: List[Path] = list(args.csv_files)
    if args.dir:
        files.extend(
            p for p in sorted(args.dir.glob("*.csv"))
            if not p.stem.endswith(("_labeled", "_discrepancy"))
        )
    if not files:
        ap.error("Provide CSV file paths or --dir.")

    out_dir = args.out_dir or files[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for csv_p in files:
        if not csv_p.exists():
            print(f"[err] missing: {csv_p}", file=sys.stderr)
            continue
        try:
            res = analyze_file(csv_p, out_dir)
            results.append(res)
            print_discrepancy(res)
        except Exception as e:
            print(f"[err] {csv_p.name}: {e}", file=sys.stderr)

    # Summary table of alphas
    print(f"\n{'=' * 70}\nSUMMARY: Krippendorff's α per file\n{'=' * 70}")
    for r in results:
        a = f"{r['alpha']:.4f}" if r["alpha"] is not None else f"N/A ({r['alpha_note']})"
        print(f"  {r['file']:30}  raters={r['n_annotators']}  turns={r['n_turns']}  α={a}")

    return 0


if __name__ == "__main__":
    sys.exit(main())