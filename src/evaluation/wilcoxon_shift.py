"""
Wilcoxon signed-rank test on per-turn signed shifts.

Reads MCOT + baseline transcripts, computes per-turn signed shift, then tests
whether the median shift differs from 0 per (model, protocol) group.

The test answers: "is MCOT systematically different from baseline?"
  - Reject H0 (p < 0.05) -> shift != 0 -> MCOT redirects
  - Fail to reject      -> shift ~ 0  -> MCOT acts like baseline (entrains)

Usage:
  python wilcoxon_shift.py \\
      --input_spec Mistral:output/mistral:output/mistral_baseline \\
                   GPT:output/gpt:output/gpt_baseline \\
                   Gemma:output/gemma:output/gemma_baseline \\
      --output_per_turn shifts_per_turn.csv \\
      --output_wilcoxon wilcoxon_shift.csv

Each --input_spec entry is "ModelName:mcot_dir:baseline_dir".
Filename pairing: cbt_mcot_transcript_N.json <-> baseline_transcript_N.json
"""

import argparse, csv, glob, json, os, re, sys
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.stats import wilcoxon


def extract_protocol(turn):
    return {
        "socratic_questioning":    "socratic",
        "validate_and_reflect":    "validation",
        "alternative_perspective": "alternative",
    }.get(turn.get("llm_response", {}).get("protocol_used"))


def signed_shift(m_emb, b_emb, p_emb):
    """shift = cos_dist(MCOT, patient) - cos_dist(baseline, patient)."""
    d_mp = 1.0 - float(np.dot(m_emb, p_emb))
    d_bp = 1.0 - float(np.dot(b_emb, p_emb))
    return d_mp - d_bp


def baseline_filename(mcot_name):
    m = re.search(r"(\d+)\.json$", mcot_name)
    return f"baseline_transcript_{m.group(1)}.json" if m else f"baseline_{mcot_name}"


def load_baseline_texts(path):
    with open(path) as f:
        data = json.load(f)
    return {b["turn"]: b["baseline_response"].strip()
            for b in data.get("baseline", [])}


def per_turn_shifts_for_transcript(mcot_path, baseline_path, encoder):
    """Returns list of (turn_id, protocol, shift) tuples for one transcript."""
    with open(mcot_path) as f:
        mcot_data = json.load(f)
    baseline_by_turn = load_baseline_texts(baseline_path)

    patient_queries, mcot_responses, baseline_responses = [], [], []
    protocols, turn_ids = [], []

    for turn in sorted(mcot_data.get("transcript", []),
                       key=lambda t: t.get("turn", 0)):
        idx = turn.get("turn")
        if idx not in baseline_by_turn:
            continue
        p_text = turn.get("patient", {}).get("query", "").strip()
        m_text = turn.get("llm_response", {}).get("response", "").strip()
        b_text = baseline_by_turn[idx]
        if not (p_text and m_text and b_text):
            continue
        patient_queries.append(p_text)
        mcot_responses.append(m_text)
        baseline_responses.append(b_text)
        protocols.append(extract_protocol(turn))
        turn_ids.append(idx)

    if not mcot_responses:
        return []

    p_emb = encoder.encode(patient_queries, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
    m_emb = encoder.encode(mcot_responses, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
    b_emb = encoder.encode(baseline_responses, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)

    out = []
    for i in range(len(mcot_responses)):
        s = signed_shift(m_emb[i], b_emb[i], p_emb[i])
        out.append((turn_ids[i], protocols[i], s))
    return out


def collect_per_turn(model_name, mcot_dir, baseline_dir, encoder):
    """Yields per-turn (model, file, turn, protocol, shift) rows across a model's transcripts."""
    mcot_files = sorted(glob.glob(os.path.join(mcot_dir, "*.json")))
    if not mcot_files:
        print(f"  [warn] no transcripts in {mcot_dir}")
        return []
    all_rows = []
    for mp in mcot_files:
        fname = os.path.basename(mp)
        bpath = os.path.join(baseline_dir, baseline_filename(fname))
        if not os.path.exists(bpath):
            print(f"  skip {fname}: no baseline {os.path.basename(bpath)}")
            continue
        for turn_id, proto, shift in per_turn_shifts_for_transcript(mp, bpath, encoder):
            all_rows.append({
                "model": model_name,
                "file": fname,
                "turn": turn_id,
                "protocol": proto,
                "shift": shift,
            })
    return all_rows


def significance_stars(p):
    if p is None:
        return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def run_wilcoxon(per_turn_rows):
    """Groups per-turn shifts by (model, protocol) and runs Wilcoxon signed-rank against 0."""
    grouped = defaultdict(list)
    for r in per_turn_rows:
        if r["protocol"]:  # drop turns with no protocol label
            grouped[(r["model"], r["protocol"])].append(r["shift"])

    results = []
    for (model, protocol), shifts in grouped.items():
        shifts = np.array(shifts, dtype=float)
        n = len(shifts)
        median = float(np.median(shifts))
        mean   = float(np.mean(shifts))

        # Wilcoxon needs n >= 1 and not all zeros. Two-sided, H0: median = 0.
        # zero_method="wilcox" drops exact zeros before ranking (standard default).
        if n < 1:
            W, pvalue = None, None
        elif np.all(shifts == 0):
            W, pvalue = 0.0, 1.0
        else:
            try:
                res = wilcoxon(shifts, zero_method="wilcox", alternative="two-sided")
                W = float(res.statistic)
                pvalue = float(res.pvalue)
            except ValueError as e:
                # e.g. too few non-zero differences
                W, pvalue = None, None
                print(f"  [warn] Wilcoxon failed for ({model},{protocol}): {e}")

        results.append({
            "model": model,
            "protocol": protocol,
            "n": n,
            "median_shift": round(median, 5),
            "mean_shift":   round(mean,   5),
            "W":            round(W, 3) if W is not None else "",
            "p_value":      round(pvalue, 5) if pvalue is not None else "",
            "sig":          significance_stars(pvalue),
        })

    # Stable ordering
    proto_order = {"socratic": 0, "validation": 1, "alternative": 2}
    results.sort(key=lambda r: (r["model"], proto_order.get(r["protocol"], 99)))
    return results


def main():
    ap = argparse.ArgumentParser(description="Per-turn signed shifts + Wilcoxon.")
    ap.add_argument("--input_spec", nargs="+", required=True,
                    help='Entries of form "ModelName:mcot_dir:baseline_dir".')
    ap.add_argument("--output_per_turn", default="shifts_per_turn.csv")
    ap.add_argument("--output_wilcoxon", default="wilcoxon_shift.csv")
    ap.add_argument("--sbert_model", default="all-mpnet-base-v2")
    args = ap.parse_args()

    print(f"Loading SBERT: {args.sbert_model}")
    encoder = SentenceTransformer(args.sbert_model)

    all_per_turn = []
    for spec in args.input_spec:
        parts = spec.split(":")
        if len(parts) != 3:
            sys.exit(f"bad --input_spec entry (need ModelName:mcot_dir:baseline_dir): {spec}")
        model_name, mcot_dir, baseline_dir = parts
        print(f"\n[{model_name}] mcot={mcot_dir}  baseline={baseline_dir}")
        rows = collect_per_turn(model_name, mcot_dir, baseline_dir, encoder)
        print(f"  collected {len(rows)} per-turn shifts")
        all_per_turn.extend(rows)

    # Save per-turn CSV (one row per turn per model)
    with open(args.output_per_turn, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "file", "turn", "protocol", "shift"])
        w.writeheader()
        for r in all_per_turn:
            w.writerow({**r, "shift": round(r["shift"], 6)})
    print(f"\nSaved per-turn shifts: {args.output_per_turn}")

    # Wilcoxon
    wilcox_rows = run_wilcoxon(all_per_turn)

    with open(args.output_wilcoxon, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "model", "protocol", "n", "median_shift", "mean_shift",
            "W", "p_value", "sig"])
        w.writeheader()
        w.writerows(wilcox_rows)
    print(f"Saved Wilcoxon results: {args.output_wilcoxon}")

    print("\n=== Wilcoxon signed-rank on per-turn shifts (H0: median = 0) ===")
    hdr = f"{'model':10s}  {'protocol':12s}  {'n':>4s}  {'median':>9s}  {'mean':>9s}  {'W':>10s}  {'p':>9s}  sig"
    print(hdr)
    print("-" * len(hdr))
    for r in wilcox_rows:
        p_disp = f"{r['p_value']:.4g}" if r['p_value'] != "" else "-"
        W_disp = f"{r['W']:.2f}" if r['W'] != "" else "-"
        print(f"{r['model']:10s}  {r['protocol']:12s}  {r['n']:4d}  "
              f"{r['median_shift']:>9.4f}  {r['mean_shift']:>9.4f}  "
              f"{W_disp:>10s}  {p_disp:>9s}  {r['sig']}")


if __name__ == "__main__":
    main()