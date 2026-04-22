import argparse, csv, glob, json, os, sys
from collections import defaultdict
from itertools import combinations
import re

import numpy as np
from sentence_transformers import SentenceTransformer


def extract_protocol(turn):
    return {
        "socratic_questioning":    "socratic",
        "validate_and_reflect":    "validation",
        "alternative_perspective": "alternative",
    }.get(turn.get("llm_response", {}).get("protocol_used"))


def compute_nclid(therapist_turns, user_turns, encoder,
                  context_window=2, protocols=None):
    n = len(therapist_turns)

    # Encode once, L2-normalize so cosine distance = 1 - dot(a, b)
    t_emb = encoder.encode(therapist_turns[:n], normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
    u_emb = encoder.encode(user_turns[:n], normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)

    def cos_dist(a, b):
        return 1.0 - float(np.dot(a, b))

    # Step 1: local distances
    local_distances = [
        min(cos_dist(t_emb[i], u_emb[j])
            for j in range(i, min(i + context_window, n)))
        for i in range(n)
    ]
    uCLiD = float(np.mean(local_distances))

    # Step 2: alpha
    pair_sum = 0.0
    for i, j in combinations(range(n), 2):
        pair_sum += cos_dist(t_emb[i], t_emb[j]) + cos_dist(u_emb[i], u_emb[j])
    for i in range(n):
        for j in range(i, n):
            pair_sum += cos_dist(t_emb[i], u_emb[j])
    alpha = (2.0 / (n * (n - 1))) * pair_sum

    nCLiD = uCLiD / alpha

    # Step 3: per-protocol grouping, normalized by the same alpha.
    per_protocol = {}
    if protocols:
        grouped = defaultdict(list)
        for p, d in zip(protocols[:n], local_distances):
            if p:
                grouped[p].append(d)
        per_protocol = {
            p: {"n": len(ds), "nCLiD_local": float(np.mean(ds) / alpha)}
            for p, ds in grouped.items()
        }

    return {"N": n, "uCLiD": uCLiD, "alpha": alpha, "nCLiD": nCLiD,
            "per_protocol": per_protocol}


def compute_shift(mcot_path, baseline_path, encoder):
    """Signed shift at each turn: how much MCOT moved the response relative
    to the patient query, compared to where baseline would have landed.
 
        shift_i = dist(MCOT_i, patient_i) - dist(baseline_i, patient_i)
 
    Positive: MCOT landed further from patient than baseline would have.
    Negative: MCOT landed closer to patient than baseline would have.
    Zero:     MCOT made no difference in patient-closeness.
    """
    with open(mcot_path) as f:
        mcot_data = json.load(f)
    with open(baseline_path) as f:
        baseline_data = json.load(f)
 
    baseline_by_turn = {b["turn"]: b["baseline_response"].strip()
                        for b in baseline_data.get("baseline", [])}
 
    patient_queries, mcot_responses, baseline_responses, protocols = [], [], [], []
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
 
    if not mcot_responses:
        return {"N": 0, "mean_shift": None, "per_protocol": {}}
 
    p_emb = encoder.encode(patient_queries, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
    m_emb = encoder.encode(mcot_responses, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
    b_emb = encoder.encode(baseline_responses, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
 
    # Signed shift: positive = MCOT moved response away from patient vs baseline
    shifts = [
        (1.0 - float(np.dot(m_emb[i], p_emb[i])))
        - (1.0 - float(np.dot(b_emb[i], p_emb[i])))
        for i in range(len(mcot_responses))
    ]
 
    grouped = defaultdict(list)
    for p, s in zip(protocols, shifts):
        if p:
            grouped[p].append(s)
 
    per_protocol = {
        p: {"n": len(ss), "mean_shift": float(np.mean(ss))}
        for p, ss in grouped.items()
    }
 
    return {
        "N": len(shifts),
        "mean_shift": float(np.mean(shifts)),
        "per_protocol": per_protocol,
    }
 
 
def process_transcript(path, encoder, k):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
 
    meta = data.get("metadata", {})
    therapist, user, protocols = [], [], []
 
    for turn in sorted(data.get("transcript", []), key=lambda t: t.get("turn", 0)):
        u_text = turn.get("patient", {}).get("query", "").strip()
        t_text = turn.get("llm_response", {}).get("response", "").strip()
        if u_text and t_text:
            user.append(u_text)
            therapist.append(t_text)
            protocols.append(extract_protocol(turn))
 
    result = compute_nclid(therapist, user, encoder, k, protocols)
    result["file"] = os.path.basename(path)
    result["model"] = meta.get("llm_response", "unknown")
    result["mode"] = meta.get("mode", "unknown")
    return result
 
 
def build_csv_rows(results, protocol_columns):
    for r in results:
        row = {k: r.get(k) for k in ("file", "model", "mode", "N",
                                     "uCLiD", "alpha", "nCLiD")}
        pp = r.get("per_protocol") or {}
        for p in protocol_columns:
            stats = pp.get(p, {})
            row[f"nCLiD_local_{p}"] = stats.get("nCLiD_local")
            row[f"n_{p}"] = stats.get("n", 0)
        yield row
 
 
def run_nclid(args, encoder):
    files = ([args.input_file] if args.input_file
             else sorted(glob.glob(os.path.join(args.input_dir, "*.json"))))
    if not files:
        sys.exit(f"No JSON files found in {args.input_dir}")
 
    print(f"Processing {len(files)} file(s) with k={args.k}\n")
    results = []
    for path in files:
        r = process_transcript(path, encoder, args.k)
        results.append(r)
        pp = r.get("per_protocol") or {}
        summary = ", ".join(f"{p}={s['nCLiD_local']:.3f} (n={s['n']})"
                            for p, s in pp.items()) or "no protocol labels"
        print(f"  {r['file']:40s}  N={r['N']:3d}  "
              f"nCLiD={r['nCLiD']:.4f}  | {summary}")
 
    core = ("socratic", "validation", "alternative")
    observed = {p for r in results for p in (r.get("per_protocol") or {})}
    protocol_columns = list(core) + sorted(observed - set(core))
 
    fieldnames = (["file", "model", "mode", "N", "uCLiD", "alpha", "nCLiD"]
                  + [c for p in protocol_columns
                       for c in (f"nCLiD_local_{p}", f"n_{p}")])
 
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(build_csv_rows(results, protocol_columns))
 
    print(f"\nSaved {args.output}")
 
    nclids = [r["nCLiD"] for r in results]
    print(f"\nDialogue-level nCLiD: mean={np.mean(nclids):.4f}  "
          f"std={np.std(nclids):.4f}  n={len(nclids)}")
    for p in protocol_columns:
        vals = [r["per_protocol"][p]["nCLiD_local"]
                for r in results if p in (r.get("per_protocol") or {})]
        if vals:
            print(f"  {p:12s} mean={np.mean(vals):.4f}  "
                  f"std={np.std(vals):.4f}  transcripts={len(vals)}")
 
 
def run_shift(args, encoder):
    mcot_files = sorted(glob.glob(os.path.join(args.mcot_dir, "*.json")))
    if not mcot_files:
        sys.exit(f"No transcripts in {args.mcot_dir}")
 
    print(f"Processing {len(mcot_files)} file(s)\n")
    results = []
    for mcot_path in mcot_files:
        fname = os.path.basename(mcot_path)
        # cbt_mcot_transcript_N.json -> baseline_transcript_N.json
        match = re.search(r"(\d+)\.json$", fname)
        baseline_name = (f"baseline_transcript_{match.group(1)}.json"
                         if match else f"baseline_{fname}")
        baseline_path = os.path.join(args.baseline_dir, baseline_name)
        if not os.path.exists(baseline_path):
            print(f"  skip (no baseline: {baseline_name}): {fname}")
            continue
        r = compute_shift(mcot_path, baseline_path, encoder)
        r["file"] = fname
        results.append(r)
        pp = r.get("per_protocol") or {}
        summary = ", ".join(f"{p}={s['mean_shift']:.3f} (n={s['n']})"
                            for p, s in pp.items()) or "no protocol labels"
        print(f"  {fname:40s}  N={r['N']:3d}  "
              f"shift={r['mean_shift']:.4f}  | {summary}")
 
    shifts = [r["mean_shift"] for r in results if r["mean_shift"] is not None]
    print(f"\nDialogue-level shift: mean={np.mean(shifts):.4f}  "
          f"std={np.std(shifts):.4f}  n={len(shifts)}")
    for p in ("socratic", "validation", "alternative"):
        vals = [r["per_protocol"][p]["mean_shift"] for r in results
                if p in (r.get("per_protocol") or {})]
        if vals:
            print(f"  {p:12s} mean={np.mean(vals):.4f}  "
                  f"std={np.std(vals):.4f}  transcripts={len(vals)}")
 
 
def main():
    ap = argparse.ArgumentParser(description="Compute nCLiD or shift with sentence-BERT")
    sub = ap.add_subparsers(dest="command", required=True)
 
    ncl = sub.add_parser("nclid", help="Compute nCLiD on MCOT transcripts")
    src = ncl.add_mutually_exclusive_group(required=True)
    src.add_argument("--input_dir",  help="Directory of transcript JSONs")
    src.add_argument("--input_file", help="Single transcript JSON")
    ncl.add_argument("--output", default="nclid_results.csv")
    ncl.add_argument("--k", type=int, default=2, help="Context window (default 2)")
 
    sh = sub.add_parser("shift", help="Compute MCOT-vs-baseline shift")
    sh.add_argument("--mcot_dir", required=True, help="MCOT transcript dir")
    sh.add_argument("--baseline_dir", required=True, help="Baseline transcript dir")
 
    ap.add_argument("--sbert_model", default="all-mpnet-base-v2",
                    help="sentence-transformers model name (default all-mpnet-base-v2)")
    args = ap.parse_args()
 
    print(f"Loading sentence-BERT model: {args.sbert_model}")
    encoder = SentenceTransformer(args.sbert_model)
 
    if args.command == "nclid":
        run_nclid(args, encoder)
    else:
        run_shift(args, encoder)
 
 
if __name__ == "__main__":
    main()