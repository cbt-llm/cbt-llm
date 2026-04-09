"""
compute_nclid.py
────────────────────────────────────────────────────────────────────────────
Drop-in script to compute nCLiD (normalized Conversational Linguistic
Distance) per Nasir et al. (2019), as used in Kian et al. NAACL 2025.

Usage
-----
    python compute_nclid.py --input_dir output/gpt --output results_gpt.csv
    python compute_nclid.py --input_dir output/gemma --output results_gemma.csv

    # or point at a single file:
    python compute_nclid.py --input_file output/gpt/cbt_mcot_transcript_1.json

Transcript format expected
--------------------------
Your JSON files match the structure in the paper's dataset:
  {
    "metadata": { "llm_response": ..., ... },
    "transcript": [
      {
        "turn": 0,
        "patient":      { "query": "<patient utterance>" },
        "llm_response": { "response": "<LLM/therapist utterance>" }
      },
      ...
    ]
  }

nCLiD definition (Eq. 1-2 + Appendix C in the paper)
------------------------------------------------------
  anchor      A = LLM/therapist turns  [t1, t2, ..., tN]
  coordinator C = patient turns        [p1, p2, ..., pN]

  Local distance (context window k):
      d_i^{C→A} = min_{i ≤ j ≤ i+k-1} WMD(a_i, c_j)

  uCLiD = (1/N) * Σ d_i^{C→A}

  α = (2 / N(N-1)) * [
        Σ_{i<j} WMD(a_i, a_j)    # within-anchor
      + Σ_{i<j} WMD(c_i, c_j)    # within-coordinator
      + Σ_{i≤j} WMD(a_i, c_j)    # cross anchor→coordinator
      ]

  nCLiD = uCLiD / α

Note: stop words are NOT removed (following Nasir et al. 2019).
      Whitespace tokenization only.

Dependencies
------------
    pip install gensim numpy
    # word2vec model downloaded automatically on first run (~1.5 GB)
"""

import argparse
import glob
import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

# ── gensim imports ────────────────────────────────────────────────────────────
try:
    import gensim.downloader as api
    from gensim.models import KeyedVectors
except ImportError:
    sys.exit("gensim not found. Run: pip install gensim")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Word Mover's Distance helpers
# ══════════════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> list[str]:
    """Whitespace split; no stop-word removal (per Nasir et al. 2019)."""
    return text.lower().split()


def wmd(model: KeyedVectors, text_a: str, text_b: str) -> float:
    """
    Word Mover's Distance between two strings using gensim's WMD.
    Returns a large fallback value if either string has no in-vocabulary tokens.
    """
    tokens_a = [t for t in _tokenize(text_a) if t in model.key_to_index]
    tokens_b = [t for t in _tokenize(text_b) if t in model.key_to_index]
    if not tokens_a or not tokens_b:
        return 1.0          # fallback: maximum distance
    return model.wmdistance(tokens_a, tokens_b)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Core nCLiD computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_nclid(
    anchor_turns: list[str],
    coord_turns: list[str],
    model: KeyedVectors,
    k: int = 2,
) -> dict:
    """
    Compute nCLiD for one conversation.

    Parameters
    ----------
    anchor_turns : list of str
        Therapist / LLM utterances  [t1 .. tN]
    coord_turns  : list of str
        Patient utterances           [p1 .. pN]
    model        : gensim KeyedVectors (word2vec)
    k            : context window size (default 2, paper uses unspecified k;
                   2 is the most common convention in follow-up work)

    Returns
    -------
    dict with keys: uCLiD, alpha, nCLiD, N, local_distances
    """
    N = min(len(anchor_turns), len(coord_turns))
    if N < 2:
        return {"uCLiD": None, "alpha": None, "nCLiD": None, "N": N,
                "local_distances": []}

    # ── Step 1: local distances d_i (Eq. 1) ──────────────────────────────────
    local_d = []
    for i in range(N):
        # look ahead up to k coordinator turns starting from position i
        j_max = min(i + k, N)          # exclusive upper bound
        candidates = [wmd(model, anchor_turns[i], coord_turns[j])
                      for j in range(i, j_max)]
        local_d.append(min(candidates))

    uCLiD = float(np.mean(local_d))

    # ── Step 2: normalization factor α (Appendix C, Eq. 4) ───────────────────
    denom = N * (N - 1)                 # used as 2 / N(N-1) * Σ per term

    # within-anchor: all pairs (a_i, a_j) with i < j
    within_a = sum(
        wmd(model, anchor_turns[i], anchor_turns[j])
        for i, j in combinations(range(N), 2)
    )

    # within-coordinator: all pairs (c_i, c_j) with i < j
    within_c = sum(
        wmd(model, coord_turns[i], coord_turns[j])
        for i, j in combinations(range(N), 2)
    )

    # cross anchor→coordinator: all pairs (a_i, c_j) with i ≤ j
    # Appendix C uses i ≤ j (not strict i < j) for the cross term
    cross_ac = sum(
        wmd(model, anchor_turns[i], coord_turns[j])
        for i in range(N)
        for j in range(i, N)
    )

    if denom == 0:
        alpha = 1.0
    else:
        alpha = (2.0 / denom) * (within_a + within_c + cross_ac)

    nCLiD = uCLiD / alpha if alpha > 0 else float("nan")

    return {
        "uCLiD": uCLiD,
        "alpha": alpha,
        "nCLiD": nCLiD,
        "N": N,
        "local_distances": local_d,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Transcript parsing
# ══════════════════════════════════════════════════════════════════════════════

def parse_transcript(path: str) -> tuple[list[str], list[str], dict]:
    """
    Parse one of your cbt_mcot_transcript_*.json files.

    Returns (anchor_turns, coord_turns, metadata)
    anchor = llm_response.response  (therapist side)
    coord  = patient.query          (patient side)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    turns = data.get("transcript", [])

    anchor_turns = []
    coord_turns  = []

    for turn in sorted(turns, key=lambda t: t["turn"]):
        patient_text = turn.get("patient", {}).get("query", "").strip()
        llm_text     = turn.get("llm_response", {}).get("response", "").strip()

        if patient_text:
            coord_turns.append(patient_text)
        if llm_text:
            anchor_turns.append(llm_text)

    return anchor_turns, coord_turns, metadata


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def load_word2vec() -> KeyedVectors:
    """Download / load the 300-d Google News word2vec model via gensim."""
    print("Loading word2vec (Google News 300d) — downloads ~1.5 GB on first run …")
    model = api.load("word2vec-google-news-300")
    print("Model loaded.")
    return model


def process_file(path: str, model: KeyedVectors, k: int) -> dict:
    anchor, coord, meta = parse_transcript(path)
    result = compute_nclid(anchor, coord, model, k=k)
    result["file"]  = os.path.basename(path)
    result["model"] = meta.get("llm_response", "unknown")
    result["mode"]  = meta.get("mode", "unknown")
    result["turns"] = meta.get("turns", len(anchor))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compute nCLiD for CBT-MCOT transcripts (Nasir et al. 2019)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_dir",  help="Directory of JSON transcript files")
    group.add_argument("--input_file", help="Single JSON transcript file")

    parser.add_argument(
        "--output", default="nclid_results.csv",
        help="Output CSV path (default: nclid_results.csv)"
    )
    parser.add_argument(
        "--k", type=int, default=2,
        help="Context window size for local distance (default: 2)"
    )
    parser.add_argument(
        "--model_path", default=None,
        help="Path to a local .bin word2vec file (skips gensim download)"
    )
    args = parser.parse_args()

    # ── Load word2vec ──────────────────────────────────────────────────────────
    if args.model_path:
        print(f"Loading word2vec from {args.model_path} …")
        model = KeyedVectors.load_word2vec_format(args.model_path, binary=True)
    else:
        model = load_word2vec()

    # ── Gather files ───────────────────────────────────────────────────────────
    if args.input_file:
        files = [args.input_file]
    else:
        files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
        if not files:
            sys.exit(f"No JSON files found in {args.input_dir}")

    print(f"Processing {len(files)} file(s) with k={args.k} …\n")

    # ── Process & collect ─────────────────────────────────────────────────────
    rows = []
    for fpath in files:
        try:
            res = process_file(fpath, model, k=args.k)
            rows.append(res)
            print(
                f"  {res['file']:40s}  "
                f"N={res['N']:3d}  "
                f"uCLiD={res['uCLiD']:.4f}  "
                f"α={res['alpha']:.4f}  "
                f"nCLiD={res['nCLiD']:.4f}"
            )
        except Exception as e:
            print(f"  ERROR processing {fpath}: {e}")

    # ── Write CSV ─────────────────────────────────────────────────────────────
    import csv
    cols = ["file", "model", "mode", "turns", "N", "uCLiD", "alpha", "nCLiD"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {args.output}")

    # ── Quick summary stats ───────────────────────────────────────────────────
    valid = [r["nCLiD"] for r in rows if r["nCLiD"] is not None]
    if valid:
        print(f"\nSummary across {len(valid)} transcripts:")
        print(f"  mean nCLiD = {np.mean(valid):.4f}")
        print(f"  std  nCLiD = {np.std(valid):.4f}")
        print(f"  min  nCLiD = {np.min(valid):.4f}")
        print(f"  max  nCLiD = {np.max(valid):.4f}")


if __name__ == "__main__":
    main()