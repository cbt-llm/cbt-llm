"""
Compute centrifugal force F per turn. Five variants, same numerator, different r.

  numerator = m * shift^2

    F_A = numerator / (d(MCOT, baseline_centroid) + perturbation)
    F_B = numerator / (d(MCOT, baseline_centroid) + perturbation + 1)
    F_C = numerator / (perturbation + 1)
    F_D = numerator / (d(protocol_purpose, baseline_centroid) + perturbation)
    F_E = numerator / d(protocol_purpose, baseline_centroid)

Where:
  m             = mean pairwise SBERT cosine distance from this protocol's
                  PURPOSE statement to the other two protocols' purposes
                  (fixed, per protocol)
  shift         = signed shift per turn
                  = cos_dist(MCOT, patient) - cos_dist(baseline, patient)
  baseline_centroid
                = mean of baseline-response embeddings within transcript
                  (one per dialogue), L2-renormalized
  perturbation  = mean pairwise cosine distance between baseline centroids
                  computed under different prompt-variant baselines
                  (one value per transcript)

Usage:
  python compute_force.py \
      --mcot_dir output/gemma \
      --baseline_dirs output/gemma_baseline output/gemma_baseline_v2 \
                      output/gemma_baseline_v3 \
      --output force_gemma.csv

Filename pairing:
  cbt_mcot_transcript_N.json  <->  baseline_transcript_N.json
"""

import argparse, csv, glob, json, os, re, sys
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer


# Purpose statements from cbt_protocols.json (safety_check excluded — MCOT
# routes among these three only).
PROTOCOL_PURPOSES = {
    "validation":  ("Establish accurate understanding of the patient's "
                    "experience and psychological safety before cognitive "
                    "intervention."),
    "socratic":    ("Facilitate insight through questioning to help patient "
                    "draw their own conclusions rather than providing answers "
                    "directly"),
    "alternative": ("Guide the patient toward a more balanced view of their "
                    "situation by gently broadening how they interpret their "
                    "experience"),
}


def extract_protocol(turn):
    return {
        "socratic_questioning":    "socratic",
        "validate_and_reflect":    "validation",
        "alternative_perspective": "alternative",
    }.get(turn.get("llm_response", {}).get("protocol_used"))


def cos_dist(a, b):
    return 1.0 - float(np.dot(a, b))


def compute_protocol_mass(encoder):
    """Returns (mass_by_protocol, embedding_by_protocol).

    Mass = mean cos_dist from this protocol's purpose embedding to the others.
    Embeddings are L2-normalized (unit vectors) — keep them for use in F_D.
    """
    labels = list(PROTOCOL_PURPOSES.keys())
    purposes = [PROTOCOL_PURPOSES[l] for l in labels]
    embs = encoder.encode(purposes, normalize_embeddings=True,
                          convert_to_numpy=True, show_progress_bar=False)
    mass = {}
    embedding_by_protocol = {}
    for i, label in enumerate(labels):
        others = [embs[j] for j in range(len(labels)) if j != i]
        mass[label] = float(np.mean([cos_dist(embs[i], e) for e in others]))
        embedding_by_protocol[label] = embs[i]
    return mass, embedding_by_protocol


def baseline_filename(mcot_name):
    m = re.search(r"(\d+)\.json$", mcot_name)
    return f"baseline_transcript_{m.group(1)}.json" if m else f"baseline_{mcot_name}"


def load_baseline_texts(path):
    with open(path) as f:
        data = json.load(f)
    return {b["turn"]: b["baseline_response"].strip()
            for b in data.get("baseline", [])}


def centroid(vectors):
    """Mean of row-embeddings, L2-renormalized to live on the unit sphere."""
    c = np.mean(vectors, axis=0)
    n = np.linalg.norm(c)
    return c / n if n > 0 else c


def compute_transcript_force(mcot_path, baseline_paths, encoder,
                             mass_by_protocol, embedding_by_protocol):
    """Per-turn F for one transcript across N baseline prompt variants.

    Four variants computed per turn:
      F_A = m * shift^2 / (d(MCOT, centroid) + perturbation)
      F_B = m * shift^2 / (d(MCOT, centroid) + perturbation + 1)
      F_C = m * shift^2 / (perturbation + 1)
      F_D = m * shift^2 / (d(protocol_purpose, centroid) + perturbation)
    """
    with open(mcot_path) as f:
        mcot_data = json.load(f)

    baseline_variants = []
    for bp in baseline_paths:
        if not os.path.exists(bp):
            return None, f"missing baseline variant: {bp}"
        baseline_variants.append(load_baseline_texts(bp))

    patient_queries, mcot_responses, protocols, turn_ids = [], [], [], []
    baseline_variant_responses = [[] for _ in baseline_variants]

    for turn in sorted(mcot_data.get("transcript", []), key=lambda t: t.get("turn", 0)):
        idx = turn.get("turn")
        if any(idx not in bv for bv in baseline_variants):
            continue
        p_text = turn.get("patient", {}).get("query", "").strip()
        m_text = turn.get("llm_response", {}).get("response", "").strip()
        b_texts = [bv[idx] for bv in baseline_variants]
        if not (p_text and m_text and all(b_texts)):
            continue
        patient_queries.append(p_text)
        mcot_responses.append(m_text)
        for k, bt in enumerate(b_texts):
            baseline_variant_responses[k].append(bt)
        protocols.append(extract_protocol(turn))
        turn_ids.append(idx)

    if not mcot_responses:
        return None, "no aligned turns"

    p_emb = encoder.encode(patient_queries, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
    m_emb = encoder.encode(mcot_responses, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
    b_embs = [
        encoder.encode(bv, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False)
        for bv in baseline_variant_responses
    ]

    primary_b_emb = b_embs[0]

    centroids = [centroid(bv) for bv in b_embs]

    if len(centroids) >= 2:
        pair_dists = [cos_dist(centroids[i], centroids[j])
                      for i in range(len(centroids))
                      for j in range(i + 1, len(centroids))]
        perturbation = float(np.mean(pair_dists))
    else:
        perturbation = 0.0

    primary_centroid = centroids[0]

    per_turn = []
    for i in range(len(mcot_responses)):
        shift_i = cos_dist(m_emb[i], p_emb[i]) - cos_dist(primary_b_emb[i], p_emb[i])
        d_mcot_centroid = cos_dist(m_emb[i], primary_centroid)
        m_mass = mass_by_protocol.get(protocols[i], 1.0) if protocols[i] else 1.0

        # F_D needs distance from protocol's purpose embedding to the baseline centroid
        if protocols[i] and protocols[i] in embedding_by_protocol:
            d_protocol_centroid = cos_dist(embedding_by_protocol[protocols[i]],
                                           primary_centroid)
        else:
            d_protocol_centroid = None

        numerator = m_mass * (shift_i ** 2)

        denom_A = d_mcot_centroid + perturbation
        denom_B = denom_A + 1.0
        denom_C = perturbation + 1.0
        denom_D = (d_protocol_centroid + perturbation
                   if d_protocol_centroid is not None else None)
        denom_E = d_protocol_centroid if d_protocol_centroid is not None else None

        F_A = numerator / denom_A if denom_A > 0 else None
        F_B = numerator / denom_B
        F_C = numerator / denom_C
        F_D = numerator / denom_D if denom_D and denom_D > 0 else None
        F_E = numerator / denom_E if denom_E and denom_E > 0 else None

        per_turn.append({
            "turn": turn_ids[i],
            "protocol": protocols[i],
            "shift": shift_i,
            "d_mcot_centroid": d_mcot_centroid,
            "d_protocol_centroid": d_protocol_centroid,
            "mass": m_mass,
            "F_A": F_A,
            "F_B": F_B,
            "F_C": F_C,
            "F_D": F_D,
            "F_E": F_E,
        })

    return {
        "turns": per_turn,
        "perturbation": perturbation,
        "N": len(per_turn),
    }, None


def main():
    ap = argparse.ArgumentParser(description="Compute centrifugal F per transcript")
    ap.add_argument("--mcot_dir", required=True)
    ap.add_argument("--baseline_dirs", required=True, nargs="+",
                    help="One or more baseline dirs. First = primary. "
                         "Perturbation computed across all given.")
    ap.add_argument("--output", default="force_results.csv")
    ap.add_argument("--sbert_model", default="all-mpnet-base-v2")
    args = ap.parse_args()

    print(f"Loading SBERT: {args.sbert_model}")
    encoder = SentenceTransformer(args.sbert_model)

    print("\nComputing protocol mass from purpose statements...")
    mass_by_protocol, embedding_by_protocol = compute_protocol_mass(encoder)
    for p, m in mass_by_protocol.items():
        print(f"  m_{p:12s} = {m:.4f}")

    mcot_files = sorted(glob.glob(os.path.join(args.mcot_dir, "*.json")))
    if not mcot_files:
        sys.exit(f"No MCOT transcripts in {args.mcot_dir}")

    print(f"\nProcessing {len(mcot_files)} transcripts "
          f"with {len(args.baseline_dirs)} baseline variants\n")

    F_KEYS = ("F_A", "F_B", "F_C", "F_D", "F_E")

    all_rows = []
    per_protocol = defaultdict(lambda: {k: [] for k in F_KEYS})

    for mp in mcot_files:
        fname = os.path.basename(mp)
        bfname = baseline_filename(fname)
        baseline_paths = [os.path.join(d, bfname) for d in args.baseline_dirs]

        result, err = compute_transcript_force(
            mp, baseline_paths, encoder, mass_by_protocol, embedding_by_protocol
        )
        if err:
            print(f"  skip {fname}: {err}")
            continue

        row = {
            "file": fname,
            "N": result["N"],
            "perturbation": result["perturbation"],
        }
        for fk in F_KEYS:
            vals = [t[fk] for t in result["turns"] if t[fk] is not None]
            row[f"mean_{fk}"] = float(np.mean(vals)) if vals else None

        by_p = defaultdict(lambda: {k: [] for k in F_KEYS})
        for t in result["turns"]:
            if t["protocol"]:
                for fk in F_KEYS:
                    if t[fk] is not None:
                        by_p[t["protocol"]][fk].append(t[fk])
                        per_protocol[t["protocol"]][fk].append(t[fk])

        for p in ("socratic", "validation", "alternative"):
            pdata = by_p.get(p, {k: [] for k in F_KEYS})
            for fk in F_KEYS:
                row[f"mean_{fk}_{p}"] = (float(np.mean(pdata[fk]))
                                         if pdata[fk] else None)
            row[f"n_{p}"] = len(pdata["F_B"])
        all_rows.append(row)

        print(f"  {fname:40s}  N={result['N']:3d}  "
              f"perturbation={result['perturbation']:.4f}  "
              f"F_A={row['mean_F_A']:.5f}  F_B={row['mean_F_B']:.5f}  "
              f"F_C={row['mean_F_C']:.5f}  F_D={row['mean_F_D']:.5f}  "
              f"F_E={row['mean_F_E']:.5f}")

    fieldnames = ["file", "N", "perturbation",
                  "mean_F_A", "mean_F_B", "mean_F_C", "mean_F_D"]
    for p in ("socratic", "validation", "alternative"):
        for fk in F_KEYS:
            fieldnames.append(f"mean_{fk}_{p}")
        fieldnames.append(f"n_{p}")

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved {args.output}")

    print("\n=== Corpus-level F ===")
    for fk in F_KEYS:
        vals = [r[f"mean_{fk}"] for r in all_rows if r[f"mean_{fk}"] is not None]
        if vals:
            print(f"  {fk}: mean={np.mean(vals):.5f}  "
                  f"std={np.std(vals):.5f}  n={len(vals)}")

    print("\n=== Per-protocol F (turn-level pooled) ===")
    for p in ("socratic", "validation", "alternative"):
        for fk in F_KEYS:
            vs = per_protocol[p][fk]
            if vs:
                print(f"  {p:12s}  {fk} mean={np.mean(vs):.5f}  "
                      f"std={np.std(vs):.5f}  turns={len(vs)}")


if __name__ == "__main__":
    main()