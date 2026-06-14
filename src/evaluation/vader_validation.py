import json
import re
import pandas as pd
from datasets import load_dataset
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

SEED_PATH = "data/processed/user_case_studies.json"

NRC_VAD_PATH = "external_libs/NRC-VAD-Lexicon-v2.1/NRC-VAD-Lexicon-v2.1.txt"
OUT_PATH = "data/processed/vader_intensity_table.csv"
START_ID = 76

TERM_COL, VAL_COL, ARO_COL, DOM_COL = 0, 1, 2, 3

TOKEN_RE = re.compile(r"[a-z][a-z'\-]*")


def build_esconv_flat():
    ds = load_dataset("thu-coai/esconv")
    flat = []
    for split in ["train", "validation", "test"]:
        if split in ds:
            for row in ds[split]:
                flat.append(json.loads(row["text"]))
    return flat


def get_initial_intensity(rec):
    try:
        return int(rec["survey_score"]["seeker"]["initial_emotion_intensity"])
    except (KeyError, TypeError, ValueError):
        return None


def first_usr_utterance(rec):
    for turn in rec.get("dialog", []):
        if turn.get("speaker") == "usr":
            return turn.get("text")
    return None


def load_vad_lexicon(path):
    lex = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(TERM_COL, VAL_COL, ARO_COL, DOM_COL):
                continue
            try:
                v = float(parts[VAL_COL]); a = float(parts[ARO_COL]); float(parts[DOM_COL])
            except ValueError:
                continue
            lex[parts[TERM_COL].lower()] = (v, a)
    return lex


def score_vad(text, lex):
    """Mean valence/arousal over in-lexicon tokens."""
    toks = TOKEN_RE.findall(text.lower())
    hits = [lex[t] for t in toks if t in lex]
    if not hits:
        return (None, None)
    v = sum(h[0] for h in hits) / len(hits)
    a = sum(h[1] for h in hits) / len(hits)
    return (v, a)


def main():
    with open(SEED_PATH, "r", encoding="utf-8") as f:
        core = json.load(f)
    core = [c for c in core if int(c["id"]) >= START_ID]

    esconv_rows = [c for c in core if c["source"].startswith("esconv_")]
    skipped = [c["source"] for c in core if not c["source"].startswith("esconv_")]
    print(f"Total rows id>={START_ID}: {len(core)} | esconv: {len(esconv_rows)} | skipped non-esconv: {len(skipped)}")
    if skipped:
        print("  skipped sources (first 10):", skipped[:10])

    esconv = build_esconv_flat()
    sia = SentimentIntensityAnalyzer()
    vad = load_vad_lexicon(NRC_VAD_PATH)

    rows = []
    for item in esconv_rows:
        source = item["source"]
        seed = item["user_case_seed_query"]
        idx = int(source.split("_")[1])
        rec = esconv[idx] if 0 <= idx < len(esconv) else None

        p = sia.polarity_scores(seed)
        vint = -p["compound"]  # VADER distress proxy, [-1, 1]
        v, a = score_vad(seed, vad)

        rows.append({
            "esconv_id": source,
            "user_case_seed_query": seed,
            "esconv_situation": rec.get("situation") if rec else None,
            "esconv_first_user_utterance": first_usr_utterance(rec) if rec else None,
            "esconv_initial_emotion_intensity": get_initial_intensity(rec) if rec else None,
            "vader_intensity": round(vint, 4),
            "vad_arousal": round(a, 4) if a is not None else None,
            "vad_valence": round(v, 4) if v is not None else None,
        })

    df = pd.DataFrame(rows)

    # Normalize each variable separately (per advisor): min-max -> [0,1], z-score.
    def minmax(s):
        return (s - s.min()) / (s.max() - s.min())

    def zscore(s):
        return (s - s.mean()) / s.std()

    for col in ["vader_intensity", "vad_arousal", "esconv_initial_emotion_intensity"]:
        df[col + "_minmax"] = minmax(df[col]).round(4)
        df[col + "_zscore"] = zscore(df[col]).round(4)

    e_mm, e_z = "esconv_initial_emotion_intensity_minmax", "esconv_initial_emotion_intensity_zscore"
    df["vader_minmax_abs_diff"] = (df["vader_intensity_minmax"] - df[e_mm]).abs().round(4)
    df["vader_zscore_abs_diff"] = (df["vader_intensity_zscore"] - df[e_z]).abs().round(4)
    df["arousal_minmax_abs_diff"] = (df["vad_arousal_minmax"] - df[e_mm]).abs().round(4)
    df["arousal_zscore_abs_diff"] = (df["vad_arousal_zscore"] - df[e_z]).abs().round(4)

    # correlations: each proxy vs self-reported intensity
    def report(label, col):
        sub = df.dropna(subset=[col, "esconv_initial_emotion_intensity"])
        if len(sub) < 2:
            print(f"{label}: insufficient data"); return
        x, y = sub[col], sub["esconv_initial_emotion_intensity"]
        print(f"{label} (n={len(sub)}): "
              f"Spearman={x.corr(y, method='spearman'):.3f} "
              f"Kendall={x.corr(y, method='kendall'):.3f} "
              f"Pearson={x.corr(y, method='pearson'):.3f}")

    print("\n=== correlation vs esconv_initial_emotion_intensity ===")
    report("VADER -compound ", "vader_intensity")
    report("NRC-VAD arousal ", "vad_arousal")
    print("VADER==0 rows:", int((df["vader_intensity"] == 0).sum()))

    out_cols = [
        "esconv_id", "user_case_seed_query", "esconv_situation", "esconv_first_user_utterance",
        "esconv_initial_emotion_intensity",
        "vader_intensity", "vad_arousal", "vad_valence",
        "vader_intensity_minmax", "vad_arousal_minmax", e_mm,
        "vader_intensity_zscore", "vad_arousal_zscore", e_z,
        "vader_minmax_abs_diff", "arousal_minmax_abs_diff",
        "vader_zscore_abs_diff", "arousal_zscore_abs_diff",
    ]
    df[out_cols].to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(df)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()