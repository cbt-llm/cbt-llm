import json, random, re
from datasets import load_dataset
from cbt_llm.utils import paths

PROCESSED = paths.get_project_root() / "data/processed"
REALCBT = PROCESSED / "realcbt.json"
OUT = PROCESSED / "user_case_studies.json"
SEED = 42

N_ESCONV = 76
N_SWMH = 48

ESCONV_REV = None
SWMH_REV = None
SWMH_MAX_CHARS = 700
SWMH_MIN_CHARS = 50
SWMH_KEEP = {"depression", "suicidewatch"}
SWMH_LABELS = {"depression": "Safety: Depression", "suicidewatch": "Safety: Suicidal thoughts"}
DELETED_RE = re.compile(r"\[\s*(deleted|removed)", re.I)
EDIT_RE = re.compile(r"\n[\s\-\u2013\u2014_*>]*\b(?:edit|update)\b\s*\d*\s*[:\-\u2013\u2014].*", re.I | re.S)

TEXT_COLS = ["text", "body", "selftext", "post", "content", "clean_text", "title_text"]
LABEL_COLS = ["label", "labels", "class", "target", "category", "subreddit"]
ID_COLS = ["id", "index", "idx", "post_id"]

clean = lambda s: re.sub(r"\s+", " ", str(s)).strip()


def _pick(colnames, candidates, required=True):
    for c in candidates:
        if c in colnames:
            return c
    if required:
        raise KeyError(f"none of {candidates} found in columns {list(colnames)}")
    return None


def _norm_label(v):
    s = str(v).strip().lower()
    for p in ("self.", "/r/", "r/"):
        if s.startswith(p):
            s = s[len(p):]
    return s


# ESConv
def build_esconv():
    pool = []
    for idx, r in enumerate(load_dataset("thu-coai/esconv", revision=ESCONV_REV)["train"]):
        d = json.loads(r["text"]) if "text" in r else r
        if d.get("situation") and d.get("problem_type"):
            pool.append((idx, clean(d["situation"]), d["problem_type"].strip()))

    sample = random.Random(SEED).sample(pool, N_ESCONV)
    return [{"id": str(i + 1), "source": f"esconv_{idx}",
             "user_case_seed_query": s, "core_issue": c}
            for i, (idx, s, c) in enumerate(sample)]


# SWMH (depression / SuicideWatch)
def build_swmh():
    ds = load_dataset("AIMH/SWMH", revision=SWMH_REV)["train"]
    text_col = _pick(ds.column_names, TEXT_COLS)
    label_col = _pick(ds.column_names, LABEL_COLS)
    id_col = _pick(ds.column_names, ID_COLS, required=False)

    feat = ds.features[label_col]
    int2str = getattr(feat, "int2str", None)

    def label_str(v):
        return int2str(v) if (int2str and isinstance(v, int)) else v

    pool, seen = [], set()
    for idx, r in enumerate(ds):
        lbl = _norm_label(label_str(r[label_col]))
        if lbl not in SWMH_KEEP:
            continue
        txt = clean(EDIT_RE.sub("", str(r[text_col])))
        if DELETED_RE.search(txt):
            continue
        if not (SWMH_MIN_CHARS <= len(txt) <= SWMH_MAX_CHARS):
            continue
        if txt in seen:
            continue
        seen.add(txt)
        rid = r[id_col] if id_col else idx
        pool.append((rid, txt, SWMH_LABELS[lbl]))

    k = min(N_SWMH, len(pool))
    if k < N_SWMH:
        print(f"WARNING: only {k} SWMH posts passed filters (wanted {N_SWMH})")
    sample = random.Random(SEED).sample(pool, k)
    return [{"id": str(i + 1), "source": f"swmh_{rid}",
             "user_case_seed_query": s, "core_issue": c}
            for i, (rid, s, c) in enumerate(sample)]

# RealCBT
def combine_with_realcbt(records):
    realcbt = json.loads(REALCBT.read_text())
    combined = realcbt + records
    for i, r in enumerate(combined, 1):
        r["id"] = str(i)
    return combined


def main():
    records = build_esconv() + build_swmh()
    combined = combine_with_realcbt(records)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(combined, ensure_ascii=False, indent=2))

    from collections import Counter
    print(f"wrote {len(combined)} records -> {OUT}")
    print("ids:", combined[0]["id"], "..", combined[-1]["id"])
    print("by source prefix:", Counter(r["source"].split("_")[0] for r in combined))
    print("by core_issue:", Counter(r["core_issue"] for r in combined))


if __name__ == "__main__":
    main()