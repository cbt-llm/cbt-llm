#!/usr/bin/env python3
"""
cbt_llm.evaluating_benchmark

Heuristic + optional LLM-judge evaluation for generated CBT multi-turn transcripts.

Supports transcript JSON shapes:
- {"transcript": [{"role": "...", "content": "..."}], ...}   (your project output)
- {"messages": [...]}, {"turns": [...]}, etc.
- or a raw list: [{"role": "...", "content": "..."}]

Folder mode:
- If you pass a directory, it will recursively scan *.json/*.jsonl and only evaluate
  files that actually contain a transcript-like message list.

Outputs (per file) into --out:
- <stem>.metrics.json   (aggregate metrics)
- <stem>.turns.csv      (per-therapist-turn rows)
- <stem>.judge.jsonl    (optional, if --judge_model is set)
And also:
- summary.csv           (across all evaluated files)

Run examples:
  # Evaluate all transcript outputs under root/output/
  python -m cbt_llm.evaluating_benchmark output --out output/eval

  # Evaluate one transcript file
  python -m cbt_llm.evaluating_benchmark output/rag_transcript.json --out output/eval

  # Heuristics + OpenAI LLM judge (optional)
  export OPENAI_API_KEY="..."
  python -m cbt_llm.evaluating_benchmark output --out output/eval --judge_model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# Message parsing
# -----------------------------

ROLE_MAP = {
    # human side
    "patient": "user",
    "user": "user",
    "client": "user",
    "human": "user",
    # therapist side
    "therapist": "assistant",
    "assistant": "assistant",
    "counselor": "assistant",
    "model": "assistant",
}

TRANSCRIPT_KEYS = (
    "transcript",  # your project output
    "messages",
    "turns",
    "dialogue",
    "conversation",
    "chat",
    "utterances",
)

def _get_text(msg: Dict[str, Any]) -> str:
    """Extract text from different message schemas."""
    c = msg.get("content")
    if isinstance(c, str):
        return c.strip()

    # sometimes {text: "..."}
    t = msg.get("text")
    if isinstance(t, str):
        return t.strip()

    # OpenAI "content parts" style: content=[{text: "..."}]
    if isinstance(c, list):
        parts = []
        for p in c:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
        return ("\n".join(parts)).strip()

    # fallback keys
    for k in ("utterance", "value", "message", "msg"):
        v = msg.get(k)
        if isinstance(v, str):
            return v.strip()

    return ""


def _normalize_role(role_raw: Any) -> str:
    if not isinstance(role_raw, str):
        return "other"
    r = role_raw.strip().lower()
    return ROLE_MAP.get(r, r or "other")


def _extract_messages(obj: Any) -> Optional[List[Dict[str, Any]]]:
    """Return message list if obj contains transcript/messages; else None."""
    if isinstance(obj, list):
        # raw list of messages
        if obj and isinstance(obj[0], dict) and ("role" in obj[0] or "speaker" in obj[0]):
            return [m for m in obj if isinstance(m, dict)]
        return None

    if isinstance(obj, dict):
        for k in TRANSCRIPT_KEYS:
            v = obj.get(k)
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return [m for m in v if isinstance(m, dict)]
    return None


def _is_probably_transcript(messages: List[Dict[str, Any]]) -> bool:
    """Light validation to avoid scoring retrieval logs / prompt traces."""
    n_role = 0
    n_content = 0
    for m in messages[:30]:
        if "role" in m or "speaker" in m or "author" in m:
            n_role += 1
        txt = _get_text(m)
        if txt:
            n_content += 1
    return (n_role >= 3) and (n_content >= 3)


def load_transcript_file(path: Path) -> Optional[List[Tuple[str, str]]]:
    """Return normalized [(role, content), ...] or None if not a transcript."""
    try:
        text = path.read_text(encoding="utf-8").strip()
    except Exception:
        return None

    # JSONL support
    obj: Any
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        obj = rows
    else:
        try:
            obj = json.loads(text)
        except Exception:
            return None

    msgs_raw = _extract_messages(obj)
    if not msgs_raw:
        return None
    if not _is_probably_transcript(msgs_raw):
        return None

    out: List[Tuple[str, str]] = []
    for m in msgs_raw:
        role_raw = m.get("role") or m.get("speaker") or m.get("author") or ""
        role = _normalize_role(role_raw)
        content = _get_text(m)
        if not content:
            continue
        out.append((role, content))
    if len(out) < 2:
        return None
    return out


def list_inputs(paths: List[str]) -> List[Path]:
    files: List[Path] = []
    for p in paths:
        pp = Path(os.path.expanduser(p)).resolve()
        if pp.is_file():
            files.append(pp)
        elif pp.is_dir():
            files.extend(sorted(pp.rglob("*.json")))
            files.extend(sorted(pp.rglob("*.jsonl")))
    # de-dupe
    seen = set()
    uniq = []
    for f in files:
        s = str(f)
        if s not in seen:
            uniq.append(f)
            seen.add(s)
    return uniq


# -----------------------------
# Turn pairing
# -----------------------------

@dataclass
class EvalTurn:
    therapist_turn_idx: int
    patient_turn_idx: int
    patient: str
    therapist: str
    context: str


def build_eval_turns(messages: List[Tuple[str, str]], context_window: int) -> List[EvalTurn]:
    """
    For each therapist (assistant) message, find nearest previous patient (user) message.
    Context is a small window of messages before the patient turn.
    """
    turns: List[EvalTurn] = []
    for i, (role, content) in enumerate(messages):
        if role != "assistant":
            continue

        # find previous user
        j = i - 1
        patient_msg = None
        patient_idx = None
        while j >= 0:
            if messages[j][0] == "user":
                patient_msg = messages[j][1]
                patient_idx = j
                break
            j -= 1

        if patient_msg is None or patient_idx is None:
            continue

        ctx_start = max(0, patient_idx - context_window)
        ctx_msgs = messages[ctx_start:patient_idx]
        context = "\n".join([f"{r}: {t}" for (r, t) in ctx_msgs]) if ctx_msgs else ""

        turns.append(
            EvalTurn(
                therapist_turn_idx=i,
                patient_turn_idx=patient_idx,
                patient=patient_msg,
                therapist=content,
                context=context,
            )
        )
    return turns


# -----------------------------
# Heuristic metrics
# -----------------------------

THERAPIST_LEAK_TERMS = re.compile(
    r"\b(snomed|neo4j|embedding|vector|retrieval|rag|cosine|similarity|icd|umls|gds\.)\b",
    re.IGNORECASE,
)
CODE_LIKE_RE = re.compile(r"\b\d{4,}\b|[A-Z]{2,}\d{2,}", re.IGNORECASE)

PATIENT_DRIFT_PHRASES = re.compile(
    r"\b(it sounds like|let's explore|what evidence|could there be|have you considered|reframe|cognitive distortion|behavioral experiment)\b",
    re.IGNORECASE,
)

def word_count(s: str) -> int:
    return len(re.findall(r"\w+", s))

def sentence_count(s: str) -> int:
    parts = [p.strip() for p in re.split(r"[.!?]+", s.strip()) if p.strip()]
    return len(parts)

def compute_heuristics(eval_turns: List[EvalTurn], raw_messages: List[Tuple[str, str]]) -> Dict[str, Any]:
    therapist_texts = [t.therapist for t in eval_turns]
    patient_texts = [t.patient for t in eval_turns]

    therapist_wc = [word_count(x) for x in therapist_texts]
    patient_wc = [word_count(x) for x in patient_texts]

    therapist_q1 = [x.count("?") == 1 for x in therapist_texts]
    therapist_leak = [
        bool(THERAPIST_LEAK_TERMS.search(x) or CODE_LIKE_RE.search(x)) for x in therapist_texts
    ]

    # patient drift: look over ALL patient messages (not just paired ones)
    all_patient_msgs = [txt for (r, txt) in raw_messages if r == "user"]
    patient_drift = [bool(PATIENT_DRIFT_PHRASES.search(x)) for x in all_patient_msgs]

    def rate(flags: List[bool]) -> float:
        return round(sum(flags) / len(flags), 3) if flags else 0.0

    return {
        "n_therapist_turns": len(eval_turns),
        "therapist_avg_words": round(statistics.mean(therapist_wc), 2) if therapist_wc else 0.0,
        "patient_avg_words": round(statistics.mean(patient_wc), 2) if patient_wc else 0.0,
        "therapist_avg_sentences": round(statistics.mean([sentence_count(x) for x in therapist_texts]), 2) if therapist_texts else 0.0,
        "patient_avg_sentences": round(statistics.mean([sentence_count(x) for x in patient_texts]), 2) if patient_texts else 0.0,
        "therapist_exactly_one_question_rate": rate(therapist_q1),
        "therapist_leak_rate": rate(therapist_leak),
        "patient_drift_rate": rate(patient_drift),
    }


# -----------------------------
# Optional OpenAI judge (only if --judge_model set)
# -----------------------------

JUDGE_SYSTEM = """You are an expert CBT supervisor.
Return ONLY strict JSON (no markdown)."""

JUDGE_USER_TEMPLATE = """Evaluate the therapist response using CBT principles.

CONTEXT (earlier messages, may be empty):
{context}

PATIENT:
{patient}

THERAPIST:
{therapist}

Score 1-5 (integer) on:
- empathy
- guided_discovery (one focused question; collaborative; not lecturing)
- cbt_technique (reflect + clarify + gentle cognitive work)
- non_diagnostic (no diagnosis/labels; no medical codes; no technical retrieval terms)
- concision (not overly long)

Return strict JSON exactly:
{{
  "empathy": 1,
  "guided_discovery": 1,
  "cbt_technique": 1,
  "non_diagnostic": 1,
  "concision": 1,
  "avg": 1.0
}}
"""

def maybe_import_openai():
    try:
        from openai import OpenAI  # type: ignore
        return OpenAI
    except Exception as e:
        raise RuntimeError(
            "OpenAI SDK not installed. Run: pip install openai\n"
            f"Import error: {e}"
        )

def parse_json_strict(s: str) -> Optional[Dict[str, Any]]:
    s = (s or "").strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

def run_openai_judge(
    eval_turns: List[EvalTurn],
    judge_model: str,
    out_jsonl_path: Path,
    max_turns: Optional[int] = None,
) -> Dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set in environment.")

    OpenAI = maybe_import_openai()
    client = OpenAI()

    rows = []
    for idx, t in enumerate(eval_turns):
        if max_turns is not None and idx >= max_turns:
            break

        prompt = JUDGE_USER_TEMPLATE.format(
            context=t.context or "",
            patient=t.patient,
            therapist=t.therapist,
        )

        resp = client.chat.completions.create(
            model=judge_model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        content = (resp.choices[0].message.content or "").strip()
        obj = parse_json_strict(content) or {"error": "parse_failed", "raw": content}

        row = {
            "therapist_turn_idx": t.therapist_turn_idx,
            "patient_turn_idx": t.patient_turn_idx,
            "judge": obj,
        }
        rows.append(row)

    out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # aggregate
    def avg_key(key: str) -> float:
        vals = []
        for r in rows:
            j = r.get("judge", {})
            if isinstance(j, dict) and isinstance(j.get(key), (int, float)):
                vals.append(float(j[key]))
        return round(statistics.mean(vals), 3) if vals else 0.0

    return {
        "judge_model": judge_model,
        "judge_n_scored": len(rows),
        "judge_avg_empathy": avg_key("empathy"),
        "judge_avg_guided_discovery": avg_key("guided_discovery"),
        "judge_avg_cbt_technique": avg_key("cbt_technique"),
        "judge_avg_non_diagnostic": avg_key("non_diagnostic"),
        "judge_avg_concision": avg_key("concision"),
        "judge_avg_avg": avg_key("avg"),
    }


# -----------------------------
# Writing outputs
# -----------------------------

def write_turns_csv(eval_turns: List[EvalTurn], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow([
            "therapist_turn_idx",
            "patient_turn_idx",
            "patient",
            "therapist",
            "context",
        ])
        for t in eval_turns:
            w.writerow([t.therapist_turn_idx, t.patient_turn_idx, t.patient, t.therapist, t.context])

def write_metrics_json(metrics: Dict[str, Any], out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

def write_summary_csv(summary_rows: List[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in summary_rows for k in r.keys()})
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in summary_rows:
            w.writerow([r.get(k, "") for k in keys])


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="Files and/or directories (directories scanned recursively).")
    ap.add_argument("--out", required=True, help="Output directory, e.g. output/eval")
    ap.add_argument("--context_window", type=int, default=3, help="How many messages before patient turn to include.")
    ap.add_argument("--judge_model", default=None, help="Optional OpenAI judge model. If set, writes .judge.jsonl per file.")
    ap.add_argument("--max_judge_turns", type=int, default=None, help="Optional cap for judged therapist turns per file.")
    args = ap.parse_args()

    out_dir = Path(os.path.expanduser(args.out)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = list_inputs(args.inputs)
    evaluated_files = 0
    summary_rows: List[Dict[str, Any]] = []

    for fp in candidates:
        msgs = load_transcript_file(fp)
        if msgs is None:
            continue

        eval_turns = build_eval_turns(msgs, context_window=args.context_window)
        if not eval_turns:
            continue

        stem = fp.stem
        per_file_dir = out_dir  # flat output dir
        turns_csv = per_file_dir / f"{stem}.turns.csv"
        metrics_json = per_file_dir / f"{stem}.metrics.json"

        write_turns_csv(eval_turns, turns_csv)

        metrics = compute_heuristics(eval_turns, msgs)
        metrics.update({
            "file": str(fp),
            "stem": stem,
            "context_window": args.context_window,
        })

        # Optional judge
        if args.judge_model:
            judge_jsonl = per_file_dir / f"{stem}.judge.jsonl"
            judge_summary = run_openai_judge(
                eval_turns,
                judge_model=args.judge_model,
                out_jsonl_path=judge_jsonl,
                max_turns=args.max_judge_turns,
            )
            metrics.update(judge_summary)

        write_metrics_json(metrics, metrics_json)

        # add to summary
        summary_rows.append(metrics)
        evaluated_files += 1
        print(f"[OK] {fp} -> {metrics_json.name}")

    if evaluated_files == 0:
        raise SystemExit(
            "No transcript files were evaluated.\n"
            "Make sure you pointed to output/ where transcript JSONs exist."
        )

    summary_csv = out_dir / "summary.csv"
    write_summary_csv(summary_rows, summary_csv)
    print(f"\nWrote summary: {summary_csv}  (rows={len(summary_rows)})")

if __name__ == "__main__":
    main()