#!/usr/bin/env python3
"""
cbt_llm.evaluating_benchmark

Evaluate therapist responses in multi-turn transcripts.

- Loads JSON transcripts shaped like:
  {"transcript": [{"role": "...", "content": "..."}, ...], ...}

- Builds therapist-eval turns by pairing each therapist message with the nearest
  previous patient message + optional context window.

- Produces:
  <stem>.turns.csv        (per therapist turn rows; includes protocol execution heuristics)
  <stem>.judge.jsonl      (per therapist turn LLM-judge scores; optional)
  <stem>.metrics.json     (per-file aggregates)
  summary.csv             (across all evaluated files)

Run:
  export PYTHONPATH=src
  export OPENAI_API_KEY="..."
  python -m src.cbt_llm.evaluating_benchmark output/gemma --out output/results/gemma --judge_model gpt-4o-mini
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

# ======================================================
# CBT JUDGE CONFIG (USER TEMPLATE: DO NOT CHANGE)
# ======================================================

DIMENSIONS = [
    "understanding",
    "cognitive_restructuring",
    "behavioral_guidance",
    "cognitive_focus",
    "self_reflection",
    "cbt_adherence",
]

# Strict grading lives here (system), without touching your user template text.
JUDGE_SYSTEM = """You are an expert Cognitive-Behavioral Therapy (CBT) supervisor.

Strict grading rules:
- Scores must reflect CBT TECHNIQUE quality, not general supportiveness.
- If a CBT skill is missing or only implied, score it 1–2.
- Reserve 4–5 for explicit, concrete CBT execution (not vague exploration).
- Use decimal scores if needed (e.g., 2.5, 3.5).
- Return ONLY valid JSON. No extra text.
"""

JUDGE_USER_TEMPLATE = """
You will now evaluate a therapist's response in a CBT framework.

=== CONTEXT (earlier conversation, may be empty) ===
{context_block}

=== PATIENT'S CURRENT MESSAGE ===
{patient_text}

=== THERAPIST'S RESPONSE TO EVALUATE ===
{therapist_text}

Please rate the therapist response from 1 to 5 (1 = very poor, 5 = excellent) on the following dimensions:

1.⁠ ⁠understanding:
   - How accurately does the therapist understand and reflect the patient's main concerns,
     thoughts, feelings, and situation?
   - Does the response show a clear grasp of what the patient is actually saying?

2.⁠ ⁠cognitive_restructuring:
   - Does the therapist help the patient notice, examine, or question unhelpful thoughts or beliefs?
   - Does the response move toward identifying cognitive patterns or gently challenging them?

3.⁠ ⁠behavioral_guidance:
   - Does the therapist suggest or hint at concrete, realistic behavioral steps or strategies
     that fit within CBT (e.g., activity scheduling, exposure, skills practice)?
   - Even if not giving explicit homework, is there a behavioral direction?

4.⁠ ⁠cognitive_focus:
   - Does the therapist keep the focus on the relationship between thoughts, feelings, and behaviors?
   - Is the response grounded in CBT's cognitive model rather than staying vague or purely supportive?

5.⁠ ⁠self_reflection:
   - Does the therapist encourage the patient to reflect on their own thoughts, feelings, triggers,
     or underlying patterns (e.g., asking helpful questions, using guided discovery/laddering)?

6.⁠ ⁠cbt_adherence:
   - Overall, how well does this response adhere to CBT principles and good CBT practice?
   - Consider structure, focus on present problems, collaborative tone, and alignment with CBT methods.

Output MUST be valid JSON, with exactly these numeric keys:

{{
  "understanding": <number 1-5>,
  "cognitive_restructuring": <number 1-5>,
  "behavioral_guidance": <number 1-5>,
  "cognitive_focus": <number 1-5>,
  "self_reflection": <number 1-5>,
  "cbt_adherence": <number 1-5>,
  "avg_score": <number 1-5>  # the simple arithmetic mean of the 6 scores above
}}

Return ONLY the JSON. No explanation, no extra text.
"""

# ======================================================
# ROLE NORMALIZATION + LOADING
# ======================================================

ROLE_MAP = {
    "patient": "user",
    "user": "user",
    "client": "user",
    "human": "user",
    "therapist": "assistant",
    "assistant": "assistant",
    "counselor": "assistant",
    "model": "assistant",
}

def _norm_role(r: Any) -> str:
    if not isinstance(r, str):
        return "other"
    rr = r.strip().lower()
    return ROLE_MAP.get(rr, rr or "other")

def load_transcript(path: Path) -> Optional[List[Tuple[str, str]]]:
    """
    Returns normalized list of (role, text), or None if not a transcript JSON.
    Only supports {"transcript":[...]} intentionally.
    """
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    msgs = obj.get("transcript")
    if not isinstance(msgs, list):
        return None

    out: List[Tuple[str, str]] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = _norm_role(m.get("role", ""))
        text = (m.get("content") or "").strip()
        if not isinstance(text, str) or not text.strip():
            continue
        if role in ("user", "assistant"):
            out.append((role, text.strip()))

    return out if len(out) >= 2 else None

# ======================================================
# TURN PAIRING
# ======================================================

@dataclass
class EvalTurn:
    therapist_turn_idx: int
    patient_turn_idx: int
    patient: str
    therapist: str
    context: str

def build_eval_turns(messages: List[Tuple[str, str]], ctx_window: int = 2) -> List[EvalTurn]:
    """
    For each assistant message, pair with most recent preceding user message.
    context = up to ctx_window messages BEFORE patient turn.
    """
    turns: List[EvalTurn] = []
    for i, (role, ttext) in enumerate(messages):
        if role != "assistant":
            continue
        j = i - 1
        while j >= 0 and messages[j][0] != "user":
            j -= 1
        if j < 0:
            continue
        ctx_start = max(0, j - ctx_window)
        ctx_msgs = messages[ctx_start:j]
        context = "\n".join(f"{r}: {txt}" for r, txt in ctx_msgs) if ctx_msgs else ""
        turns.append(
            EvalTurn(
                therapist_turn_idx=i,
                patient_turn_idx=j,
                patient=messages[j][1],
                therapist=ttext,
                context=context,
            )
        )
    return turns

# ======================================================
# PROTOCOL EXECUTION HEURISTICS (NO PAIRWISE NEEDED)
# ======================================================

# Validation/Reflection signals (broad)
VALIDATE_RE = re.compile(
    r"\b(i (understand|hear)|it sounds like|it seems like|that sounds|makes sense|you're feeling|you are feeling)\b",
    re.I,
)

# Socratic signals (open-ended + probing evidence/alternatives)
SOCRATIC_Q_RE = re.compile(r"\b(what|how|when|where|why|could|might)\b.*\?", re.I)
EVIDENCE_ALT_RE = re.compile(
    r"\b(evidence|supports|against|alternative|another explanation|different interpretation|what makes you think)\b",
    re.I,
)

# Reframing execution: explicit thought/belief + alternative interpretation
THOUGHT_RE = re.compile(r"\b(thought|belief|assumption|interpretation|story i'm telling myself)\b", re.I)
ALT_RE = re.compile(r"\b(alternative|another way|different perspective|reframe|more balanced)\b", re.I)

# Refusal / bailout patterns that should tank CBT adherence even if "nice"
REFUSAL_RE = re.compile(
    r"\b(i'?m unable to|i cannot provide|can'?t help with|contact a (crisis hotline|hotline)|seek professional help)\b",
    re.I,
)

def score_validation_execution(text: str) -> float:
    # 0.0/1.0 simple and interpretable
    return 1.0 if VALIDATE_RE.search(text) else 0.0

def score_socratic_execution(text: str) -> float:
    # 0.0 / 0.5 / 1.0
    score = 0.0
    if SOCRATIC_Q_RE.search(text):
        score += 0.5
    if EVIDENCE_ALT_RE.search(text):
        score += 0.5
    return min(1.0, score)

def score_reframing_execution(text: str) -> float:
    # 0.0 / 0.5 / 1.0
    has_thought = bool(THOUGHT_RE.search(text))
    has_alt = bool(ALT_RE.search(text))
    if has_thought and has_alt:
        return 1.0
    if has_thought or has_alt:
        return 0.5
    return 0.0

def score_refusal(text: str) -> float:
    return 1.0 if REFUSAL_RE.search(text) else 0.0

# ======================================================
# OPENAI JUDGE
# ======================================================

def _clip_1_5(x: float) -> float:
    return max(1.0, min(5.0, float(x)))

def parse_json_safe(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.S)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}

def run_openai_judge(
    turns: List[EvalTurn],
    model: str,
    out_jsonl: Path,
    max_turns: Optional[int] = None,
) -> Dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set in environment.")

    from openai import OpenAI
    client = OpenAI()

    rows: List[Dict[str, Any]] = []

    for idx, t in enumerate(turns):
        if max_turns is not None and idx >= max_turns:
            break

        prompt = JUDGE_USER_TEMPLATE.format(
            context_block=t.context,
            patient_text=t.patient,
            therapist_text=t.therapist,
        )

        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )

        obj = parse_json_safe(resp.choices[0].message.content or "")

        # Enforce required keys, allow floats, clip 1-5
        vals: List[float] = []
        for d in DIMENSIONS:
            v = obj.get(d)
            if isinstance(v, (int, float)):
                vv = _clip_1_5(float(v))
                obj[d] = vv
                vals.append(vv)
            else:
                # strict default if missing
                obj[d] = 1.0
                vals.append(1.0)

        obj["avg_score"] = _clip_1_5(sum(vals) / len(vals)) if vals else 1.0

        # attach indices for alignment
        obj["_therapist_turn_idx"] = t.therapist_turn_idx
        obj["_patient_turn_idx"] = t.patient_turn_idx

        rows.append(obj)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def mean_key(key: str) -> float:
        xs = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]
        return round(statistics.mean(xs), 3) if xs else 0.0

    summary = {f"judge_avg_{d}": mean_key(d) for d in DIMENSIONS}
    summary["judge_avg_avg_score"] = mean_key("avg_score")
    summary["judge_n_scored"] = len(rows)
    summary["judge_model"] = model
    return summary

# ======================================================
# OUTPUT WRITING
# ======================================================

def write_turns_csv(
    turns: List[EvalTurn],
    out_csv: Path,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "therapist_turn_idx",
            "patient_turn_idx",
            "patient",
            "therapist",
            "context",
            "validation_execution",
            "socratic_execution",
            "reframing_execution",
            "refusal_flag",
        ])
        for t in turns:
            w.writerow([
                t.therapist_turn_idx,
                t.patient_turn_idx,
                t.patient,
                t.therapist,
                t.context,
                score_validation_execution(t.therapist),
                score_socratic_execution(t.therapist),
                score_reframing_execution(t.therapist),
                score_refusal(t.therapist),
            ])

def write_metrics_json(metrics: Dict[str, Any], out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

def write_summary_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)

# ======================================================
# MAIN
# ======================================================

def list_inputs(paths: List[str]) -> List[Path]:
    files: List[Path] = []
    for p in paths:
        pp = Path(p).expanduser().resolve()
        if pp.is_file():
            files.append(pp)
        elif pp.is_dir():
            files.extend(sorted(pp.rglob("*.json")))
    # de-dupe
    seen = set()
    uniq: List[Path] = []
    for f in files:
        s = str(f)
        if s not in seen:
            uniq.append(f)
            seen.add(s)
    return uniq

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="JSON file(s) or directory(ies) containing transcript JSONs")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--context_window", type=int, default=2, help="How many messages before patient turn to include as context")
    ap.add_argument("--judge_model", default=None, help="Optional OpenAI judge model")
    ap.add_argument("--max_judge_turns", type=int, default=None, help="Cap judged therapist turns per file")
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = list_inputs(args.inputs)
    print(f"[INFO] Found {len(candidates)} json files")

    summary_rows: List[Dict[str, Any]] = []
    evaluated = 0

    for fp in candidates:
        if "debug" in fp.name.lower():
            continue

        msgs = load_transcript(fp)
        if msgs is None:
            continue

        turns = build_eval_turns(msgs, ctx_window=args.context_window)
        if not turns:
            continue

        stem = fp.stem
        turns_csv = out_dir / f"{stem}.turns.csv"
        metrics_json = out_dir / f"{stem}.metrics.json"
        judge_jsonl = out_dir / f"{stem}.judge.jsonl"

        write_turns_csv(turns, turns_csv)

        # Derived protocol metrics (aggregate)
        val_scores = [score_validation_execution(t.therapist) for t in turns]
        soc_scores = [score_socratic_execution(t.therapist) for t in turns]
        ref_scores = [score_reframing_execution(t.therapist) for t in turns]
        refusal_flags = [score_refusal(t.therapist) for t in turns]

        metrics: Dict[str, Any] = {
            "file": str(fp),
            "stem": stem,
            "n_therapist_turns": len(turns),
            "context_window": args.context_window,

            "validation_execution_avg": round(statistics.mean(val_scores), 3) if val_scores else 0.0,
            "socratic_execution_avg": round(statistics.mean(soc_scores), 3) if soc_scores else 0.0,
            "reframing_execution_avg": round(statistics.mean(ref_scores), 3) if ref_scores else 0.0,
            "refusal_rate": round(statistics.mean(refusal_flags), 3) if refusal_flags else 0.0,

            # A single summary number you can plot:
            "protocol_execution_avg": round(
                statistics.mean([
                    statistics.mean(val_scores) if val_scores else 0.0,
                    statistics.mean(soc_scores) if soc_scores else 0.0,
                    statistics.mean(ref_scores) if ref_scores else 0.0,
                ]),
                3,
            ) if turns else 0.0,
        }

        # Optional LLM judge
        if args.judge_model:
            judge_summary = run_openai_judge(
                turns,
                model=args.judge_model,
                out_jsonl=judge_jsonl,
                max_turns=args.max_judge_turns,
            )
            metrics.update(judge_summary)

        write_metrics_json(metrics, metrics_json)

        summary_rows.append(metrics)
        evaluated += 1
        print(f"[OK] {fp} -> {metrics_json.name}")

    if evaluated == 0:
        print("[WARN] No files evaluated.")
        return

    write_summary_csv(summary_rows, out_dir / "summary.csv")
    print(f"[DONE] Evaluated {evaluated} transcripts. Wrote: {out_dir / 'summary.csv'}")

if __name__ == "__main__":
    main()
