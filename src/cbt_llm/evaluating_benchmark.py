"""
cbt_llm.evaluating_benchmark

Evaluate therapist responses in multi-turn CBT transcripts.

Key features:
- Therapist-only evaluation
- Protocol-aligned CBT judging
- Schema + retrieval attribution
- Automatic output routing to: evaluation/{model}
- LLM-as-a-judge (OpenAI)
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
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

PROTOCOL_DIMENSIONS = [
    "validate_and_reflect_quality",
    "socratic_questioning_quality",
    "cognitive_reframing_quality",
]

CBT_DIMENSIONS = [
    "cbt_adherence",
    "cognitive_focus",
    "self_reflection",
]

ATTRIBUTION_DIMENSIONS = [
    "schema_grounding",
    "retrieval_utility",
]

ALL_DIMENSIONS = PROTOCOL_DIMENSIONS + CBT_DIMENSIONS + ATTRIBUTION_DIMENSIONS

JUDGE_SYSTEM = """You are an expert Cognitive Behavioral Therapy (CBT) supervisor.

You evaluate therapist responses for CBT protocol execution and quality.

Rules:
- Judge ONLY the therapist response.
- Do NOT reward generic empathy.
- Score low (1–2) if a protocol is missing or vague.
- Score high (4–5) only for concrete execution.
- Return ONLY valid JSON.
"""

JUDGE_USER_TEMPLATE = """
Evaluate the therapist response below.

=== RECENT CONTEXT ===
{context_block}

=== PATIENT MESSAGE ===
{patient_text}

=== THERAPIST RESPONSE ===
{therapist_text}

Score each dimension from 1 (poor) to 5 (excellent).

CBT PROTOCOL EXECUTION:
1. validate_and_reflect_quality
2. socratic_questioning_quality
3. cognitive_reframing_quality

CBT PRACTICE:
4. cbt_adherence
5. cognitive_focus
6. self_reflection

ATTRIBUTION:
7. schema_grounding
8. retrieval_utility

Return JSON with EXACTLY these keys plus avg_score.
"""

ROLE_MAP = {
    "patient": "user",
    "user": "user",
    "client": "user",
    "therapist": "assistant",
    "assistant": "assistant",
}

def _norm_role(r: Any) -> str:
    return ROLE_MAP.get(str(r).lower(), "other")

def load_transcript(path: Path) -> Optional[List[Tuple[str, str]]]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    msgs = obj.get("transcript")
    if not isinstance(msgs, list):
        return None

    out: List[Tuple[str, str]] = []
    for m in msgs:
        role = _norm_role(m.get("role"))
        text = (m.get("content") or "").strip()
        if role in ("user", "assistant") and text:
            out.append((role, text))

    return out if len(out) >= 2 else None

@dataclass
class EvalTurn:
    therapist_turn_idx: int
    patient_turn_idx: int
    patient: str
    therapist: str
    context: str

def build_eval_turns(
    messages: List[Tuple[str, str]],
    ctx_window: int = 2,
) -> List[EvalTurn]:
    """
    Pair each therapist turn with the immediately preceding patient turn.
    Include up to ctx_window earlier messages for context.
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
        context = "\n".join(f"{r}: {txt}" for r, txt in messages[ctx_start:j])

        turns.append(EvalTurn(
            therapist_turn_idx=i,
            patient_turn_idx=j,
            patient=messages[j][1],
            therapist=ttext,
            context=context,
        ))
    return turns


def run_openai_judge(
    turns: List[EvalTurn],
    model: str,
    out_jsonl: Path,
    max_turns: Optional[int] = None,
) -> Dict[str, Any]:
    client = OpenAI()

    rows: List[Dict[str, Any]] = []

    for idx, t in enumerate(turns):
        if max_turns and idx >= max_turns:
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

        obj = json.loads(resp.choices[0].message.content)
        scores = [float(obj[k]) for k in ALL_DIMENSIONS]
        obj["avg_score"] = sum(scores) / len(scores)
        obj["_therapist_turn_idx"] = t.therapist_turn_idx
        rows.append(obj)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    return {
        f"judge_avg_{k}": round(statistics.mean(r[k] for r in rows), 3)
        for k in ALL_DIMENSIONS
    } | {
        "judge_avg_avg_score": round(statistics.mean(r["avg_score"] for r in rows), 3),
        "judge_n_scored": len(rows),
        "judge_model": model,
    }


def infer_model_name(input_path: Path) -> str:
    """
    output/gemma -> gemma
    output/mistral -> mistral
    """
    return input_path.name

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Directory containing transcript JSONs (e.g. output/gemma)")
    ap.add_argument("--out", default=None, help="Base evaluation directory (default: evaluation/)")
    ap.add_argument("--context_window", type=int, default=2)
    ap.add_argument("--judge_model", default="gpt-4o-mini")
    ap.add_argument("--max_judge_turns", type=int)
    args = ap.parse_args()

    input_dir = Path(args.input).resolve()
    model_name = infer_model_name(input_dir)

    base_out = Path(args.out) if args.out else Path("evaluation")
    out_dir = (base_out / model_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []

    for fp in sorted(input_dir.rglob("*.json")):
        msgs = load_transcript(fp)
        if not msgs:
            continue

        turns = build_eval_turns(msgs, args.context_window)
        if not turns:
            continue

        stem = fp.stem
        judge_jsonl = out_dir / f"{stem}.judge.jsonl"
        metrics_json = out_dir / f"{stem}.metrics.json"

        metrics = {
            "file": str(fp),
            "model": model_name,
            "n_therapist_turns": len(turns),
            "context_window": args.context_window,
        }

        metrics |= run_openai_judge(
            turns,
            model=args.judge_model,
            out_jsonl=judge_jsonl,
            max_turns=args.max_judge_turns,
        )

        metrics_json.write_text(json.dumps(metrics, indent=2))
        summary_rows.append(metrics)

    if summary_rows:
        summary_csv = out_dir / "summary.csv"
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, summary_rows[0].keys())
            w.writeheader()
            w.writerows(summary_rows)

    print(f"Evaluated {len(summary_rows)} transcripts. Saved to {out_dir}")

if __name__ == "__main__":
    main()
