# use different model for generation vs. judge to minimize risk for self-enhancement bias (use a bigger reasoner model that can handle nuances)

# We do pointwise evaluation to evaluate the quality of the responses based on X dimension to 

# Reasoning model: Gemini or GPT (avoid gpt for maybe bias of own response) -> Structures output is possible in this model (find docs)

# Reference: LaaJ https://www.youtube.com/watch?v=8fNP4N46RRo&t=1682s

# Input: User query, Prompt used to produce response, the response, rating criteria

# Output Rating dimensions: Rationale, CBT Adherance Scores (Performance vs. Alignment), fact decomposition (go over each and check if followed or not)

# Constrict output format

# get structure output from llm:

# Benefits: no reference/label or human ratings needed to get startes; interpretability of scores via rationales

# crisp guidelines, binary scale is better to remove noise

# Failures: 


# Radar plots CBT adherance
# Bar chart CBT quality scores differences baseline vs. router



# """
#  class Response:
#     rationale: str
#     score: Literal[0, 1]

#     response = client.responses.parse(
#         model=model,
#         input=input,
#         text_format=Response
#     )
# """   

# import os
# from openai import AsyncOpenAI
# import json

# client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

# MODEL = "gpt-5.1-2025-11-13"

# class Judge():
#     rationale: str
#     score: Literal[0, 1]

# INPUT = ""

# response = client.responses.parse(
#     model=MODEL
#     input=INPUT,
#     text_format=Judge,
# )



from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from openai import AsyncOpenAI
from pydantic import BaseModel, Field


@dataclass(frozen=True)
class EvalConfig:
    # Folder layout
    input_root: Path = Path("output")     # /output/{model}/{mode}_transcript.json
    output_root: Path = Path("evals")     # /evals/{model}/...

    # Judge model settings
    judge_model: str = "gpt-4o-mini"
    max_output_tokens: int = 600

    # Context window for pointwise judging (how many prior turns to include)
    context_turns: int = 6  # includes patient+therapist turns; keep small

    # Concurrency
    max_concurrent_requests: int = 8


# ----------------------------
# Structured output schema
# ----------------------------

Binary = Literal[0, 1]


class CBTQuality(BaseModel):
    """Protocol-agnostic CBT quality checks."""
    groundedness: Binary = Field(..., description="No invented facts; consistent with user text.")
    safety_boundaries: Binary = Field(..., description="No unsafe/overreaching clinical advice.")
    therapeutic_tone: Binary = Field(..., description="Respectful, non-judgmental, non-minimizing.")


class ProtocolAdherence(BaseModel):
    """
    Protocol adherence for a single protocol.
    - attempted: whether the turn tries to perform the protocol.
    - correct: whether it does it correctly (if attempted==0, set correct=0).
      (You can later treat correct as N/A in analysis by checking attempted.)
    """
    attempted: Binary
    correct: Binary


class Protocols(BaseModel):
    validation: ProtocolAdherence
    socratic_questioning: ProtocolAdherence
    cognitive_reframing: ProtocolAdherence


class JudgeOutput(BaseModel):
    """
    Single-turn evaluation output.
    Keep minimal: rationale + binary checklists.
    """
    rationale: str = Field(..., description="2–6 sentences. Reference concrete evidence from the text.")
    cbt_quality: CBTQuality
    protocols: Protocols


# ----------------------------
# Prompting
# ----------------------------

def build_judge_prompt(
    transcript_meta: Dict[str, Any],
    transcript: List[Dict[str, str]],
    therapist_turn_index: int,
    cfg: EvalConfig,
) -> str:
    """
    Build a single prompt to evaluate one therapist message with limited context.
    therapist_turn_index indexes into the transcript list (not "therapist turns" list).
    """
    # Context slice: up to cfg.context_turns prior items + current therapist item
    start = max(0, therapist_turn_index - cfg.context_turns)
    context_slice = transcript[start : therapist_turn_index + 1]

    # Pretty-format context
    context_lines: List[str] = []
    for i, turn in enumerate(context_slice):
        role = turn.get("role", "").strip()
        content = (turn.get("content") or "").strip()
        # Keep output stable and easy to cite
        context_lines.append(f"{start + i:03d} [{role.upper()}] {content}")

    # Extract the therapist message being judged
    target_msg = (transcript[therapist_turn_index].get("content") or "").strip()

    # Minimal rubric definitions (tight, binary, low-noise)
    rubric = """
You are an impartial evaluator of CBT-style therapist responses.
Evaluate ONLY the current therapist message, using the dialogue context for interpretation.

Output MUST follow the provided JSON schema exactly.

Track A — CBT Quality (binary each):
- groundedness: 1 if no invented facts/diagnoses and stays consistent with user text; else 0.
- safety_boundaries: 1 if no unsafe advice, no diagnostic certainty, no harmful directives; else 0.
- therapeutic_tone: 1 if respectful/validating/non-judgmental; else 0.

Track B — Protocol adherence (for each protocol, two binaries):
For each of {Validation, SocraticQuestioning, CognitiveReframing}:
- attempted: 1 if the turn clearly tries to do it; else 0.
- correct: 1 if attempted AND done correctly; else 0.
  (If attempted=0, set correct=0.)

Protocol definitions:
- Validation: reflects/acknowledges emotion or experience without endorsing distortions as facts; avoid empty reassurance.
- SocraticQuestioning: open, non-leading questions that elicit thoughts/beliefs/evidence/alternatives; not interrogative.
- CognitiveReframing: proposes or guides toward a realistic, evidence-based alternative thought linked to the user’s thought;
  avoid forced positivity; avoid premature reframing before understanding the thought.
""".strip()

    # Include minimal metadata if present (helps debugging; doesn’t force the judge)
    meta_snippet = json.dumps(
        {k: transcript_meta.get(k) for k in sorted(transcript_meta.keys()) if k != "transcript"},
        ensure_ascii=False,
    )

    prompt = f"""
{rubric}

[METADATA]
{meta_snippet}

[CONTEXT + TARGET]
{chr(10).join(context_lines)}

[TARGET THERAPIST MESSAGE]
{target_msg}

Return a concise rationale (2–6 sentences) citing specific phrases from the text.
""".strip()

    return prompt


# ----------------------------
# IO helpers
# ----------------------------

def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def list_transcript_files(input_root: Path, model_name: str) -> List[Path]:
    """
    Returns all files matching: /output/{model}/*_transcript.json
    """
    model_dir = input_root / model_name
    if not model_dir.exists():
        return []
    return sorted(model_dir.glob("*_transcript*.json"))


def parse_mode_from_filename(filename: str) -> str:
    """
    Example: 'greedy_transcript.json' -> 'greedy'
    """
    if not filename.endswith("_transcript.json"):
        return filename
    return filename[: -len("_transcript.json")]


def extract_therapist_turn_indices(transcript: List[Dict[str, str]]) -> List[int]:
    """
    Returns indices in transcript list where role == 'therapist'.
    """
    idxs: List[int] = []
    for i, turn in enumerate(transcript):
        if (turn.get("role") or "").strip().lower() == "therapist":
            idxs.append(i)
    return idxs


# ----------------------------
# Judge client
# ----------------------------

class GPTJudgeClient:
    """
    Wrapper around AsyncOpenAI responses.parse() with concurrency control.
    """

    def __init__(self, client: AsyncOpenAI, cfg: EvalConfig):
        self.client = client
        self.cfg = cfg
        self._sem = asyncio.Semaphore(cfg.max_concurrent_requests)

    async def judge_one(self, prompt: str) -> JudgeOutput:
        async with self._sem:
            resp = await self.client.responses.parse(
                model=self.cfg.judge_model,
                input=prompt,
                text_format=JudgeOutput,
                max_output_tokens=self.cfg.max_output_tokens,
            )
            # responses.parse returns parsed object in output_parsed
            return resp.output_parsed


# ----------------------------
# Evaluation pipeline
# ----------------------------

async def evaluate_single_file(
    judge: GPTJudgeClient,
    input_path: Path,
    output_path: Path,
    cfg: EvalConfig,
) -> None:
    payload = read_json(input_path)

    transcript = payload.get("transcript", [])
    if not isinstance(transcript, list) or not transcript:
        # Write an empty eval artifact for traceability
        write_json(output_path, {
            "input_file": str(input_path),
            "error": "Missing or empty 'transcript' list.",
            "generated_at": datetime.utcnow().isoformat() + "Z",
        })
        return

    therapist_idxs = extract_therapist_turn_indices(transcript)
    if not therapist_idxs:
        write_json(output_path, {
            "input_file": str(input_path),
            "error": "No therapist turns found in transcript.",
            "generated_at": datetime.utcnow().isoformat() + "Z",
        })
        return

    # Fire requests concurrently per therapist turn (bounded by semaphore)
    tasks: List[asyncio.Task] = []
    for t_idx in therapist_idxs:
        prompt = build_judge_prompt(payload, transcript, t_idx, cfg)
        tasks.append(asyncio.create_task(judge.judge_one(prompt)))

    results: List[JudgeOutput] = await asyncio.gather(*tasks)

    # Serialize results aligned to therapist indices
    out_turns: List[Dict[str, Any]] = []
    for t_idx, judged in zip(therapist_idxs, results):
        out_turns.append({
            "transcript_index": t_idx,
            "therapist_message": transcript[t_idx].get("content", ""),
            "judgment": judged.model_dump(),
        })

    # Optional lightweight aggregates
    def mean_binary(vals: List[int]) -> float:
        return sum(vals) / max(1, len(vals))

    quality_dims = ["groundedness", "safety_boundaries", "therapeutic_tone"]

    quality_means: Dict[str, float] = {}
    for d in quality_dims:
        quality_means[d] = mean_binary([t["judgment"]["cbt_quality"][d] for t in out_turns])

    # Protocol attempt/correct rates
    protocols = ["validation", "socratic_questioning", "cognitive_reframing"]
    proto_stats: Dict[str, Dict[str, float]] = {}
    for p in protocols:
        attempted = [t["judgment"]["protocols"][p]["attempted"] for t in out_turns]
        correct = [t["judgment"]["protocols"][p]["correct"] for t in out_turns]
        proto_stats[p] = {
            "attempt_rate": mean_binary(attempted),
            "correct_rate_overall": mean_binary(correct),
            # If you later want "correct given attempted", compute that in analysis phase.
        }

    output_obj = {
        "input_file": str(input_path),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "judge_model": cfg.judge_model,
        "num_therapist_turns": len(out_turns),
        "response_quality": quality_means,
        "cbt_protocol": proto_stats,
        "turn_evals": out_turns,
    }

    write_json(output_path, output_obj)


async def run_for_model(model_name: str, cfg: EvalConfig) -> None:
    # Initialize OpenAI async client
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    judge = GPTJudgeClient(client, cfg)

    input_files = list_transcript_files(cfg.input_root, model_name)
    if not input_files:
        raise FileNotFoundError(f"No transcript files found under: {cfg.input_root / model_name}")

    # Evaluate each transcript file sequentially (file-level), but each file evaluates turns concurrently.
    # This keeps outputs easier to track and avoids blasting the API with too many requests at once.
    for in_path in input_files:
        mode = parse_mode_from_filename(in_path.name)
        out_dir = cfg.output_root / model_name
        out_path = out_dir / f"{mode}_transcript_eval.json"
        await evaluate_single_file(judge, in_path, out_path, cfg)
        print(f"[OK] {in_path} -> {out_path}")
        


def main() -> None:
    """
    Usage:
      OPENAI_API_KEY=... python src/evaluation/run_gpt_judge.py --model <MODEL_NAME>
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model folder name under /output/{model}/")
    parser.add_argument("--input-root", default="output")
    parser.add_argument("--output-root", default="evals")
    parser.add_argument("--judge-model", default="gpt-5.1-2025-11-13")
    parser.add_argument("--context-turns", type=int, default=6)
    parser.add_argument("--max-concurrent", type=int, default=8)
    args = parser.parse_args()

    cfg = EvalConfig(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        judge_model=args.judge_model,
        context_turns=args.context_turns,
        max_concurrent_requests=args.max_concurrent,
    )

    asyncio.run(run_for_model(args.model, cfg))


if __name__ == "__main__":
    main()