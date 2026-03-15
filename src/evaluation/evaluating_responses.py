"""
Notes:

Use different model for generation vs. judge to minimize risk for self-enhancement bias (use a bigger reasoner model that can handle nuances)

We do pointwise evaluation to evaluate the quality of the responses based on X dimension to 

Reasoning model: Gemini or GPT (avoid gpt for maybe bias of own response) -> Structures output is possible in this model (find docs)

Reference: LaaJ https://www.youtube.com/watch?v=8fNP4N46RRo&t=1682s

Input: User query, Prompt used to produce response, the response, rating criteria

Output Rating dimensions: Rationale, CBT Adherance Scores (Performance vs. Alignment), fact decomposition (go over each and check if followed or not)

Constrict output format

get structure output from llm:

Benefits: no reference/label or human ratings needed to get startes; interpretability of scores via rationales

crisp guidelines, binary scale is better to remove noise

Failures: 


Radar plots CBT adherance
Bar chart CBT quality scores differences baseline vs. router
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
import ollama

load_dotenv()


@dataclass
class EvalConfig:
    input_root: Path = Path("output")
    output_root: Path = Path("evals")
    protocol_file: Path = Path("references/cbt-protocols.json")

    judge_backend: str = "openai"
    judge_model: str = "gpt-5.1"

    max_tokens: int = 3000
    max_concurrent: int = 6
    retries: int = 3


class Score(BaseModel):
    score: float = Field(..., ge=0, le=5)
    rationale: str


class ProtocolScores(BaseModel):
    validate_and_reflect: Score
    socratic_questioning: Score
    cognitive_restructuring: Score


class ProtocolEffectiveness(BaseModel):
    effectiveness: float = Field(..., ge=0, le=5)
    rationale: str


class CBTBestPractices(BaseModel):
    therapeutic_relationship: float = Field(..., ge=0, le=5)
    collaboration: float = Field(..., ge=0, le=5)
    goal_oriented: float = Field(..., ge=0, le=5)
    present_focused: float = Field(..., ge=0, le=5)
    educative: float = Field(..., ge=0, le=5)
    guided_discovery: float = Field(..., ge=0, le=5)


class JudgeOutput(BaseModel):
    rationale: str
    protocol_scores: ProtocolScores
    protocol_effectiveness: ProtocolEffectiveness
    cbt_best_practices: CBTBestPractices


def clamp(x):
    try:
        x = float(x)
        return max(0, min(5, x))
    except:
        return 0


def load_protocol_definitions(path):

    with open(path) as f:
        data = json.load(f)

    lines = []

    for p in data["cbt_protocols"]:
        lines.append(f"PROTOCOL: {p['name']}")
        lines.append(f"PURPOSE: {p['purpose']}")

        lines.append("KEY FUNCTIONS:")
        for k in p["key_functions"]:
            lines.append(f"- {k}")

        lines.append("TECHNIQUES:")
        for t in p["techniques"]:
            lines.append(f"- {t}")

        lines.append("")

    return "\n".join(lines)


def load_cbt_best_practices():
    return """
CBT BEST PRACTICES

therapeutic_relationship
The therapist conveys empathy and validation.

collaboration
Therapist and patient explore the problem together.

goal_oriented
Moves toward constructive cognitive change.

present_focused
Focus on current thoughts and behaviors.

educative
Helps the patient understand thinking patterns.

guided_discovery
Encourages the patient to examine beliefs themselves.
"""


def build_prompt(prev_query, query, response, protocol_defs, best_practices):

    prev_section = prev_query if prev_query else "None"

    return f"""
You are evaluating a CBT therapist response.

CBT PROTOCOL DEFINITIONS
{protocol_defs}

CBT BEST PRACTICES
{best_practices}

PREVIOUS USER QUERY
{prev_section}

CURRENT USER QUERY
{query}

THERAPIST RESPONSE
{response}

Use a 0–5 scale.

Return STRICT JSON:

{{
"rationale": "string",

"protocol_scores": {{
"validate_and_reflect": {{"score": number, "rationale": "string"}},
"socratic_questioning": {{"score": number, "rationale": "string"}},
"cognitive_restructuring": {{"score": number, "rationale": "string"}}
}},

"protocol_effectiveness": {{
"effectiveness": number,
"rationale": "string"
}},

"cbt_best_practices": {{
"therapeutic_relationship": number,
"collaboration": number,
"goal_oriented": number,
"present_focused": number,
"educative": number,
"guided_discovery": number
}}
}}

Return ONLY JSON.
"""


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, obj):

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def normalize(v):
    return v / 5


class Judge:

    def __init__(self, cfg, protocol_defs, best_practices):

        self.cfg = cfg
        self.protocol_defs = protocol_defs
        self.best_practices = best_practices
        self.sem = asyncio.Semaphore(cfg.max_concurrent)

        if cfg.judge_backend == "openai":

            api_key = os.getenv("OPENAI_API_KEY")

            if not api_key:
                raise RuntimeError("OPENAI_API_KEY missing")

            self.client = AsyncOpenAI(api_key=api_key)


    async def judge(self, prev_query, query, response):

        prompt = build_prompt(
            prev_query,
            query,
            response,
            self.protocol_defs,
            self.best_practices
        )

        print(f"[DEBUG] Judge request: {query[:60]}")

        if self.cfg.judge_backend == "openai":

            resp = await self.client.responses.parse(
                model=self.cfg.judge_model,
                input=prompt,
                text_format=JudgeOutput,
                max_output_tokens=self.cfg.max_tokens
            )

            return resp.output_parsed

        elif self.cfg.judge_backend == "ollama":

            async with self.sem:

                print("[DEBUG] Sending request to Ollama")

                loop = asyncio.get_running_loop()
                start = time.time()

                result = await loop.run_in_executor(
                    None,
                    lambda: ollama.chat(
                        model=self.cfg.judge_model,
                        messages=[{"role": "user", "content": prompt}],
                        format="json",
                        options={
                            "temperature": 0.0,
                            "num_ctx": 2048
                        }
                    )
                )

                elapsed = time.time() - start
                print(f"[DEBUG] Ollama response received ({elapsed:.2f}s)")

                text = result["message"]["content"]

                print("[DEBUG] Raw output preview:")
                print(text[:300])

                parsed = json.loads(text)

                if "json" in parsed:
                    parsed = parsed["json"]

                # normalize scores
                for k in parsed["protocol_scores"]:
                    parsed["protocol_scores"][k]["score"] = clamp(
                        parsed["protocol_scores"][k]["score"]
                    )

                parsed["protocol_effectiveness"]["effectiveness"] = clamp(
                    parsed["protocol_effectiveness"]["effectiveness"]
                )

                for k in parsed["cbt_best_practices"]:
                    parsed["cbt_best_practices"][k] = clamp(
                        parsed["cbt_best_practices"][k]
                    )

                return JudgeOutput(**parsed)


async def evaluate_file(judge, input_file, output_file):

    payload = read_json(input_file)
    transcript = payload["transcript"]

    tasks = []
    prev_query = None

    for turn_idx, turn in enumerate(transcript):

        query = turn["patient"]["query"]
        response = turn["llm_response"]["response"]

        print(f"[DEBUG] Scheduling turn {turn_idx}")

        tasks.append(
            asyncio.create_task(
                judge.judge(prev_query, query, response)
            )
        )

        prev_query = query

    print("[DEBUG] Waiting for judge results...")

    results = await asyncio.gather(*tasks)

    print("[DEBUG] All judge results returned.")

    turns = []

    for i, r in enumerate(results):

        turns.append({
            "turn": i,
            "judgment": r.model_dump()
        })

    write_json(output_file, {
        "generated": datetime.utcnow().isoformat(),
        "turn_evals": turns
    })


def summarize_modes(eval_dir):

    proto = [
        "validate_and_reflect",
        "socratic_questioning",
        "cognitive_restructuring"
    ]

    modes: Dict[str, Dict] = {}

    for f in sorted(eval_dir.glob("*eval.json")):

        name = f.name

        if name.startswith("baseline"):
            mode = "baseline"
        elif name.startswith("cbt_mcot"):
            mode = "cbt_mcot"
        elif name.startswith("cbt"):
            mode = "cbt"
        else:
            continue

        if mode not in modes:

            modes[mode] = {
                "proto": {p: [] for p in proto},
                "effectiveness": [],
                "best_practices": []
            }

        data = read_json(f)

        for t in data["turn_evals"]:

            j = t["judgment"]

            for p in proto:

                modes[mode]["proto"][p].append(
                    normalize(j["protocol_scores"][p]["score"])
                )

            modes[mode]["effectiveness"].append(
                normalize(j["protocol_effectiveness"]["effectiveness"])
            )

            bp = j["cbt_best_practices"]

            vals = [
                bp["therapeutic_relationship"],
                bp["collaboration"],
                bp["goal_oriented"],
                bp["present_focused"],
                bp["educative"],
                bp["guided_discovery"],
            ]

            modes[mode]["best_practices"].append(
                normalize(sum(vals) / len(vals))
            )

    print("===== MODE COMPARISON =====")

    for mode, data in modes.items():

        print(f"\nMODE: {mode}")

        print("Protocol Execution")

        for p in proto:

            vals = data["proto"][p]
            print(f"{p:25s} {sum(vals)/len(vals):.2f}")

        print("\nProtocol Effectiveness")
        print(sum(data["effectiveness"]) / len(data["effectiveness"]))

        print("\nCBT Best Practices")
        print(sum(data["best_practices"]) / len(data["best_practices"]))


async def run_model(model, cfg):

    protocol_defs = load_protocol_definitions(cfg.protocol_file)
    best_practices = load_cbt_best_practices()

    judge = Judge(cfg, protocol_defs, best_practices)

    files = sorted((cfg.input_root / model).glob("*transcript*.json"))

    for f in files:

        print(f"\n[DEBUG] Starting evaluation for file: {f}")

        out = cfg.output_root / model / f"{f.stem}_eval.json"

        await evaluate_file(judge, f, out)

        print(f"[OK] {f}")


def main():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--judge-backend", default="openai")
    parser.add_argument("--judge-model", default="gpt-5.1")

    args = parser.parse_args()

    cfg = EvalConfig(
        judge_backend=args.judge_backend,
        judge_model=args.judge_model
    )

    asyncio.run(run_model(args.model, cfg))

    eval_dir = cfg.output_root / args.model

    summarize_modes(eval_dir)


if __name__ == "__main__":
    main()