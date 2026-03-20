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
import re
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
    max_concurrent: int = 4
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


def safe_json_load(text):

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise ValueError("No JSON found in model output")

    cleaned = match.group(0)

    cleaned = re.sub(r",\s*}", "}", cleaned)
    cleaned = re.sub(r",\s*]", "]", cleaned)

    return json.loads(cleaned)


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
    The AI agent conveys empathy, validation, and emotional understanding toward the user.

    collaboration
    The AI agent and the user explore the problem together rather than the agent prescribing solutions unilaterally.

    goal_oriented
    The AI agent guides the user toward constructive cognitive or behavioral change.

    present_focused
    The AI agent focuses primarily on the user's current thoughts, emotions, and behaviors.

    educative
    The AI agent helps the user understand cognitive patterns, distortions, or CBT concepts.

    guided_discovery
    The AI agent encourages the user to reflect on and examine their own beliefs through questions and prompts rather than simply giving conclusions.
    """


def build_prompt(prev_query, query, response, protocol_defs, best_practices):

    prev_section = prev_query if prev_query else "None"

    return f"""
    You are evaluating a CBT AI agent response.

    CBT PROTOCOL DEFINITIONS
    {protocol_defs}

    CBT BEST PRACTICES
    {best_practices}

    PREVIOUS USER QUERY
    {prev_section}

    CURRENT USER QUERY
    {query}

    AI AGENT RESPONSE
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

                loop = asyncio.get_running_loop()

                result = await loop.run_in_executor(
                    None,
                    lambda: ollama.chat(
                        model=self.cfg.judge_model,
                        messages=[{"role": "user", "content": prompt}],
                        format="json",
                        options={"temperature": 0.0, "num_ctx": 2048},
                    ),
                )

                text = result["message"]["content"]

                try:
                    parsed = safe_json_load(text)
                except Exception as e:
                    print("JSON parse failure:", e)
                    print(text[:500])
                    raise

                parsed.setdefault("protocol_scores", {})
                parsed.setdefault("protocol_effectiveness", {})
                parsed.setdefault("cbt_best_practices", {})

                for k in [
                    "validate_and_reflect",
                    "socratic_questioning",
                    "cognitive_restructuring",
                ]:

                    item = parsed["protocol_scores"].get(k, {})

                    if isinstance(item, dict):
                        score = clamp(item.get("score", 0))
                        rationale = item.get("rationale", "")
                    else:
                        score = clamp(item)
                        rationale = ""

                    parsed["protocol_scores"][k] = {
                        "score": score,
                        "rationale": rationale,
                    }

                pe = parsed["protocol_effectiveness"]

                parsed["protocol_effectiveness"] = {
                    "effectiveness": clamp(pe.get("effectiveness", 0)),
                    "rationale": pe.get("rationale", ""),
                }

                fields = [
                    "therapeutic_relationship",
                    "collaboration",
                    "goal_oriented",
                    "present_focused",
                    "educative",
                    "guided_discovery",
                ]

                bp = parsed["cbt_best_practices"]

                for k in fields:
                    bp[k] = clamp(bp.get(k, 0))

                parsed["cbt_best_practices"] = bp

                return JudgeOutput(**parsed)


async def evaluate_file(judge, input_file, output_file):

    payload = read_json(input_file)
    transcript = payload["transcript"]

    tasks = []
    prev_query = None

    for turn in transcript:

        query = turn["patient"]["query"]
        response = turn["llm_response"]["response"]

        tasks.append(asyncio.create_task(judge.judge(prev_query, query, response)))

        prev_query = query

    results = await asyncio.gather(*tasks, return_exceptions=True)

    turns = []

    for i, r in enumerate(results):

        if isinstance(r, Exception):
            print("Judge failure:", r)
            continue

        turns.append({"turn": i, "judgment": r.model_dump()})

    write_json(
        output_file,
        {"generated": datetime.utcnow().isoformat(), "turn_evals": turns},
    )


async def run_model(model, cfg):

    protocol_defs = load_protocol_definitions(cfg.protocol_file)
    best_practices = load_cbt_best_practices()

    judge = Judge(cfg, protocol_defs, best_practices)

    files = sorted((cfg.input_root / model).glob("*transcript*.json"))

    for f in files:

        print("Evaluating:", f)

        out = cfg.output_root / model / f"{f.stem}_eval.json"

        await evaluate_file(judge, f, out)

        print("Done:", f)


def main():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--judge-backend", default="openai")
    parser.add_argument("--judge-model", default="gpt-5.1")

    args = parser.parse_args()

    cfg = EvalConfig(
        judge_backend=args.judge_backend,
        judge_model=args.judge_model,
    )

    asyncio.run(run_model(args.model, cfg))


if __name__ == "__main__":
    main()