"""
Ablation pipeline for CBT-LLM.

Runs the full CBT session with one component removed at a time:
  - no_rag:      removes SNOMED retrieval + NLI verification
  - no_schema:   removes cognitive model (schema extraction)
  - no_protocol: removes CBT principles/protocols

Usage:
    python -m cbt_llm.ablation_convo \
        --therapist_model gemma3:12b \
        --ablation_variant no_rag \
        --seed "I keep overthinking everything..." \
        --core_issue "anxiety" \
        --transcript_json output/ablation/no_rag_001.json
"""

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI
from neo4j import GraphDatabase

from cbt_llm.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from cbt_llm.pipelines.findings_pipeline import FindingsPipeline
from cbt_llm.multiturn_convo import (
    OllamaChat,
    OpenAIChat,
    THERAPIST_CBT_PROMPT,
    PATIENT_SYSTEM,
    safe_extract_schema,
    build_hidden_context,
    looks_like_therapist_leak,
    looks_like_patient_drift,
    parse_gpt_oss_output,
    classify_transcript,
)

ABLATION_VARIANTS = {
    "no_rag":      {"use_rag": False, "use_schema": True,  "use_protocol": True},
    "no_schema":   {"use_rag": True,  "use_schema": False, "use_protocol": True},
    # "no_protocol": {"use_rag": True,  "use_schema": True,  "use_protocol": False},
}

OLLAMA_MODELS = {"gpt-oss:20b", "mistral:7b", "gemma3:12b", "deepseek-r1:8b"}


def run_ablation_session(
    therapist_model: str,
    patient_model: str,
    ablation_variant: str,
    turns: int,
    k: int,
    seed: str,
    core_issue: str,
    transcript_json: str,
):
    if ablation_variant not in ABLATION_VARIANTS:
        raise ValueError(
            f"Unknown ablation_variant '{ablation_variant}'. "
            f"Choose from: {list(ABLATION_VARIANTS)}"
        )

    flags = ABLATION_VARIANTS[ablation_variant]
    use_rag      = flags["use_rag"]
    use_schema   = flags["use_schema"]
    use_protocol = flags["use_protocol"]

    therapist_llm = (
        OllamaChat(therapist_model) if therapist_model in OLLAMA_MODELS
        else OpenAIChat(therapist_model)
    )

    patient_llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    findings_pipeline = FindingsPipeline(driver, k=k)

    base_prompt = THERAPIST_CBT_PROMPT

    transcript = []
    patient_chat = [{"role": "system", "content": PATIENT_SYSTEM.format(core_issue=core_issue)}]
    last_patient = seed

    for turn_idx in range(turns):

        schema = safe_extract_schema(last_patient) if use_schema else None

        rag = None
        reasoning = None

        if use_rag:
            rag = findings_pipeline.get_findings(last_patient)

        hidden_context = build_hidden_context(schema, rag, use_protocol)

        therapist_messages = [{"role": "system", "content": base_prompt}]

        if hidden_context:
            therapist_messages.append({
                "role": "system",
                "content": "IMPORTANT CONTEXT (DO NOT REVEAL):\n" + hidden_context,
            })

        therapist_messages.append({"role": "user", "content": last_patient})

        raw_reply = therapist_llm.chat(
            therapist_messages,
            temperature=0.15,
            num_predict=8000,
        ).strip()

        therapist_reply = raw_reply
        print(raw_reply)

        if therapist_model == "gpt-oss:20b":
            therapist_reply, reasoning = parse_gpt_oss_output(raw_reply)
        else:
            if "REASONING:" in raw_reply and "FINAL RESPONSE:" in raw_reply:
                try:
                    reasoning_text = raw_reply.split("REASONING:")[1].split("FINAL RESPONSE:")[0].strip()
                    therapist_reply = raw_reply.split("FINAL RESPONSE:")[1].strip()
                    reasoning = json.loads(reasoning_text)

                    print("\n======= RAW COT REASONING ========")
                    print(json.dumps(reasoning, indent=2))
                    print("\n===============\n")
                except Exception:
                    reasoning = None

        if looks_like_therapist_leak(therapist_reply):
            rewritten = therapist_llm.chat(
                therapist_messages + [{
                    "role": "system",
                    "content": (
                        "Rewrite the response strictly following the rules. "
                        "Interpret meaning and advance insight."
                    ),
                }],
                temperature=0.1,
                num_predict=8000,
            ).strip()
            if rewritten:
                therapist_reply = rewritten

        if not therapist_reply:
            raise RuntimeError(
                f"Empty therapist response at turn {turn_idx} "
                f"from model {therapist_model}. Aborting run."
            )

        patient_resp = patient_llm.chat.completions.create(
            model=patient_model,
            messages=patient_chat + [{"role": "user", "content": therapist_reply}],
            temperature=0.9,
            max_tokens=100,
        )

        patient_reply = patient_resp.choices[0].message.content.strip()

        if looks_like_patient_drift(patient_reply):
            patient_reply = patient_llm.chat.completions.create(
                model=patient_model,
                messages=patient_chat + [{
                    "role": "system",
                    "content": (
                        "Rewrite strictly as the patient. "
                        "No advice. "
                        "Speak only from personal feelings and experience."
                    ),
                }],
                temperature=0.9,
                max_tokens=100,
            ).choices[0].message.content.strip()

        patient_chat.append({"role": "assistant", "content": patient_reply})

        turn_record = {
            "turn": turn_idx,
            "patient": {
                "role": "patient",
                "query": last_patient,
                "core_issue": core_issue,
                "schema": schema,
                "retrieval": rag,
            },
            "llm_response": {
                "role": "cbt_agent",
                "response": therapist_reply,
                "reasoning": reasoning,
            },
        }

        transcript.append(turn_record)
        last_patient = patient_reply

    driver.close()

    output = {
        "metadata": {
            "llm_response": therapist_model,
            "patient_model": patient_model,
            "mode": "cbt",
            "ablation_variant": ablation_variant,
            "turns": turns,
            "seed": seed,
            "core_issue": core_issue,
            "intervention_flags": {
                "use_schema": use_schema,
                "use_rag": use_rag,
                "use_protocol": use_protocol,
            },
        },
        "transcript": transcript,
    }

    Path(transcript_json).parent.mkdir(parents=True, exist_ok=True)
    output["transcript"] = classify_transcript(output["transcript"])
    Path(transcript_json).write_text(json.dumps(output, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--therapist_model", required=True)
    ap.add_argument("--patient_model", default="gpt-4o-mini")
    ap.add_argument(
        "--ablation_variant",
        required=True,
        choices=list(ABLATION_VARIANTS),
        help="Which component to remove: no_rag | no_schema | no_protocol",
    )
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--core_issue", required=True)
    ap.add_argument("--transcript_json", required=True)

    args = ap.parse_args()
    run_ablation_session(**vars(args))


if __name__ == "__main__":
    main()
