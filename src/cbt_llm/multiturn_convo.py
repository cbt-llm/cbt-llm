"""
Generating conversations between therapist and patient LLMs.

Run:
./run_experiment.sh [baseline|cbt] [gemma|mistral|qwen|deepseek|gpt]
"""
import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
from openai import OpenAI
from neo4j import GraphDatabase

from cbt_llm.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, ROOT
from cbt_llm.retrieve_snomed import retrieve_snomed_matches


openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

OPENAI_PATIENT_MODEL = "gpt-4o-mini"
class OllamaChat:
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def chat(self, messages, temperature, num_predict, top_p=0.9) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "top_p": top_p,
            },
            "stream": False,
        }
        r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=300)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()


class OpenAIChat:
    def __init__(self, model: str):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def chat(self, messages, temperature, num_predict, top_p=0.9) -> str:
        r = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=num_predict,
            top_p=top_p,
        )
        return r.choices[0].message.content.strip()


def load_cbt_protocols_text() -> str:
    path = ROOT / "references" / "cbt-protocols.json"
    if not path.exists():
        return ""

    data = json.loads(path.read_text())
    lines = ["[CBT Protocol Playbook — internal]"]

    for p in data.get("cbt_protocols", []):
        lines.append(f"- {p['name']}: {p.get('purpose','')}")
        for kf in (p.get("key_functions") or [])[:3]:
            lines.append(f"  ◦ {kf}")

    return "\n".join(lines)

CBT_PLAYBOOK_TEXT = load_cbt_protocols_text()

PATIENT_SYSTEM = """
You are simulating a human patient in an ongoing cognitive behavioral therapy (CBT) session.

Respond as the patient would in a real therapy session.

Your responses should be internally guided by:
- your personal history
- your core and intermediate beliefs
- your automatic thoughts, emotions, and behaviors

These structures influence how you speak,
but you MUST NOT name or reference them explicitly.

- Speak in natural, everyday language
- Responses may include hesitation, uncertainty, or emotional shifts
- Gradually reveal deeper concerns over time
- Allow inconsistencies, ambivalence, and partial insight

- 1–5 sentences per response
- Do NOT give advice
- Do NOT explain therapy concepts
- Do NOT sound analytical or instructional
- Do NOT ask questions unless it feels emotionally natural
- Never mention schemas, beliefs, CBT, or diagrams

- React to the therapist’s response, not just the surface question
- If the therapist offers an interpretation, consider it emotionally before intellectually

You are now the patient.
Respond naturally to the therapist’s next message.

""".strip()

THERAPIST_BASELINE_PROMPT = """
You are responding to the patient as a therapist.

Guidelines:
- Respond naturally and empathetically.
- Do not diagnose or label disorders.
- If the patient mentions self-harm or imminent danger, encourage immediate local help.

Style:
- 2-4 sentences.
- One paragraph maximum.
""".strip()

THERAPIST_CBT_PROMPT = """
You are a Cognitive Behavior Therapist in a live Cognitive Behavior Therapy (CBT) conversation.

You are given HIDDEN CONTEXT that includes:
- a CBT protocol playbook describing validated intervention strategies
- a structured user schema (triggers, automatic thoughts, emotions, behaviors)
- retrieved clinical concepts relevant to the user’s language

You MUST use this context on EVERY turn.

━━━━━━━━━━━━━━━━━━━
NON-NEGOTIABLE RULE
━━━━━━━━━━━━━━━━━━━

You MUST NOT mirror, paraphrase, or summarize the patient’s message.

If your response could be mistaken for a reflection of what the patient just said,
the response is INVALID.

Your task is to INTERPRET meaning and advance insight,
not to echo content.

━━━━━━━━━━━━━━━━━━━
OPENING STYLE CONSTRAINT
━━━━━━━━━━━━━━━━━━━


You may start with ONE of these forward-moving openings:
- a tentative hypothesis about an assumption
- a pattern label
- a discrepancy
- a reframed meaning
- a precision check that does NOT restate 

Or reflective starters such as:
- "It sounds like..."
- "It seems like..."
- "I hear you..."
- "What I'm hearing is..."
- "You're feeling..."
- "You are feeling..."


━━━━━━━━━━━━━━━━━━━
INTERNAL CLINICAL REASONING (SILENT)
━━━━━━━━━━━━━━━━━━━

Before writing your response, do ALL of the following internally:

1. Select the MOST RELEVANT schema element
   (trigger OR automatic thought OR emotion OR behavior)

2. Identify the implicit assumption or cognitive distortion beneath it
   (e.g., rules, expectations, self-judgments, meanings, conditional beliefs)

3. Select the SINGLE most appropriate CBT protocol:
   - validate_and_reflect → when emotional safety or alignment is primary
   - socratic_questioning → when assumptions need to be examined
   - cognitive_reframing → when interpretations are rigid or limiting

4. Use the most relevant retrieved concepts ONLY to sharpen interpretation
   (they are support signals; do NOT name or quote them)

━━━━━━━━━━━━━━━━━━━
HOW TO USE CONTEXT
━━━━━━━━━━━━━━━━━━━

- Refer to schema elements INDIRECTLY, never verbatim
- Treat retrieved concepts as patterns, never diagnoses
- Translate technical ideas into lived experience
- Focus on what the experience IMPLIES, not what it IS
- Never name CBT techniques, schemas, diagnoses, or distortions

━━━━━━━━━━━━━━━━━━━
RESPONSE CONSTRAINTS
━━━━━━━━━━━━━━━━━━━

- ONE paragraph
- 2–4 sentences total
- NO advice
- NO psychoeducation

━━━━━━━━━━━━━━━━━━━
CONTENT REQUIREMENTS
━━━━━━━━━━━━━━━━━━━

Your response MUST:
1. Indirectly reference ONE schema element
2. Make an implicit assumption or distortion visible
3. Apply the selected CBT protocol clearly
4. Introduce a NEW interpretation, pattern, or perspective

Question use:
- Ask EXACTLY ONE open-ended question ONLY if the chosen protocol requires exploration
  (e.g., socratic_questioning).
- If using validate_and_reflect or cognitive_reframing, a question is OPTIONAL.

If the response restates the patient’s experience, it is wrong.
If the response could apply to many people, it is wrong.
If the CBT protocol is not clearly applied, it is wrong.
""".strip()



CODE_LIKE_RE = re.compile(r"\b\d{4,}\b|[A-Z]{2,}\d{2,}")
FORBIDDEN_PHRASES = ["snomed", "neo4j", "embedding", "rag", "vector"]

PATIENT_DRIFT = [
    "you’re right", "that makes sense", "you should", "try to",
    "it might help", "remember that"
]

def looks_like_patient_drift(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in PATIENT_DRIFT)

def looks_like_therapist_leak(text: str) -> bool:
    return any(x in text.lower() for x in ["snomed", "rag", "embedding"])


try:
    from cbt_llm.user_schema import extract_user_schema
except Exception:
    extract_user_schema = None

def safe_extract_schema(text: str) -> Optional[Dict[str, Any]]:
    if not extract_user_schema:
        return None
    try:
        return extract_user_schema(text)
    except Exception:
        return None

def sanitize_rag(raw):
    concepts = []
    for r in raw or []:
        if r.get("term"):
            concepts.append({"term": r["term"]})
    return {"concepts": concepts[:5]}

def build_hidden_context(schema, rag, use_protocol):
    blocks = []
    if use_protocol and CBT_PLAYBOOK_TEXT:
        blocks.append(CBT_PLAYBOOK_TEXT)
    if schema:
        blocks.append("[USER SCHEMA]\n" + json.dumps(schema))
    if rag:
        blocks.append("[RAG]\n" + json.dumps(rag))
    return "\n\n".join(blocks)

def audit_grounding(reply, schema, rag):
    t = reply.lower()
    return {
        "schema_used": bool(schema and any(x.lower() in t for v in schema.values() for x in v if isinstance(x,str))),
        "rag_used": bool(rag and any(c["term"].split()[0].lower() in t for c in rag.get("concepts", [])))
    }


def run_session(
    therapist_model,
    patient_model,
    therapist_mode,
    turns,
    use_rag,
    use_schema,
    use_protocol,
    k,
    seed,
    transcript_json,
):

    if therapist_mode == "baseline":
        if use_rag or use_schema or use_protocol:
            raise ValueError(
                "Baseline mode must NOT use user schema, RAG, or CBT protocols."
            )
        use_rag = use_schema = use_protocol = False

    if therapist_mode == "cbt":
        if not (use_rag and use_schema and use_protocol):
            raise ValueError(
                "CBT mode MUST use user schema, RAG, and CBT protocols."
            )
        
    therapist_llm = (
        OpenAIChat(therapist_model)
        if therapist_model.startswith("gpt")
        else OllamaChat(therapist_model)
    )

    patient_llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    driver = GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
    )

    base_prompt = (
        THERAPIST_BASELINE_PROMPT
        if therapist_mode == "baseline"
        else THERAPIST_CBT_PROMPT
    )

    transcript = [{"role": "patient", "content": seed}]
    patient_chat = [{"role": "system", "content": PATIENT_SYSTEM}]
    last_patient = seed

    schema_trace = []  # store schema per turn for CBT transcripts

    for turn_idx in range(turns):

        schema = safe_extract_schema(last_patient) if use_schema else None
        rag = (
            sanitize_rag(retrieve_snomed_matches(driver, last_patient, k=k))
            if use_rag else None
        )

        if therapist_mode == "cbt":
            schema_trace.append({
                "turn": turn_idx,
                "patient_text": last_patient,
                "schema": schema,
                "retrieval": rag,
            })

        hidden_context = build_hidden_context(schema, rag, use_protocol)

        therapist_messages = [
            {"role": "system", "content": base_prompt}
        ]

        if hidden_context:
            therapist_messages.append({
                "role": "system",
                "content": "IMPORTANT CONTEXT (DO NOT REVEAL):\n" + hidden_context
            })

        therapist_messages.append({
            "role": "user",
            "content": last_patient
        })

        therapist_reply = therapist_llm.chat(
            therapist_messages,
            temperature=0.15,
            num_predict=140,
            top_p=0.7,
        ).strip()

        if looks_like_therapist_leak(therapist_reply):
            rewritten = therapist_llm.chat(
                therapist_messages + [{
                    "role": "system",
                    "content": (
                        "Rewrite the response strictly following the rules. "
                        "Do NOT paraphrase. "
                        "Do NOT begin with phrases like 'it sounds like' or 'it seems like'. "
                        "Interpret meaning and advance insight."
                    )
                }],
                temperature=0.1,
                num_predict=140,
                top_p=0.7,
            ).strip()

            if rewritten:
                therapist_reply = rewritten

        if not therapist_reply:
            raise RuntimeError(
                f"Empty therapist response at turn {turn_idx} "
                f"from model {therapist_model}. Aborting run."
            )

        transcript.append({
            "role": "therapist",
            "content": therapist_reply
        })

        patient_resp = patient_llm.chat.completions.create(
            model=patient_model,
            messages=patient_chat + [{
                "role": "user",
                "content": therapist_reply
            }],
            temperature=0.9,
            max_tokens=80,
        )

        patient_reply = patient_resp.choices[0].message.content.strip()

        # Retry if patient drifts into therapist mode
        if looks_like_patient_drift(patient_reply):
            patient_reply = patient_llm.chat.completions.create(
                model=patient_model,
                messages=patient_chat + [{
                    "role": "system",
                    "content": (
                        "Rewrite strictly as the patient. "
                        "No advice. No validation. "
                        "Speak only from personal feelings and experience."
                    )
                }],
                temperature=0.9,
                max_tokens=80,
            ).choices[0].message.content.strip()

        transcript.append({
            "role": "patient",
            "content": patient_reply
        })

        patient_chat.append({
            "role": "assistant",
            "content": patient_reply
        })

        last_patient = patient_reply

    driver.close()

    output = {
        "therapist_model": therapist_model,
        "patient_model": patient_model,
        "therapist_mode": therapist_mode,
        "intervention_flags": {
            "use_schema": use_schema,
            "use_rag": use_rag,
            "use_protocol": use_protocol,
        },
        "turns": turns,
        "seed": seed,
        "transcript": transcript,
    }

    if therapist_mode == "cbt":
        output["user_schema_trace"] = schema_trace

    Path(transcript_json).parent.mkdir(parents=True, exist_ok=True)
    Path(transcript_json).write_text(
        json.dumps(output, indent=2, ensure_ascii=False)
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--therapist_model", required=True)
    ap.add_argument("--patient_model", default="gpt-4o-mini")
    ap.add_argument("--therapist_mode", choices=["baseline", "cbt"], required=True)
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--transcript_json", required=True)

    ap.add_argument("--use_rag", action="store_true")
    ap.add_argument("--use_schema", action="store_true")
    ap.add_argument("--use_protocol", action="store_true")

    args = ap.parse_args()

    run_session(**vars(args))


if __name__ == "__main__":
    main()