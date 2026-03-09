"""
Generating multi-turn conversations between CBT agent and patient LLMs.

Run:
./run_experiment.sh [baseline|cbt] [gemma|mistral|qwen|deepseek|gpt]
"""
import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from cbt_llm.cbt_tot import tot_therapist_reply, CBT_PROTOCOLS


import requests
from openai import OpenAI
from neo4j import GraphDatabase

from cbt_llm.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, ROOT
from cbt_llm.retrieve_snomed import retrieve_snomed_matches
from user_schema.user_schema import extract_user_schema
from cbt_llm.pipelines.findings_pipeline import FindingsPipeline
from pydantic import BaseModel


openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class CBTResponse(BaseModel):
    response: str
    clinical_context: list[str]
    principle_used: list[str]


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
            # response_format=CBTResponse
        )
        return r.choices[0].message.content.strip()


def load_cbt_protocols():
    path = ROOT / "references" / "cbt-protocols.json"

    if not path.exists():
        raise FileNotFoundError(f"Missing CBT protocol file: {path}")

    data = json.loads(path.read_text())

    return {
        p["name"]: p
        for p in data.get("cbt_protocols", [])
    }


PATIENT_SYSTEM = """
You are simulating a human patient in an ongoing cognitive behavioral therapy (CBT) session.

Respond as the patient would in a real therapy session.

Your responses should be internally guided by:
- your personal history
- your core and intermediate beliefs
- your triggers, automatic thoughts, emotions, and behaviors

These structures influence how you speak,
but you MUST NOT name or reference them explicitly.

- Speak in natural, everyday language
- Responses may include hesitation, uncertainty, or emotional shifts
- Gradually reveal deeper concerns over time
- Allow inconsistencies, ambivalence, and partial insight

- 1–3 sentences per response
- Do NOT give advice
- Do NOT explain therapy concepts
- Do NOT sound analytical or instructional
- Do NOT ask questions unless it feels emotionally natural
- Never mention user schemas, CBT, diagrams, or principles

- Respond appropriately to the conversation, not just the surface level question
- If the therapist offers an interpretation, consider it emotionally before intellectually

You are now the patient.
Respond naturally to the next message.

""".strip()

THERAPIST_BASELINE_PROMPT = """
You are a thoughtful agent responding to a patient.

Guidelines:
- Respond naturally and thoughtfully.
- Do not diagnose or label.
- If the patient mentions self-harm or imminent danger, encourage immediate local help.

Constraints:
- One paragraph or 2–4 sentences.
""".strip()

# THERAPIST_CBT_TOT_PROMPT = """
# You are a Cognitive Behavioral Therapy (CBT) agent participating in a live therapy conversation with a patient.

# Your goal is to help the patient explore and understand patterns in their thoughts, emotions, and behaviors.

# You will receive:

# 1) The PATIENT MESSAGE (Premise)
# 2) COGNITIVE MODEL (hidden context): a structured decomposition of the patient's experience
# 3) CLINICAL CONTEXT (hidden context): clinical statements related to the patient's language
# 4) A CBT protocol guideline describing three intervention strategies

# Use the hidden context to guide your reasoning, but NEVER reveal it.

# ━━━━━━━━━━━━━━━━━━━
# COGNITIVE MODEL (DO NOT REVEAL)
# ━━━━━━━━━━━━━━━━━━━

# In Cognitive Behavioral Therapy, emotional distress is understood as emerging from patterns in how situations are interpreted.

# Experiences can often be conceptualized as:

# Triggers → Automatic Thoughts → Emotions → Behaviors

# Triggers are situations or events that activate interpretation.

# Automatic thoughts are rapid interpretations or meanings assigned to the situation. These thoughts often reflect deeper beliefs, assumptions, expectations, or rules about the self, other people, or the world.

# Emotions arise from how the situation is interpreted.

# Behaviors are the actions or reactions that follow the emotional response.

# Use this structure silently to interpret what may be driving the patient’s experience.
# Focus on the element most relevant to the patient’s message.

# ━━━━━━━━━━━━━━━━━━━
# CLINICAL CONTEXT (DO NOT REVEAL)
# ━━━━━━━━━━━━━━━━━━━

# Treat the patient’s message as the Premise.

# The Clinical Context contains statements that may relate to the patient’s underlying psychological state.

# These refer to an observable or reported clinical state or psychological pattern to help represent clinically meaningful patterns in patient language or behavior.

# Each statement is labeled as:

# ENTAILMENT — Likely relevant to the patient’s experience and useful for interpretation.

# NEUTRAL — Possibly relevant but uncertain.

# CONTRADICTION — Conflicts with the patient’s experience and should NOT be used.

# Use the clinical context only as cues when interpreting the patient’s experience.

# ━━━━━━━━━━━━━━━━━━━
# TREE OF THOUGHT REASONING
# ━━━━━━━━━━━━━━━━━━━

# Before generating the final response, internally simulate three candidate therapist responses — one for each CBT intervention principle.

# For each candidate:

# 1. Infer the belief or assumption that may be driving the patient's experience.
# 2. Apply the intervention principle using its techniques.
# 3. Consider how the patient might respond to that intervention.

# The three intervention candidates are:

# A) validate_and_reflect  
# Focus on emotional acknowledgment and alignment.

# B) socratic_questioning  
# Ask questions that help the patient examine the belief.

# C) cognitive_restructuring  
# Introduce a gentle alternative interpretation of the belief.

# Evaluate which candidate would most effectively move the conversation forward therapeutically.

# Select the strongest candidate.

# Do NOT reveal the reasoning or the other candidates.

# ━━━━━━━━━━━━━━━━━━━
# CONSTRAINTS
# ━━━━━━━━━━━━━━━━━━━

# - Do not diagnose.
# - Do not explain therapy concepts.
# - Do not mention CBT principles or reasoning.
# - Do not reveal hidden context.
# - Do not repeat the patient's statement verbatim.
# - Avoid repeating the same intervention style across consecutive turns.


# Your response should:
# - Sound like a natural therapist response
# - The therapist should focus on the belief or assumption underlying the patient's experience rather than only reflecting emotions.

# Length:
# 2–4 sentences maximum.

# Output only the therapist's response to the patient.
# """.strip()

THERAPIST_CBT_PROMPT= """
You are a Cognitive Behavioral Therapy (CBT) agent participating in a live therapy conversation with a patient.

Your goal is to help the patient explore and understand patterns in their thoughts, emotions, and behaviors.

You will receive:

1) The PATIENT MESSAGE (Premise)
2) COGNITIVE MODEL (hidden context): a structured decomposition of the patient's experience
3) CLINICAL CONTEXT (hidden context): clinical statements related to the patient's language
4) A CBT protocol guideline describing three intervention strategies

Use the hidden context to guide your reasoning, but NEVER reveal it.

━━━━━━━━━━━━━━━━━━━
COGNITIVE MODEL (DO NOT REVEAL)
━━━━━━━━━━━━━━━━━━━

In Cognitive Behavioral Therapy, emotional distress is understood as emerging from patterns in how situations are interpreted.

Experiences can often be conceptualized as:

Triggers → Automatic Thoughts → Emotions → Behaviors

Triggers are situations or events that activate interpretation.

Automatic thoughts are rapid interpretations or meanings assigned to the situation. These thoughts often reflect deeper beliefs, assumptions, expectations, or rules about the self, other people, or the world.

Emotions arise from how the situation is interpreted.

Behaviors are the actions or reactions that follow the emotional response.

Use this structure silently to interpret what may be driving the patient’s experience.
Focus on the element most relevant to the patient’s message.

━━━━━━━━━━━━━━━━━━━
CLINICAL CONTEXT (DO NOT REVEAL)
━━━━━━━━━━━━━━━━━━━

Treat the patient’s message as the Premise.

The Clinical Context contains statements that may relate to the patient’s underlying psychological state.

These refer to an observable or reported clinical state or psychological pattern to help represent clinically meaningful patterns in patient language or behavior.

Each statement is labeled as:

ENTAILMENT — Likely relevant to the patient’s experience and useful for interpretation.

NEUTRAL — Possibly relevant but uncertain.

CONTRADICTION — Conflicts with the patient’s experience and should NOT be used.

Use the clinical context only as cues when interpreting the patient’s experience.

━━━━━━━━━━━━━━━━━━━
CBT PROTOCOL SELECTION (SELECT ONE)
━━━━━━━━━━━━━━━━━━━

Follow this reasoning process internally:

Step 1 — Identify the likely belief or assumption driving the patient's statement.

Step 2 — Determine which intervention principle would best help the patient examine or shift this belief.

Choose ONE:

1) validate_and_reflect
2) socratic_questioning
3) cognitive_restructuring

Step 3 — Apply the techniques associated with the selected protocol to construct the response.

Do NOT reveal the reasoning process or the protocol name.

Guidelines:

validate_and_reflect  
Use when the patient primarily needs emotional acknowledgment or when trust and safety should be strengthened.

socratic_questioning  
Use when the patient's belief or assumption should be explored through curiosity and gentle inquiry.

cognitive_restructuring  
Use when the patient's interpretation appears rigid, overly negative, or limiting, and a new perspective may help.

Use the chosen protocol to guide how you respond.

Do NOT name the protocol in your response.

━━━━━━━━━━━━━━━━━━━
CONSTRAINTS
━━━━━━━━━━━━━━━━━━━

- Do not diagnose.
- Do not explain therapy concepts.
- Do not mention CBT principles or reasoning.
- Do not reveal hidden context.
- Do not repeat the patient's statement verbatim.
- Avoid repeating the same intervention style across consecutive turns.


Your response should:
- Sound like a natural therapist response
- The therapist should focus on the belief or assumption underlying the patient's experience rather than only reflecting emotions.

Length:
2–4 sentences maximum.

Output only the therapist's response to the patient.
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
    if use_protocol and CBT_PROTOCOLS:
        blocks.append("[CBT_PROTOCOLS]\n" + json.dumps(CBT_PROTOCOLS, indent=2))
    if schema:
        blocks.append("[USER SCHEMA]\n" + json.dumps(schema))
    if rag:
        blocks.append("[CLINICAL CONTEXT]\n" + json.dumps(rag, indent=2))
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


    if therapist_mode in {"cbt", "cbt_tot"}:
        if not (use_rag and use_schema and use_protocol):
            raise ValueError(
                "CBT mode MUST use user schema, RAG, and CBT protocols."
            )
        
    OLLAMA_MODELS = {"gpt-oss:20b", "mistral:7b", "gemma3:12b", "deepseek-r1:8b"}

    therapist_llm = (
        OllamaChat(therapist_model) if therapist_model in OLLAMA_MODELS
        else OpenAIChat(therapist_model)
    )

    patient_llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    driver = GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
    )

    findings_pipeline = FindingsPipeline(driver, k=k)

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

        rag = None

        if use_rag:
            rag = findings_pipeline.get_findings(last_patient)

        if therapist_mode in {"cbt", "cbt_tot"}:
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

        if therapist_mode == "cbt_tot":

            therapist_reply, protocol_used, candidates = tot_therapist_reply(
                therapist_llm,
                base_prompt,
                hidden_context,
                last_patient
            )

        elif therapist_mode == "cbt":

            therapist_reply = therapist_llm.chat(
                therapist_messages,
                temperature=0.15,
                num_predict=2600,
            ).strip()

        else:

            therapist_reply = therapist_llm.chat(
                therapist_messages,
                temperature=0.15,
                num_predict=2600,
            ).strip()

        if looks_like_therapist_leak(therapist_reply):
            rewritten = therapist_llm.chat(
                therapist_messages + [{
                    "role": "system",
                    "content": (
                        "Rewrite the response strictly following the rules. "
                        # "Do NOT paraphrase. "
                        # "Do NOT begin with phrases like 'it sounds like' or 'it seems like'. "
                        "Interpret meaning and advance insight."
                    )
                }],
                temperature=0.1,
                num_predict=2500,
                top_p=0.7,
            ).strip()

            if rewritten:
                therapist_reply = rewritten

        if not therapist_reply:
            raise RuntimeError(
                f"Empty therapist response at turn {turn_idx} "
                f"from model {therapist_model}. Aborting run."
            )

        entry = {
            "role": "therapist",
            "content": therapist_reply
        }

        if therapist_mode == "cbt_tot":
            entry["protocol_used"] = protocol_used
            entry["tot_candidates"] = candidates

        transcript.append(entry)

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

    if therapist_mode in {"cbt", "cbt_tot"}:
        output["user_schema_trace"] = schema_trace

    Path(transcript_json).parent.mkdir(parents=True, exist_ok=True)
    Path(transcript_json).write_text(
        json.dumps(output, indent=2, ensure_ascii=False)
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--therapist_model", required=True)
    ap.add_argument("--patient_model", default="gpt-4o-mini")
    ap.add_argument("--therapist_mode", choices=["baseline", "cbt", "cbt_tot"], required=True)
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