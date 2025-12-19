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

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        num_predict: int,
        top_p: float = 0.9,
    ) -> str:
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
        return (r.json().get("message") or {}).get("content", "").strip()

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

THERAPIST_BASELINE_PROMPT = """
You are responding to the patient as an assistant.

Guidelines:
- Respond naturally and empathetically.
- Do not diagnose or label disorders.
- If the patient mentions self-harm or imminent danger, encourage immediate local help.

Style:
- 2-4 sentences.
- One paragraph.
""".strip()

# THERAPIST_CBT_PROMPT_gemma = """
# You are a CBT-style therapist in a live conversation.

# You are given hidden contextual information derived from:
# - a structured schema extracted from the patient’s recent language
# - background psychological concepts used only for interpretation

# ABSOLUTE REQUIREMENT:
# Every response MUST clearly use the structured schema.
# If no schema element is reflected in your wording, the response is incorrect.

# AVAILABLE CBT MOVES (choose ONE per turn):
# A. Clarifying Question
# B. Gentle Reframe
# C. Pattern Highlighting

# Move definitions:
# A. Clarifying Question – helps the patient notice a specific thought or trigger  
# B. Gentle Reframe – offers a tentative alternative interpretation of a thought  
# C. Pattern Highlighting – connects the current statement to a recurring pattern across turns  

# RULES FOR MOVE SELECTION:
# - Choose EXACTLY ONE move per turn
# - Do NOT repeat the same move used in the previous turn unless clearly necessary
# - The move must directly operate on the selected schema element

# HOW TO USE THE SCHEMA:
# - Select EXACTLY ONE schema element (trigger OR automatic thought OR emotion OR behavior)
# - Paraphrase it in plain language
# - Make it clearly recognizable in your response

# BACKGROUND CONCEPTS:
# - Use only for silent interpretation
# - Translate into everyday experiences
# - Never mention diagnoses or clinical terms

# STRICT FORMAT RULES:
# - One paragraph
# - 2–3 sentences
# - End with exactly ONE open-ended question
# - No advice, coping strategies, reassurance, or explanations

# If the response could apply to another patient, it is incorrect.
# If the CBT move is unclear, it is incorrect.

# """.strip()

THERAPIST_CBT_PROMPT = """
You are a CBT-style therapist in a live conversation. 
Your goal is to help the patient gain insight into their thoughts and feelings and improve their understanding.

You are given hidden contextual information derived from:
- a structured schema extracted from the patient's recent language
- background psychological concepts used only for interpretation

ABSOLUTE REQUIREMENT:
Every response MUST clearly use the structured schema.
If no schema element is reflected in your wording, the response is incorrect.

HOW TO USE THE SCHEMA:
- Select EXACTLY ONE schema element from ONE bucket:
  (trigger OR automatic thought OR emotion OR behavior)
- Paraphrase it in plain, everyday language
- Make it clearly recognizable in your response

BACKGROUND CONCEPTS (if present):
- Use them ONLY as silent support for interpretation
- Translate them into everyday experiences (e.g., “mental fog”, “feeling stuck”)
- NEVER mention diagnoses or clinical terms

STRICT FORMAT RULES (must follow):
- Exactly 2 or 3 sentences
- Ask EXACTLY ONE open-ended question
- NO lists, NO advice, NO coping strategies
- NO psychoeducation
- NO therapy explanations or instructions

CONTENT REQUIREMENTS (in order):
1. Reflect ONE specific schema element (by paraphrase) or
2. Reflect ONE specific patient thought or pattern
3. Offer ONE alternative interpretation or discrepancy
4. End with ONE open-ended question about that thought

If your response could apply to another patient, it is incorrect.
If the schema is not clearly visible in the wording, it is incorrect.

""".strip()

PATIENT_SYSTEM = """
You are simulating a human patient.

Rules (must follow):
- 1–2 sentences only.
- Speak only about your own feelings, thoughts, or experiences.
- Engage with the therapist's prompts naturally and address your issues.
- Do NOT give advice, reassurance, validation, or guidance.
- Do NOT ask questions.
- Informal, emotional, sometimes messy language.
- Never sound like a therapist.
- Do not mention being an AI.
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
        blocks.append("[SCHEMA]\n" + json.dumps(schema))
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
    therapist_model: str,
    patient_model: str,
    therapist_mode: str,
    turns: int,
    use_rag: bool,
    use_schema: bool,
    use_protocol: bool,
    k: int,
    seed: str,
    transcript_json: str,
    print_prompts: bool,
):
    therapist_llm = OllamaChat(therapist_model)

    from openai import OpenAI
    openai_client = OpenAI()

    driver = GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
    )

    base_therapist_prompt = (
        THERAPIST_BASELINE_PROMPT
        if therapist_mode == "baseline"
        else THERAPIST_CBT_PROMPT
    )

    therapist_chat: List[Dict[str, str]] = []
    patient_chat: List[Dict[str, str]] = [
        {"role": "system", "content": PATIENT_SYSTEM}
    ]

    transcript = [{"role": "patient", "content": seed}]
    cbt_context_log = []

    patient_chat.append({"role": "assistant", "content": seed})
    last_patient = seed

    for t in range(1, turns + 1):
        schema = safe_extract_schema(last_patient) if use_schema else None

        rag_safe = None
        if use_rag:
            rag_raw = retrieve_snomed_matches(driver, last_patient, k=k) or []
            rag_safe = sanitize_rag(rag_raw)

        hidden_context = build_hidden_context(schema, rag_safe, use_protocol)

        therapist_system_prompt = base_therapist_prompt
        if hidden_context:
            therapist_system_prompt += (
                "\n\nIMPORTANT CONTEXT (MUST BE USED):\n"
                + hidden_context
            )

        therapist_chat = [
            {"role": "system", "content": therapist_system_prompt},
            {"role": "user", "content": last_patient},
        ]

        if print_prompts:
            print("\n" + "=" * 80)
            print(f"TURN {t} — THERAPIST INPUT")
            for m in therapist_chat:
                print(f"[{m['role']}]\n{m['content']}\n")

        # -------- Therapist generation --------
        therapist_reply = therapist_llm.chat(
            therapist_chat,
            temperature=0.15,
            num_predict=140,
            top_p=0.7,
        )

        # Retry if therapist violates constraints
        if looks_like_therapist_leak(therapist_reply):
            therapist_reply = therapist_llm.chat(
                therapist_chat + [{
                    "role": "system",
                    "content": (
                        "Rewrite strictly following the rules. "
                        "No lists. No advice. One question only."
                    )
                }],
                temperature=0.1,
                num_predict=120,
            )

        transcript.append({"role": "therapist", "content": therapist_reply})

        # -------- Grounding audit --------
        if therapist_mode == "cbt":
            cbt_context_log.append({
                "turn": t,
                "patient_text": last_patient,
                "schema": schema,
                "rag": rag_safe,
                "grounding_audit": audit_grounding(
                    therapist_reply, schema, rag_safe
                ),
            })

        patient_messages = patient_chat + [
            {"role": "user", "content": therapist_reply}
        ]

        patient_resp = openai_client.chat.completions.create(
            model=patient_model,
            messages=patient_messages,
            temperature=0.9,
            max_tokens=80,
        )

        patient_reply = patient_resp.choices[0].message.content.strip()

        # Retry if patient drifts into therapist mode
        if looks_like_patient_drift(patient_reply):
            patient_resp = openai_client.chat.completions.create(
                model=patient_model,
                messages=patient_messages + [{
                    "role": "system",
                    "content": (
                        "Rewrite as the patient. "
                        "Feelings only. No advice. "
                        "No validation. 1–2 sentences."
                    )
                }],
                temperature=0.9,
                max_tokens=80,
            )
            patient_reply = patient_resp.choices[0].message.content.strip()

        transcript.append({"role": "patient", "content": patient_reply})

        patient_chat.append({"role": "assistant", "content": patient_reply})
        last_patient = patient_reply

    driver.close()

    output = {
        "therapist_model": therapist_model,
        "patient_model": patient_model,
        "therapist_mode": therapist_mode,
        "turns": turns,
        "seed": seed,
        "transcript": transcript,
    }

    if therapist_mode == "cbt":
        output["cbt_context"] = cbt_context_log

    Path(transcript_json).parent.mkdir(parents=True, exist_ok=True)
    Path(transcript_json).write_text(
        json.dumps(output, indent=2, ensure_ascii=False)
    )


def main():
    ap = argparse.ArgumentParser()

    # ap.add_argument("--therapist_model", default="mistral:7b-instruct")
    ap.add_argument("--therapist_model", default="gemma2:9b")

    ap.add_argument("--patient_model", default="gpt-4o-mini")

    ap.add_argument("--therapist_mode", choices=["baseline", "cbt"], required=True)
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", default="I keep overthinking everything at work.")
    ap.add_argument("--transcript_json", default="output/debug_transcript.json")
    ap.add_argument("--use_rag", action="store_true")
    ap.add_argument("--use_schema", action="store_true")
    ap.add_argument("--use_protocol", action="store_true")
    ap.add_argument("--print_prompts", action="store_true")

    args = ap.parse_args()

    run_session(
        therapist_model=args.therapist_model,
        patient_model=args.patient_model,
        therapist_mode=args.therapist_mode,
        turns=args.turns,
        use_rag=args.use_rag,
        use_schema=args.use_schema,
        use_protocol=args.use_protocol,
        k=args.k,
        seed=args.seed,
        transcript_json=args.transcript_json,
        print_prompts=args.print_prompts,
    )

if __name__ == "__main__":
    main()