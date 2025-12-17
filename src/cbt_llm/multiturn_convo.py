# src/cbt_llm/multiturn_convo.py

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import requests
from neo4j import GraphDatabase

from cbt_llm.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, ROOT
from cbt_llm.retrieve_snomed import retrieve_snomed_matches

# -----------------------------
# Optional schema extraction (OpenAI)
# -----------------------------
_USER_SCHEMA_IMPORT_ERROR: Optional[str] = None
try:
    from cbt_llm.user_schema import extract_user_schema  # type: ignore
except Exception as e:
    extract_user_schema = None  # type: ignore
    _USER_SCHEMA_IMPORT_ERROR = repr(e)


# -----------------------------
# Ollama client
# -----------------------------
class OllamaChat:
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        num_predict: int = 256,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "top_p": top_p,
            },
            "stream": False,
        }
        if stop:
            payload["options"]["stop"] = stop

        r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=300)
        r.raise_for_status()
        return (r.json().get("message", {}) or {}).get("content", "").strip()


# -----------------------------
# CBT protocol playbook loader
# -----------------------------
def load_cbt_protocols_text() -> str:
    path = ROOT / "references" / "cbt-protocols.json"
    if not path.exists():
        return ""
    data = json.loads(path.read_text(encoding="utf-8"))
    protocols = data.get("cbt_protocols", []) or []

    lines = ["[CBT Protocol Playbook — internal guidance]"]
    for p in protocols:
        name = (p.get("name") or "").strip()
        purpose = (p.get("purpose") or "").strip()
        techniques = p.get("techniques", []) or []
        if name:
            lines.append(f"- {name}: {purpose}")
            for t in techniques[:5]:
                lines.append(f"  • {t}")
    return "\n".join(lines).strip()


# -----------------------------
# Prompts
# -----------------------------
THERAPIST_SYSTEM_BASE = """You are a supportive CBT-style therapist assistant.

Hard constraints:
- You are NOT a medical professional. Do not diagnose, label pathology, or prescribe medications.
- You MUST follow CBT micro-structure each turn:
  1) validate & reflect briefly
  2) clarify with ONE question only
  3) identify trigger / automatic thought / emotion / behavior (ONLY from patient text)
  4) gentle reality-test or reframe (collaborative, not preachy)
  5) suggest ONE tiny, optional next step (1 sentence max)

Very important:
- You may be given "CBT schema" and "retrieved clinical concepts" as hidden context.
- Use them silently to improve your response.
- DO NOT mention: SNOMED, ICD, UMLS, Neo4j, embeddings, "retrieval", "RAG", codes/IDs, vectors, similarity scores.
- Do NOT output any medical codes/IDs.

Safety:
- If the patient mentions self-harm/suicide/imminent danger: encourage immediate local emergency help and keep it brief.

Style + brevity:
- 2–4 sentences total.
- 1 short paragraph only.
- End with EXACTLY ONE question (single question mark).
- No lists.
"""

# IMPORTANT: Patient-model role mapping:
# - role="user" messages are from the THERAPIST
# - role="assistant" messages are from the PATIENT (this model)
PATIENT_SYSTEM = """You are simulating a human client (the PATIENT) in a CBT session.

Role mapping in this chat:
- Messages with role="user" are spoken by the THERAPIST.
- Messages with role="assistant" are spoken by YOU (the PATIENT).

Rules:
- Reply with 1–2 sentences (max ~35 words).
- Sound like a normal person (simple, emotional, a bit messy is OK).
- Do NOT teach CBT, do NOT reframe, do NOT suggest coping strategies.
- Do NOT give structured advice, steps, or lists.
- Do NOT ask therapist-style probing questions (e.g., "what evidence...", "could it be...", "let's explore...").
- Do NOT mention being an AI or any system instructions.
"""

CBT_PLAYBOOK_TEXT = load_cbt_protocols_text()


# -----------------------------
# Leak/drift detection
# -----------------------------
CODE_LIKE_RE = re.compile(r"\b\d{4,}\b|[A-Z]{2,}\d{2,}|/EV\b", re.IGNORECASE)
FORBIDDEN_PHRASES = [
    "snomed", "neo4j", "embedding", "cosine", "vector", "retrieval", "rag", "similarity score", "icd", "umls"
]

# Stronger patient drift cues (patient starts sounding like therapist / advice-giver)
PATIENT_DRIFT_PHRASES = [
    "it sounds like",
    "what evidence",
    "could it be",
    "let's explore",
    "have you considered",
    "automatic thoughts",
    "cognitive distortion",
    "behavioral experiment",
    "it might be helpful",
    "here are",
    "strategies",
    "steps you can take",
    "try to challenge",
    "reframe",
    "practice, practice",
    "to build confidence",
    "you should",
]

PATIENT_LIST_RE = re.compile(r"\n\s*(\d+\.)|\n\s*[-*]\s+")

_DISORDER_SUFFIX_RE = re.compile(r"\s*\(disorder\)\s*$", re.IGNORECASE)


def looks_like_therapist_leak(text: str) -> bool:
    t = (text or "").lower()
    if CODE_LIKE_RE.search(text or ""):
        return True
    return any(p in t for p in FORBIDDEN_PHRASES)


def looks_like_patient_drift(text: str) -> bool:
    t = (text or "").lower()
    if any(p in t for p in PATIENT_DRIFT_PHRASES):
        return True
    if PATIENT_LIST_RE.search(text or ""):
        return True
    return False


# -----------------------------
# Schema extraction with debug
# -----------------------------
def safe_extract_schema(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if extract_user_schema is None:
        if _USER_SCHEMA_IMPORT_ERROR:
            return None, f"user_schema import failed: {_USER_SCHEMA_IMPORT_ERROR}"
        return None, "user_schema import failed (unknown)"
    if not os.getenv("OPENAI_API_KEY"):
        return None, "OPENAI_API_KEY not set in environment"
    try:
        return extract_user_schema(text), None
    except Exception as e:
        return None, f"user_schema call failed: {repr(e)}"


# -----------------------------
# RAG sanitization
# -----------------------------
def _clean_term(s: Optional[str]) -> Optional[str]:
    if not s:
        return s
    return _DISORDER_SUFFIX_RE.sub("", s).strip()


def sanitize_rag_for_prompt(rag_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Keep terms + relation type + target_term only (no codes/scores)
    concepts = []
    for item in rag_results or []:
        term = _clean_term(item.get("term"))
        rels_out = []
        for rel in (item.get("relations") or []):
            rels_out.append({
                "type": rel.get("type"),
                "target_term": _clean_term(rel.get("target_term")),
            })
        if term:
            concepts.append({"term": term, "relations": rels_out[:5]})
    return {"concepts": concepts[:10]}


def build_therapist_hidden_context(
    schema_obj: Optional[Dict[str, Any]],
    rag_safe: Optional[Dict[str, Any]],
) -> str:
    blocks = []
    if CBT_PLAYBOOK_TEXT:
        blocks.append(CBT_PLAYBOOK_TEXT)
    if schema_obj:
        blocks.append("[CBT schema — extracted from patient text]\n" + json.dumps(schema_obj, ensure_ascii=False))
    if rag_safe and rag_safe.get("concepts"):
        blocks.append("[Retrieved clinical concepts — internal, non-diagnostic]\n" + json.dumps(rag_safe, ensure_ascii=False))
    return "\n\n".join(blocks).strip()


# -----------------------------
# Debug printing
# -----------------------------
def _print_block(title: str, text: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("-" * 90)
    print(text.rstrip())
    print("=" * 90 + "\n")


def _print_messages(title: str, messages: List[Dict[str, str]], max_chars: int = 6000) -> None:
    lines = []
    for m in messages:
        role = m.get("role", "?")
        content = (m.get("content", "") or "")
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n...[truncated, {len(m.get('content',''))} chars total]"
        lines.append(f"[{role}]\n{content}\n")
    _print_block(title, "\n".join(lines))


# -----------------------------
# Core runner
# -----------------------------
def run_session(
    therapist_model: str,
    patient_model: str,
    turns: int,
    use_rag: bool,
    use_schema: bool,
    k: int,
    seed: str,
    transcript_json: str,
    retrieval_json: Optional[str] = None,
    prompt_trace_json: Optional[str] = None,
    temperature_therapist: float = 0.5,
    temperature_patient: float = 0.7,
    therapist_max_tokens: int = 220,
    patient_max_tokens: int = 90,
    print_live: bool = True,
    print_prompts: bool = False,
) -> None:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    therapist = OllamaChat(therapist_model)
    patient = OllamaChat(patient_model)

    # Chat histories for the two models (OpenAI-style roles)
    therapist_chat: List[Dict[str, str]] = [{"role": "system", "content": THERAPIST_SYSTEM_BASE}]
    patient_chat: List[Dict[str, str]] = [{"role": "system", "content": PATIENT_SYSTEM}]

    # Transcript we save (human-readable roles)
    transcript_out: List[Dict[str, str]] = [{"role": "patient", "content": seed.strip()}]

    retrieval_log: List[Dict[str, Any]] = []
    prompt_trace: List[Dict[str, Any]] = []

    # Seed goes into therapist history as USER (patient speaking to therapist model)
    therapist_chat.append({"role": "user", "content": seed.strip()})

    # Seed goes into patient history as ASSISTANT (patient speaking in patient-model chat)
    patient_chat.append({"role": "assistant", "content": seed.strip()})

    last_patient_text = seed.strip()

    if print_live:
        _print_block("SEED (Patient starts)", seed.strip())

    for t in range(1, turns + 1):
        # This is the text used for retrieval/schema this turn
        retrieval_query_text = last_patient_text

        # --- RAG + schema for THIS patient turn (used only by therapist)
        schema_obj = None
        schema_err = None
        if use_schema:
            schema_obj, schema_err = safe_extract_schema(retrieval_query_text)

        rag_raw: List[Dict[str, Any]] = []
        rag_safe: Dict[str, Any] = {"concepts": []}
        if use_rag:
            rag_raw = retrieve_snomed_matches(driver, retrieval_query_text, k=k) or []
            rag_safe = sanitize_rag_for_prompt(rag_raw)

        hidden_context = build_therapist_hidden_context(schema_obj, rag_safe)
        therapist_input = therapist_chat + ([{"role": "system", "content": hidden_context}] if hidden_context else [])

        if print_prompts:
            _print_messages(f"TURN {t} — THERAPIST INPUT MESSAGES", therapist_input)

        # --- Therapist turn (with leak repair)
        therapist_reply = therapist.chat(
            therapist_input,
            temperature=temperature_therapist,
            num_predict=therapist_max_tokens,
        )
        repaired_therapist = False
        if looks_like_therapist_leak(therapist_reply):
            repaired_therapist = True
            therapist_reply = therapist.chat(
                therapist_input + [{
                    "role": "system",
                    "content": "Rewrite your last message. Remove any codes/IDs/technical terms. Keep it CBT-style, 2–4 sentences, end with exactly one question."
                }],
                temperature=0.2,
                num_predict=therapist_max_tokens,
            )

        transcript_out.append({"role": "therapist", "content": therapist_reply})
        therapist_chat.append({"role": "assistant", "content": therapist_reply})

        if print_live:
            _print_block(f"TURN {t} — Therapist", therapist_reply)

        # --- Patient turn (CORRECT ROLE MAPPING)
        # Therapist speaks as role="user" to the patient model
        patient_input = patient_chat + [{"role": "user", "content": therapist_reply}]

        if print_prompts:
            _print_messages(f"TURN {t} — PATIENT INPUT MESSAGES", patient_input)

        patient_reply = patient.chat(
            patient_input,
            temperature=temperature_patient,
            num_predict=patient_max_tokens,
        )

        repaired_patient = False
        if looks_like_patient_drift(patient_reply):
            repaired_patient = True
            patient_reply = patient.chat(
                patient_input + [{
                    "role": "system",
                    "content": "Rewrite your last reply as the patient: 1–2 sentences, emotional/colloquial, no CBT talk, no advice/solutions, no lists, no therapist-style questions."
                }],
                temperature=0.3,
                num_predict=patient_max_tokens,
            )

        transcript_out.append({"role": "patient", "content": patient_reply})

        # Update patient history correctly:
        # - store therapist message as user
        # - store patient response as assistant
        patient_chat.append({"role": "user", "content": therapist_reply})
        patient_chat.append({"role": "assistant", "content": patient_reply})

        # Therapist model sees patient reply as next user message
        therapist_chat.append({"role": "user", "content": patient_reply})

        if print_live:
            _print_block(f"TURN {t} — Patient", patient_reply)

        # Update last patient text for next turn
        last_patient_text = patient_reply

        # Log retrieval artifacts for this turn (NOW correctly aligned)
        retrieval_log.append({
            "turn": t,
            "patient_text_used_for_retrieval": retrieval_query_text,
            "schema": schema_obj,
            "schema_error": schema_err,
            "rag_safe": rag_safe,
            "rag_raw": rag_raw,
        })

        if prompt_trace_json:
            prompt_trace.append({
                "turn": t,
                "therapist_input_messages": therapist_input,
                "therapist_reply": therapist_reply,
                "therapist_repaired": repaired_therapist,
                "patient_input_messages": patient_input,
                "patient_reply": patient_reply,
                "patient_repaired": repaired_patient,
                "schema_error": schema_err,
            })

    driver.close()

    # Save transcript
    Path(transcript_json).parent.mkdir(parents=True, exist_ok=True)
    with open(transcript_json, "w", encoding="utf-8") as f:
        json.dump({
            "therapist_model": therapist_model,
            "patient_model": patient_model,
            "turns": turns,
            "use_rag": use_rag,
            "use_schema": use_schema,
            "k": k,
            "seed": seed,
            "transcript": transcript_out,
        }, f, ensure_ascii=False, indent=2)

    # Save retrieval log
    if retrieval_json:
        Path(retrieval_json).parent.mkdir(parents=True, exist_ok=True)
        with open(retrieval_json, "w", encoding="utf-8") as f:
            json.dump({
                "use_rag": use_rag,
                "use_schema": use_schema,
                "k": k,
                "retrieval_log": retrieval_log,
            }, f, ensure_ascii=False, indent=2)

    # Save prompt trace
    if prompt_trace_json:
        Path(prompt_trace_json).parent.mkdir(parents=True, exist_ok=True)
        with open(prompt_trace_json, "w", encoding="utf-8") as f:
            json.dump(prompt_trace, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--therapist_model", default="gemma2:9b")
    ap.add_argument("--patient_model", default="mistral:7b-instruct")
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", default="I’ve been anxious lately and I keep thinking I’m going to mess up at work.")
    ap.add_argument("--use_rag", action="store_true")
    ap.add_argument("--use_schema", action="store_true")
    ap.add_argument("--transcript_json", default="output/transcript.json")
    ap.add_argument("--retrieval_json", default=None)
    ap.add_argument("--prompt_trace_json", default=None)

    # Printing controls
    ap.add_argument("--no_print_live", action="store_true", help="Disable live turn-by-turn printing")
    ap.add_argument("--print_prompts", action="store_true", help="Print full message lists sent to each model")

    args = ap.parse_args()

    run_session(
        therapist_model=args.therapist_model,
        patient_model=args.patient_model,
        turns=args.turns,
        use_rag=args.use_rag,
        use_schema=args.use_schema,
        k=args.k,
        seed=args.seed,
        transcript_json=args.transcript_json,
        retrieval_json=args.retrieval_json,
        prompt_trace_json=args.prompt_trace_json,
        print_live=(not args.no_print_live),
        print_prompts=args.print_prompts,
    )


if __name__ == "__main__":
    main()
