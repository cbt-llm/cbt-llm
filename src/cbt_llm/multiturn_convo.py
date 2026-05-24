"""
Generating multi-turn conversations between CBT agent and patient LLMs.

Run:
./run_experiment.sh [baseline|cbt] [gemma|mistral|deepseek|gpt]
"""
import math
import os
import re
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional


import requests
from openai import OpenAI
from neo4j import GraphDatabase

from cbt_llm.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, ROOT
from cbt_llm.retrieve_snomed import retrieve_snomed_matches
from user_schema.user_schema import extract_user_schema
from cbt_llm.pipelines.findings_pipeline import FindingsPipeline
from cbt_llm.cbt_mcot import mcot_therapist_reply, CBT_PROTOCOLS, generate_session_core_issue

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

These structures influence how you speak, but you MUST NOT name or reference them explicitly.

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
- If the therapist offers an interpretation, consider it emotionally

You are now the patient.
Respond naturally to the next message.
""".strip()

THERAPIST_BASELINE_PROMPT = """
Respond to the user's message.

Constraints:
- Respond naturally and thoughtfully.
- One paragraph or 2-4 sentences.
""".strip()

THERAPIST_CBT_MCOT_PROMPT = """
You are a skilled Cognitive Behavioral Therapy (CBT) therapist in an active therapy session with a real client.

Your role is to respond in a way that feels natural, warm, and therapeutically grounded — the way an experienced human therapist would in a real session.

You will receive:

1) The CLIENT MESSAGE (current turn)
2) COGNITIVE MODEL (hidden context): a structured decomposition of the client’s experience
3) CLINICAL CONTEXT (hidden context): clinically relevant statements about the client’s language
4) A CBT protocol describing the intervention strategy to apply this turn

Use the hidden context to deepen your reasoning, but NEVER reveal it directly.

━━━━━━━━━━━━━━━━━━━
COGNITIVE MODEL (DO NOT REVEAL)
━━━━━━━━━━━━━━━━━━━

In CBT, emotional distress emerges from how situations are interpreted:

  Triggers → Automatic Thoughts → Emotions → Behaviors

Triggers: situations or events that activate interpretation.
Automatic Thoughts: rapid meanings assigned to the situation — often reflecting
  deeper beliefs about self, others, or the world.
Emotions: arise from how the situation is interpreted.
Behaviors: actions or reactions that follow the emotional response.

Silently identify which element is most active in the client’s current message.
Use this to guide the depth, focus, and tone of your response.

━━━━━━━━━━━━━━━━━━━
CLINICAL CONTEXT (DO NOT REVEAL)
━━━━━━━━━━━━━━━━━━━

Treat the client’s message as the Premise.

The Clinical Context contains statements labeled:

ENTAILMENT — Likely relevant to the client’s experience. Use to inform interpretation.
NEUTRAL — Possibly relevant. Use with caution.
CONTRADICTION — Conflicts with what the client expressed. Do NOT use, even if tempting.

These are interpretive cues, not scripts. Use them to understand what may be driving
the client’s experience — never reference them directly in your response.

━━━━━━━━━━━━━━━━━━━
MULTIPLE CHAIN-OF-THOUGHT REASONING
━━━━━━━━━━━━━━━━━━━

Before generating the final response, internally simulate three candidate therapist
responses — one for each CBT intervention principle. These candidates are evaluated
and the best one is selected based on what the client most needs right now.

━━━━━━━━━━━━━━━━━━━
RESPONSE GUIDANCE
━━━━━━━━━━━━━━━━━━━

FIRST MESSAGE / INTRODUCTORY TURN:
Read the client’s actual message carefully before deciding how to open:
- If they open with a greeting ("Hello", "Hi", "How are you") → respond warmly
  and briefly, then invite them to share what brought them in. Do not ask a
  clinical question yet.
- If they immediately share a concern, feeling, or situation → engage
  therapeutically from the start. Do not waste the turn on pleasantries.

DISTRACTION / TANGENT TURNS:
If the client shifts to a topic unrelated to their core concern, do all three IN ORDER:
(1) ONE short clause (≤10 words) acknowledging what they just said.
(2) A natural bridge back that references something specific the client said
    earlier in this conversation — a feeling, a situation, a moment they described.
    e.g. "I want to come back to what you were saying about..."
    or "Going back to what you shared earlier about..."
    Do NOT say the core issue category label aloud. Ground the bridge in their
    own words and their own experience.
(3) Apply the selected CBT protocol technique to that specific thread.

ON-TOPIC TURNS:
Apply the selected CBT protocol technique directly. Let the schema and clinical
context inform the depth of your response.

GENERAL RULES:
- Do not diagnose.
- Do not explain CBT concepts or name protocols.
- Do not reveal hidden context.
- Do not repeat the client’s words verbatim.
- Avoid the same intervention style in consecutive turns.
- 2–4 sentences maximum.
- One focused question or reflection at a time — never multiple questions.
""".strip()

THERAPIST_CBT_PROMPT = """
You are a skilled Cognitive Behavioral Therapy (CBT) therapist in an active therapy session with a real client.

Your role is to respond in a way that feels natural, warm, and therapeutically grounded — the way an experienced human therapist would in a real session.

You will receive:

1) The CLIENT MESSAGE (current turn)
2) COGNITIVE MODEL (hidden context): a structured decomposition of the client’s experience
3) CLINICAL CONTEXT (hidden context): clinically relevant statements about the client’s language
4) A CBT protocol guideline describing intervention strategies

Use the hidden context to deepen your reasoning, but NEVER reveal it directly.

━━━━━━━━━━━━━━━━━━━
COGNITIVE MODEL (DO NOT REVEAL)
━━━━━━━━━━━━━━━━━━━

In CBT, emotional distress emerges from how situations are interpreted:

  Triggers → Automatic Thoughts → Emotions → Behaviors

Triggers: situations or events that activate interpretation.
Automatic Thoughts: rapid meanings assigned to the situation — often reflecting
  deeper beliefs about self, others, or the world.
Emotions: arise from how the situation is interpreted.
Behaviors: actions or reactions that follow the emotional response.

Silently identify which element is most active in the client’s current message.
Use this to guide the depth, focus, and tone of your response.

━━━━━━━━━━━━━━━━━━━
CLINICAL CONTEXT (DO NOT REVEAL)
━━━━━━━━━━━━━━━━━━━

Treat the client’s message as the Premise.

The Clinical Context contains statements labeled:

ENTAILMENT — Likely relevant to the client’s experience. Use to inform interpretation.
NEUTRAL — Possibly relevant. Use with caution.
CONTRADICTION — Conflicts with what the client expressed. Do NOT use, even if tempting.

These are interpretive cues, not scripts. Use them to understand what may be
driving the client’s experience — never reference them directly.

━━━━━━━━━━━━━━━━━━━
CBT PROTOCOL SELECTION (SELECT ONE)
━━━━━━━━━━━━━━━━━━━

Before selecting a protocol, consider what the client most needs right now:
to feel heard → validate_and_reflect
to examine a belief → socratic_questioning
to find a broader view → cognitive_restructuring

Step 1 — Identify the likely belief, assumption, or emotion driving the client’s message.
Step 2 — Choose the protocol that best matches the clinical need.
Step 3 — Apply the techniques associated with that protocol.

Do NOT reveal the protocol name or your reasoning process.

━━━━━━━━━━━━━━━━━━━
RESPONSE GUIDANCE
━━━━━━━━━━━━━━━━━━━

FIRST MESSAGE / INTRODUCTORY TURN:
Read the client’s actual message carefully before deciding how to open:
- If they open with a greeting → respond warmly and briefly, invite them to share.
- If they immediately share a concern → engage therapeutically from the start.

DISTRACTION / TANGENT TURNS:
If the client shifts to a topic unrelated to their core issue, do all three IN ORDER:
(1) ONE short clause (≤10 words) acknowledging what they just said.
(2) A direct bridge back — name the core issue explicitly:
    "I want to come back to [core issue] —"
    or "Let's return to [core issue] for a moment —"
(3) Apply the selected protocol technique focused on the core issue.
The client's confirmed core issue is provided in the SESSION CORE ISSUE block above.
Use it by name — do not be vague about what you are returning to.

ON-TOPIC TURNS:
Apply the selected protocol technique directly. Let the schema and clinical
context inform the depth of your response.

━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━

Before producing the final response, briefly reason about which retrieved clinical
concepts AND/OR user schema elements are relevant.

Your response MUST follow this exact format. Do not output anything else.
REASONING must ALWAYS appear.

REASONING:
"retrieved_concepts_used": ["concept1", "concept2"]

FINAL RESPONSE:
<therapist message>

Length: 2–4 sentences maximum.
One focused question or reflection at a time — never multiple questions.
Do not mention reasoning in the final response.
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

def load_client_transcript(path: str, index: int = 0) -> List[str]:
    """
    Load ordered client utterances from a transcript file.

    RealCBT .txt  — lines formatted as "client_turn_N: <text>"
    ESConv .json  — list of conversation dicts; `index` selects which one.
                    Each dict has a "dialog" list of {turn, content} entries.
    """
    p = Path(path)
    if p.suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        entry = data[index] if isinstance(data, list) else data
        dialog = entry.get("dialog", [])
        return [e["content"].strip() for e in dialog if e.get("content")]
    else:
        turns = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^client_turn_\d+:\s*(.*)", line)
            if m:
                turns.append(m.group(1).strip())
        return turns


def get_esconv_meta(path: str, index: int = 0) -> Optional[Dict[str, Any]]:
    """Return the ESConv conversation-level metadata for a given index, or None."""
    p = Path(path)
    if p.suffix != ".json":
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        entry = data[index] if isinstance(data, list) else data
        return {
            k: entry[k]
            for k in ("experience_type", "emotion_type", "problem_type", "situation")
            if k in entry
        }
    except Exception:
        return None


def get_core_issue(transcript_source: str, transcript_index: int = 0) -> Optional[str]:
    """
    Return the ground-truth core issue label from dataset metadata.
    RealCBT: looks up the file number in data/raw/realcbt_metadata.json.
    ESConv:  returns problem_type from the conversation entry.
    """
    if not transcript_source:
        return None
    p = Path(transcript_source)
    if p.suffix == ".json":
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            entry = data[transcript_index] if isinstance(data, list) else data
            return entry.get("problem_type")
        except Exception:
            return None
    else:
        m = re.search(r"file_(\d+)", p.stem)
        if not m:
            return None
        meta_path = ROOT / "data" / "raw" / "realcbt_metadata.json"
        if not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return meta.get(m.group(1))
        except Exception:
            return None


def load_distractions(path: str = None) -> List[Dict[str, Any]]:
    """Load distraction entries from distractions.json."""
    if path is None:
        path = str(ROOT / "data" / "raw" / "distractions.json")
    p = Path(path)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def build_combined_turns(
    client_turns: List[str],
    distraction_pool: List[Dict[str, Any]],
    ratio: float = 0.30,
) -> List[Dict[str, Any]]:
    """
    Build the full ordered sequence of turns for the session by INSERTING
    distraction turns after ~30% of real client turns (not replacing them).

    The resulting list is longer than client_turns by n_distractions entries.
    Each entry: {text, is_distraction, distraction_meta}

    Insertion positions are spread using stratified sampling across turns 0..N-2
    (never after the last real turn). The opening turn is always a real turn.
    """
    n = len(client_turns)
    base = [{"text": t, "is_distraction": False, "distraction_meta": None} for t in client_turns]

    if n < 2 or not distraction_pool:
        return base

    n_distractions = max(1, math.ceil(n * ratio))
    # Insert after any turn except the last (eligible: indices 0..n-2)
    eligible = list(range(0, n - 1))
    n_distractions = min(n_distractions, len(eligible), len(distraction_pool))

    # Stratified: split eligible range into buckets, pick one position per bucket
    segment = len(eligible) / n_distractions
    insert_after = set()
    for b in range(n_distractions):
        lo = int(b * segment)
        hi = max(lo + 1, int((b + 1) * segment))
        hi = min(hi, len(eligible))
        insert_after.add(eligible[random.randint(lo, hi - 1)])

    sampled = random.sample(distraction_pool, len(insert_after))
    insertion_map = {pos: entry for pos, entry in zip(sorted(insert_after), sampled)}

    combined = []
    for i, turn in enumerate(base):
        combined.append(turn)
        if i in insertion_map:
            d = insertion_map[i]
            combined.append({"text": d["text"], "is_distraction": True, "distraction_meta": d})

    return combined


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


def detect_anchor_decision(seed: str, patient_text: str, therapist_reply: str) -> Dict[str, Any]:
    """
    Heuristic post-hoc check for whether the therapist response anchored
    back to the seed topic. Logs CONTINUE vs BRIDGE behavior.
    """
    def content_tokens(s: str):
        s = s.lower()
        toks = re.findall(r"[a-z]{4,}", s)
        stop = {
            "that", "this", "with", "from", "have", "just", "like",
            "they", "them", "your", "want", "would", "could", "about",
            "feel", "feels", "really", "thing", "things", "going",
            "into", "much", "what", "when", "where", "there", "here",
            "than", "then", "been", "being", "some", "most", "more",
            "even", "ever", "also", "still", "again", "back", "make",
            "makes", "made", "know", "knew", "yeah",
        }
        return {t for t in toks if t not in stop}

    seed_toks = content_tokens(seed or "")
    patient_toks = content_tokens(patient_text or "")
    reply_toks = content_tokens(therapist_reply or "")

    seed_in_patient = sorted(seed_toks & patient_toks)
    seed_in_reply = sorted(seed_toks & reply_toks)

    on_seed_patient = len(seed_in_patient) >= 2
    if on_seed_patient:
        expected = "CONTINUE"
        bridged = None
    else:
        expected = "BRIDGE"
        bridged = len(seed_in_reply) >= 1

    return {
        "expected_mode": expected,
        "seed_tokens_in_patient": seed_in_patient,
        "seed_tokens_in_reply": seed_in_reply,
        "reply_anchored_to_seed": bridged,
    }


import re

def classify_reasoning_concepts(reasoning, rag, schema):
    concepts = reasoning.get("retrieved_concepts_used", [])
    labeled = []

    for concept in concepts:

        if isinstance(concept, dict):
            concept_text = concept.get("concept", "")
        else:
            concept_text = concept

        concept_clean = re.sub(r"\(.*?\)", "", concept_text).strip().lower()

        label = "unknown"

        for k in ["entailment", "neutral", "contradiction"]:
            for r in rag.get(k, []):
                r_clean = re.sub(r"\(.*?\)", "", r).strip().lower()
                if concept_clean in r_clean or r_clean in concept_clean:
                    label = k
                    break
            if label != "unknown":
                break

        if label == "unknown":
            for k in ["triggers", "automatic_thoughts", "emotions"]:
                for s in schema.get(k, []):
                    s_clean = s.strip().lower()
                    if concept_clean in s_clean or s_clean in concept_clean:
                        label = k
                        break
                if label != "unknown":
                    break

        labeled.append({
            "concept": concept_text,
            "label": label
        })

    reasoning["retrieved_concepts_used"] = labeled
    return reasoning

def classify_transcript(transcript):

    for turn in transcript:

        patient = turn.get("patient")
        llm_response = turn.get("llm_response")

        if not patient or not llm_response:
            continue

        rag = patient.get("retrieval")
        schema = patient.get("schema")

        reasoning = llm_response.get("reasoning")
        if reasoning:
            classify_reasoning_concepts(reasoning, rag, schema)

        candidates = llm_response.get("mcot_candidates", {})
        for _, data in candidates.items():
            r = data.get("reasoning")
            if r:
                classify_reasoning_concepts(r, rag, schema)

    return transcript

def parse_gpt_oss_output(raw_reply: str):

    reasoning = None
    response = raw_reply.strip()

    reasoning_matches = re.findall(
        r"REASONING:\s*(.*?)\s*(?:FINAL RESPONSE:|$)",
        raw_reply,
        re.DOTALL
    )

    response_matches = re.findall(
        r"FINAL RESPONSE:\s*(.*?)(?:REASONING:|$)",
        raw_reply,
        re.DOTALL
    )

    if response_matches:
        response = response_matches[-1].strip()

    if reasoning_matches:
        last_reasoning = reasoning_matches[-1].strip()

        try:
            if last_reasoning.startswith("{"):
                reasoning = json.loads(last_reasoning)
            else:
                reasoning = {"retrieved_concepts_used": []}

                items = re.findall(r'"([^"]+)"', last_reasoning)
                if items:
                    reasoning["retrieved_concepts_used"] = items

        except Exception:
            reasoning = None

    return response, reasoning

def run_session(
    therapist_model,
    patient_model,
    therapist_mode,
    turns,
    use_rag,
    use_schema,
    use_protocol,
    k,
    transcript_source=None,
    transcript_index=0,
    seed=None,        # kept for backward compat — turns now come from transcript
    core_issue=None,  # kept for backward compat — now inferred post-session
    distractions=None,
    transcript_json=None,
):

    if therapist_mode == "baseline":
        if use_rag or use_schema or use_protocol:
            raise ValueError(
                "Baseline mode must NOT use user schema, RAG, or CBT protocols."
            )
        use_rag = use_schema = use_protocol = False


    if therapist_mode in {"cbt", "cbt_mcot"}:
        if not (use_rag and use_schema and use_protocol):
            raise ValueError(
                "CBT mode MUST use user schema, RAG, and CBT protocols."
            )
        
    OLLAMA_MODELS = {"gpt-oss:20b", "mistral:7b", "gemma3:12b", "deepseek-r1:8b"}

    therapist_llm = (
        OllamaChat(therapist_model) if therapist_model in OLLAMA_MODELS
        else OpenAIChat(therapist_model)
    )

    # patient_llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # commented out — patient turns now come from transcript

    driver = GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
    )

    findings_pipeline = FindingsPipeline(driver, k=k)

    if therapist_mode == "baseline":
        base_prompt = THERAPIST_BASELINE_PROMPT
    elif therapist_mode == "cbt":
        base_prompt = THERAPIST_CBT_PROMPT
    elif therapist_mode == "cbt_mcot":
        base_prompt = THERAPIST_CBT_MCOT_PROMPT
    else:
        raise ValueError(f"Unknown therapist_mode: {therapist_mode}")

    # distraction_map was used with the LLM patient path; kept for reference only
    # distraction_map = {}
    # if distractions:
    #     raw = json.loads(distractions) if isinstance(distractions, str) else distractions
    #     distraction_map = {int(k): v for k, v in raw.items()}

    # Load client turns from the provided transcript file
    client_turns = load_client_transcript(transcript_source, transcript_index) if transcript_source else []
    esconv_meta = get_esconv_meta(transcript_source, transcript_index) if transcript_source else None
    # Load ground-truth core issue from dataset metadata (overrides any passed-in value)
    core_issue = get_core_issue(transcript_source, transcript_index) or core_issue

    # Build full turn sequence: real client turns with distraction turns inserted (~30%)
    distraction_pool = load_distractions()
    combined_turns = build_combined_turns(client_turns, distraction_pool, ratio=0.15) if client_turns else []

    # When a transcript is provided, run the full sequence regardless of --turns
    if combined_turns:
        max_turns = len(combined_turns)
    elif client_turns:
        max_turns = len(client_turns)
    else:
        max_turns = turns

    transcript = []
    # patient_chat = [{"role": "system", "content": PATIENT_SYSTEM.format()}]  # commented out — no LLM patient
    last_patient = combined_turns[0]["text"] if combined_turns else (seed or "")
    patient_history = []

    schema_trace = []

    for turn_idx in range(max_turns):

        # Load the current turn (real or distraction) from the combined sequence
        if combined_turns:
            current_turn = combined_turns[turn_idx]
            last_patient = current_turn["text"]
            is_distraction = current_turn["is_distraction"]
            distraction_meta = current_turn["distraction_meta"]
        else:
            is_distraction = False
            distraction_meta = None


        schema = safe_extract_schema(last_patient) if use_schema else None

        rag = None
        reasoning = None

        if use_rag:
            rag = findings_pipeline.get_findings(last_patient)

        if therapist_mode in {"cbt", "cbt_mcot"}:
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

        if core_issue:
            therapist_messages.append({
                "role": "system",
                "content": (
                    f"CLIENT'S CONFIRMED CORE ISSUE FOR THIS SESSION: \"{core_issue}\"\n\n"
                    f"This is the verified reason this client came to therapy — use it as your "
                    f"internal compass to understand what belongs in this session and what is a tangent.\n\n"
                    f"IMPORTANT: Do NOT say this label aloud in your response. "
                    f"When the client drifts, guide them back by referencing something specific "
                    f"they actually said earlier in the conversation — a feeling, a situation, "
                    f"a moment they described — not by naming the category.\n"
                    f"e.g. 'I want to come back to what you were saying about...' "
                    f"or 'Going back to what you shared earlier about...'"
                )
            })

        therapist_messages.append({
            "role": "user",
            "content": last_patient
        })

        if therapist_mode == "cbt_mcot":
            # Seed reminder removed — core concern is now inferred dynamically from patient_history
            # by build_anchor_block in cbt_mcot.py. The raw client turn is passed as-is.
            # if turn_idx > 0:
            #     augmented_patient = (
            #         f"[REMINDER — the user came in with this concern at the start of session: \"{seed}\". "
            #         f"Stay anchored to it.]\n\n"
            #         f"Patient's current message: {last_patient}"
            #     )
            # else:
            #     augmented_patient = last_patient
            augmented_patient = last_patient

            therapist_reply, protocol_used, candidates, reasoning = mcot_therapist_reply(
                therapist_llm,
                base_prompt,
                hidden_context,
                augmented_patient,
                rag,
                schema,
                seed=seed,
                turn_idx=turn_idx,
                patient_history=patient_history,
                core_issue=core_issue,
            )
            

        elif therapist_mode == "cbt":

            raw_reply = therapist_llm.chat(
                therapist_messages,
                temperature=0.15,
                num_predict=8000
            ).strip()

            therapist_reply = raw_reply
            print(raw_reply)

            if therapist_model == "gpt-oss:20b":
                print("gpt-oss")
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
                        print(raw_reply)
                        reasoning = None

        else:

            therapist_reply = therapist_llm.chat(
                therapist_messages,
                temperature=0.1,
                num_predict=8000,
            ).strip()

        if looks_like_therapist_leak(therapist_reply):
            rewritten = therapist_llm.chat(
                therapist_messages + [{
                    "role": "system",
                    "content": (
                        "Rewrite the response strictly following the rules. "
                        "Interpret meaning and advance insight."
                    )
                }],
                temperature=0.1,
                num_predict=8000
            ).strip()

            if rewritten:
                therapist_reply = rewritten

        if not therapist_reply:
            raise RuntimeError(
                f"Empty therapist response at turn {turn_idx} "
                f"from model {therapist_model}. Aborting run."
            )

        # Patient LLM generation commented out — turns come from real transcript
        # patient_resp = patient_llm.chat.completions.create(
        #     model=patient_model,
        #     messages=patient_chat + [{"role": "user", "content": therapist_reply}],
        #     temperature=0.9,
        #     max_tokens=100,
        # )
        # patient_reply = patient_resp.choices[0].message.content.strip()
        #
        # if turn_idx in distraction_map:
        #     patient_reply = distraction_map[turn_idx]
        # elif looks_like_patient_drift(patient_reply):
        #     patient_reply = patient_llm.chat.completions.create(
        #         model=patient_model,
        #         messages=patient_chat + [{
        #             "role": "system",
        #             "content": (
        #                 "Rewrite strictly as the patient. "
        #                 "No advice."
        #                 "Speak only from personal feelings and experience."
        #             )
        #         }],
        #         temperature=0.9,
        #         max_tokens=100,
        #     ).choices[0].message.content.strip()
        #
        # patient_chat.append({"role": "assistant", "content": patient_reply})

        therapist_block = {
            "role": "cbt_agent" if therapist_mode != "baseline" else "agent",
            "response": therapist_reply
        }

        if therapist_mode == "cbt":
            therapist_block["reasoning"] = reasoning

        if therapist_mode == "cbt_mcot":
            therapist_block["protocol_used"] = protocol_used
            therapist_block["mcot_candidates"] = candidates

            if turn_idx > 0:
                therapist_block["anchor_decision"] = detect_anchor_decision(
                    seed=seed,
                    patient_text=last_patient,
                    therapist_reply=therapist_reply,
                )


        turn_record = {
            "turn": turn_idx,
            "time_tag": f"T{turn_idx}",

            "patient": {
                "role": "patient",
                "query": last_patient,
                "schema": schema,
                "retrieval": rag,
                "distraction_injected": is_distraction,
                "distraction_meta": distraction_meta,
            },

            "llm_response": therapist_block
        }

        transcript.append(turn_record)

        # last_patient = patient_reply  # commented out — transcript drives turns
        patient_history.append(last_patient)  # still tracked for anchor reasoning in cbt_mcot

    driver.close()

    # Infer the client's core issue from the full session transcript
    inferred_core_issue = generate_session_core_issue(therapist_llm, transcript)

    output = {
        "metadata": {
            "llm_response": therapist_model,
            "patient_model": patient_model,
            "mode": therapist_mode,
            "turns": max_turns,
            "dataset": "esconv" if (transcript_source and transcript_source.endswith(".json")) else "realcbt",
            "transcript_source": transcript_source,
            "transcript_index": transcript_index if transcript_source and transcript_source.endswith(".json") else None,
            "esconv_meta": esconv_meta,
            "core_issue": core_issue,
            "inferred_core_issue": inferred_core_issue,
            # "seed": seed,         # commented out — turns now come from transcript
            "distraction_schedule": {
                str(i): {
                    "id": t["distraction_meta"]["id"],
                    "valence": t["distraction_meta"]["valence"],
                    "arousal": t["distraction_meta"]["arousal"],
                }
                for i, t in enumerate(combined_turns)
                if t["is_distraction"]
            } if combined_turns else None,
            "intervention_flags": {
                "use_schema": use_schema,
                "use_rag": use_rag,
                "use_protocol": use_protocol
            }
        },
        "transcript": transcript
    }

    Path(transcript_json).parent.mkdir(parents=True, exist_ok=True)

    output["transcript"] = classify_transcript(output["transcript"])

    Path(transcript_json).write_text(
        json.dumps(output, indent=2, ensure_ascii=False)
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--therapist_model", required=True)
    ap.add_argument("--patient_model", default="gpt-4o-mini")
    ap.add_argument("--therapist_mode", choices=["baseline", "cbt", "cbt_mcot"], required=True)
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--transcript_source", default=None,
                    help="Path to a RealCBT *_client.txt file or ESConv_client.json")
    ap.add_argument("--transcript_index", type=int, default=0,
                    help="Which conversation to use from an ESConv JSON file (0-based)")
    # ap.add_argument("--seed", required=True)  # commented out — turns now come from transcript
    ap.add_argument("--seed", default=None, help="(deprecated) seed sentence; replaced by --transcript_source")
    ap.add_argument("--distractions", help='JSON map of turn index to distraction text, e.g. \'{"3": "text", "5": "text"}\'')
    # ap.add_argument("--core_issue")  # commented out — now inferred post-session
    ap.add_argument("--core_issue", default=None, help="(deprecated) core issue label; now inferred post-session")
    ap.add_argument("--transcript_json", required=True)

    ap.add_argument("--use_rag", action="store_true")
    ap.add_argument("--use_schema", action="store_true")
    ap.add_argument("--use_protocol", action="store_true")

    args = ap.parse_args()

    run_session(**vars(args))


if __name__ == "__main__":
    main()