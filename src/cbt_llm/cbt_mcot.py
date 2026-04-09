"""
Runs CBT Multiple Chain-of-Thought reasoning.

Steps:
1) generate candidate response per CBT protocol
2) evaluate candidates
3) select best response
"""

import json
import re
from pathlib import Path
from cbt_llm.config import ROOT


def load_cbt_protocols():
    path = ROOT / "references" / "cbt-protocols.json"

    if not path.exists():
        raise FileNotFoundError(f"Missing CBT protocol file: {path}")

    data = json.loads(path.read_text())

    return {
        p["name"]: p
        for p in data.get("cbt_protocols", [])
    }

CBT_PROTOCOLS = load_cbt_protocols()

def parse_reasoning_block(after_reasoning: str):
    """Try multiple strategies to extract retrieved_concepts_used."""
    
    # strategy 1: valid JSON object
    match = re.search(r"\{[\s\S]*?\}", after_reasoning)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    
    # strategy 2: bare key-value without braces
    # handles: "retrieved_concepts_used": ["a", "b"]
    match = re.search(
        r'"retrieved_concepts_used"\s*:\s*(\[[\s\S]*?\])',
        after_reasoning
    )
    if match:
        try:
            concepts = json.loads(match.group(1))
            return {"retrieved_concepts_used": concepts}
        except Exception:
            pass

    # strategy 3: just pull quoted strings as concept list
    items = re.findall(r'"([^"]+)"', after_reasoning.split("FINAL RESPONSE:")[0])
    if items:
        # filter out the key name itself
        items = [i for i in items if i != "retrieved_concepts_used"]
        return {"retrieved_concepts_used": items}

    return None


def generate_candidate(llm, base_prompt, hidden_context, patient_text, protocol):

    principle_prompt = f"""
    CBT intervention protocol

    Name:
    {protocol["name"]}

    Purpose:
    {protocol["purpose"]}

    Key functions:
    {chr(10).join("- " + k for k in protocol.get("key_functions", []))}

    Techniques:
    {chr(10).join("- " + t for t in protocol.get("techniques", []))}

    Generate ONE therapist response following this protocol.

    Before producing the final response, briefly reason about
    which retrieved clinical concepts AND/OR user schema elements
    may be relevant to the response.

    Return your answer EXACTLY in this structure:

    Return your answer EXACTLY in this structure:

    REASONING:
    {{
      "retrieved_concepts_used": ["...", "..."]
    }}

    FINAL RESPONSE:
    <therapist message>

    Rules:
    - REASONING must ALWAYS appear.
    - FINAL RESPONSE must contain only the therapist message.
    - Natural therapist tone
    - Do not explain CBT
    - MAXIMUM 2 sentences.
    - Do not overload the patient with too much information or too many questions. One question at a time.
    - Output only the final response
    - BAD (too long, too many questions): "What evidence do you have that you'll fail? How did you feel last time? What would a friend say? What small step could you take?"
    """


    messages = [
        {"role": "system", "content": base_prompt}
    ]

    if hidden_context:
        messages.append({
            "role": "system",
            "content": "IMPORTANT CONTEXT (DO NOT REVEAL):\n" + hidden_context
        })

    messages.append({
        "role": "system",
        "content": principle_prompt
    })

    messages.append({
        "role": "user",
        "content": patient_text
    })

    raw = llm.chat(
        messages,
        temperature=0.1,
        num_predict=16000
    ).strip()

    print("\n========== RAW MCOT OUTPUT ==========")
    print(raw)
    print("=====================================\n")

    reasoning = None
    response = raw

    if "REASONING:" in raw:
        try:
            after_reasoning = raw.split("REASONING:", 1)[1]

            reasoning = parse_reasoning_block(after_reasoning)

            # extract response
            if "FINAL RESPONSE:" in after_reasoning:
                response = after_reasoning.split("FINAL RESPONSE:", 1)[1].strip()
            elif reasoning:
                # fallback: everything after the reasoning block
                match = re.search(r'\][\s\S]*', after_reasoning)
                if match:
                    response = match.group(0).lstrip(']\n ').strip()

        except Exception:
            reasoning = None
            response = raw


    return {
        "response": response,
        "reasoning": reasoning
    }

def evaluate_candidates(llm, patient_text, candidates, protocols):

    judge_prompt = """
    You are evaluating therapist responses in a Cognitive Behavioral Therapy (CBT) conversation.

    Each response was generated using a different CBT intervention protocol.

    Your task: select the response that would most effectively move the therapy forward.

    Step 1 — Read the patient message carefully. Identify:
    - Is the patient primarily expressing emotion and needing to feel heard?
    - Is the patient articulating a specific belief or thought that could be examined?
    - Is the patient stuck in a rigid interpretation that a broader view could help shift?

    Step 2 — Match the clinical need to the protocol:
    - validate_and_reflect: patient needs emotional acknowledgment BEFORE they can examine beliefs. Use early in rapport-building or when distress is high and the patient is not yet ready to reflect.
    - socratic_questioning: patient has articulated a specific belief or assumption that can be tested with evidence. Use when the patient is stable enough to reflect and there is a clear belief to examine.
    - alternative_perspective: patient is locked into one way of seeing a situation and a broader view would reduce distress. Use when the patient already feels heard and is repeating the same interpretation without movement.

    Step 3 — Consider: has the patient already received validation in recent turns? If so, continuing to validate may not move the conversation forward. Prefer socratic_questioning or alternative_perspective if the patient is ready.

    Step 4 — Output your reasoning in one sentence, then output the number of the best response.

    Format your response EXACTLY as:
    REASONING: <one sentence>
    ANSWER: <number>
    """.strip()

    candidate_blocks = []

    for i, (response, protocol) in enumerate(zip(candidates, protocols), start=1):

        block = f"""
        {i}. Protocol: {protocol["name"]}

        Purpose:
        {protocol["purpose"]}

        Response:
        {response}
        """.strip()

        candidate_blocks.append(block)

    numbered_candidates = "\n\n".join(candidate_blocks)

    messages = [
        {"role": "system", "content": judge_prompt},
        {
            "role": "user",
            "content": (
                f"Patient message:\n{patient_text}\n\n"
                f"Candidate responses:\n\n{numbered_candidates}"
            )
        },
    ]

    result = llm.chat(
        messages,
        temperature=0.0,
        top_p=0.9,
        num_predict=400,
    ).strip()

    print(f"\n=== JUDGE REASONING ===\n{result}\n======================\n")

    try:
        # extract ANSWER: N if present, otherwise try raw int
        match = re.search(r'ANSWER:\s*(\d+)', result)
        if match:
            idx = int(match.group(1)) - 1
        else:
            idx = int(result.strip()) - 1
        return max(0, min(idx, len(candidates) - 1))
    except Exception:
        return 0


def mcot_therapist_reply(
    llm,
    base_prompt,
    hidden_context,
    patient_text,
    rag,
    schema
):

    candidate_list = []
    candidate_map = {}

    protocols = list(CBT_PROTOCOLS.values())

    for protocol in protocols:

        candidate = generate_candidate(
            llm,
            base_prompt,
            hidden_context,
            patient_text,
            protocol
        )

        candidate_list.append(candidate["response"])

        candidate_map[protocol["name"]] = {
            "response": candidate["response"],
            "reasoning": candidate["reasoning"]
        }

    best_idx = evaluate_candidates(
        llm,
        patient_text,
        candidate_list,
        protocols
    )

    best_protocol = protocols[best_idx]["name"]

    best_response = candidate_map[best_protocol]["response"]
    best_reasoning = candidate_map[best_protocol]["reasoning"]

    return best_response, best_protocol, candidate_map, best_reasoning
