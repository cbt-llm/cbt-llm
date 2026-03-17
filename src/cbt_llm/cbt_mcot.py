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

    REASONING:
    "retrieved_concepts_used": ["...", "..."]


    FINAL RESPONSE:
    <therapist message>

    Rules:
    - REASONING must ALWAYS appear.
    - FINAL RESPONSE must contain only the therapist message.
    - natural therapist tone
    - do not explain CBT
    - 2–4 sentences
    - output only the response
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

            # extract reasoning JSON
            match = re.search(r"\{[\s\S]*?\}", after_reasoning)

            if match:
                reasoning_text = match.group(0)
                reasoning = json.loads(reasoning_text)

                remainder = after_reasoning[match.end():].strip()

                if "FINAL RESPONSE:" in remainder:
                    response = remainder.split("FINAL RESPONSE:",1)[1].strip()
                else:
                    # fallback: everything after reasoning JSON is response
                    response = remainder.strip()

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

    Select the response that BEST applies its protocol and would most effectively
    move the therapy conversation forward.

    Evaluation criteria:
    - correct application of the CBT intervention
    - empathy and emotional attunement
    - exploration or reframing of beliefs
    - therapeutic usefulness
    - natural conversational flow

    Return ONLY the number of the best response.
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

    try:
        idx = int(result) - 1
        return max(0, min(idx, len(candidates) - 1))
    except Exception:
        return 0


# def classify_reasoning_concepts(reasoning, rag, schema):

#     concepts = reasoning.get("retrieved_concepts_used", [])

#     print("\n[DEBUG] Retrieved concepts from reasoning:")
#     for c in concepts:
#         print(" -", c)
#     print()

#     return reasoning



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
