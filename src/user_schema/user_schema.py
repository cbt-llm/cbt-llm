import json
from typing import Dict, Any
import os
import re
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# user_schema.py — add one function, keep extract_user_schema as-is

EVOLVING_SCHEMA_PROMPT = """
You are a clinical-reasoning assistant maintaining a cumulative CBT cognitive model
across a therapy session.

You will receive:
1. The CURRENT cognitive model extracted so far
2. A NEW patient utterance from the latest turn

Your task: update the cognitive model by merging new information into the existing one.

Field definitions:
- Triggers: external situations or contexts that precede distress (not thoughts or beliefs).
- Automatic thoughts: immediate internal interpretations or predictions.
- Emotions: momentary affective states.
- Behaviors: observable actions or avoidance responses.

The cognitive model describes how people’s thoughts and perceptions influence their lives.
Often, distress can distort people’s perceptions, and that, in turn, can lead to unhealthy
emotions and behaviors. This helps to identify and evaluate user's “automatic
thoughts” and shift their thinking to be healthier. 

Rules:
- Do NOT duplicate existing entries
- Do NOT add an entry if it expresses the same meaning as an existing one
- NEVER remove existing entries, only update or keep
- Keep phrases concise and specific
- Do NOT infer diagnoses or clinical labels
- Always return valid JSON

Current cognitive model:
{current_schema}

New patient utterance:
\"\"\"{input_text}\"\"\"

Return updated JSON ONLY with fields: triggers, automatic_thoughts, emotions, behaviors.
"""

def update_user_schema(current_schema: Dict[str, Any], new_text: str) -> Dict[str, Any]:
    """
    Merges a new patient utterance into the existing cumulative schema.
    """
    prompt = EVOLVING_SCHEMA_PROMPT.format(
        current_schema=json.dumps(current_schema, indent=2),
        input_text=new_text
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = response.choices[0].message.content.strip()
    print("\n=== RAW SCHEMA RESPONSE ===")
    print(repr(raw))  # repr to see hidden chars, fences, etc.
    print("===========================\n")


    data = safe_json_parse(response.choices[0].message.content.strip())

    for f in ["triggers", "automatic_thoughts", "emotions", "behaviors"]:
        if f not in data or data[f] is None:
            data[f] = []

    return data


# # per-turn context
# USER_SCHEMA_PROMPT = """
# You are a clinical-reasoning assistant trained to extract structured CBT cognitive model 
# from user text without diagnosing, labeling pathology, or adding interpretation.

# Field definitions:
# - Triggers: external situations or contexts that precede distress (not thoughts or beliefs).
# - Automatic thoughts: immediate internal interpretations or predictions.
# - Emotions: momentary affective states.
# - Behaviors: observable actions or avoidance responses.

# The cognitive model describes how people’s thoughts and perceptions influence their lives.
# Often, distress can distort people’s perceptions, and that, in turn, can lead to unhealthy
# emotions and behaviors. This helps to identify and evaluate user's “automatic
# thoughts” and shift their thinking to be healthier. 

# Extract ONLY the information present in the text.

# Return a JSON object with the following fields:

# {{
#   "triggers": [list of short phrases],
#   "automatic_thoughts": [list of short phrases],
#   "emotions": [list of emotion words only],
#   "behaviors": [list of short phrases],
# }}

# Guidelines:
# - Keep phrases concise and specific.
# - Do NOT repeat the concept across fields.
# - If a phrase describes an internal mental event, place it under automatic_thoughts, not triggers.
# - Do NOT infer diagnoses or clinical labels.
# - Emotions must be simple affective terms (e.g., "fear", "shame", "anxiety").
# - If something is missing, return an empty list.
# - Always return valid JSON.

# User text:
# \"\"\"{input_text}\"\"\"

# Return JSON ONLY.
# """


def safe_json_parse(text: str) -> Dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "triggers": [],
            "automatic_thoughts": [],
            "emotions": [],
            "behaviors": [],
        }
    
def extract_user_schema(input_text: str) -> Dict[str, Any]:
    """Thin per-turn extraction — used only for schema_trace audit."""
    prompt = f"""Extract CBT cognitive model fields from this text as JSON only.
Fields: triggers, automatic_thoughts, emotions, behaviors.
Return empty lists if absent. No labels, no diagnoses.

Text: \"\"\"{input_text}\"\"\"

JSON only."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return safe_json_parse(response.choices[0].message.content.strip())


if __name__ == "__main__":
    text = "I feel calm most of the time, but sometimes when small things pile up, it feels like nothing is going the way it should, and I end up blowing up."
    out = extract_user_schema(text)
    print(json.dumps(out, indent=2))
