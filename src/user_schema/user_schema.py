import json
from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# # ever growing, cyclical context?
# USER_CAT_PROMPT = """
# Using the following workflow, extract: 

# - Varying Appraisal checks
# - changes in other emotional components
# - emotion
# - new appraisals
# - changes in emotion components
# - emotional experience changes

# - view 1 or 2
# """


# per-turn context
USER_SCHEMA_PROMPT = """
You are a clinical-reasoning assistant trained to extract structured CBT cognitive model 
from user text without diagnosing, labeling pathology, or adding interpretation.

Field definitions:
- Triggers: external situations or contexts that precede distress (not thoughts or beliefs).
- Automatic thoughts: immediate internal interpretations or predictions.
- Emotions: momentary affective states.
- Behaviors: observable actions or avoidance responses.

The cognitive model describes how people’s thoughts and perceptions influence their lives.
Often, distress can distort people’s perceptions, and that, in turn, can lead to unhealthy
emotions and behaviors. This helps to identify and evaluate user's “automatic
thoughts” and shift their thinking to be healthier. 

Extract ONLY the information present in the text.

Return a JSON object with the following fields:

{{
  "triggers": [list of short phrases],
  "automatic_thoughts": [list of short phrases],
  "emotions": [list of emotion words only],
  "behaviors": [list of short phrases],
}}

Guidelines:
- Keep phrases concise and specific.
- Do NOT repeat the concept across fields.
- If a phrase describes an internal mental event, place it under automatic_thoughts, not triggers.
- Do NOT infer diagnoses or clinical labels.
- Emotions must be simple affective terms (e.g., "fear", "shame", "anxiety").
- If something is missing, return an empty list.
- Always return valid JSON.

User text:
\"\"\"{input_text}\"\"\"

Return JSON ONLY.
"""


def safe_json_parse(text: str) -> Dict[str, Any]:
    """
    Attempts to parse JSON returned by the model.
    Falls back to empty structure if invalid.
    """
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
    """
    Given raw user text, produce structured breakdown of user query components into triggers, automatic thoughts, emotions, and behaviors.
    """

    prompt = USER_SCHEMA_PROMPT.format(input_text=input_text)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response.choices[0].message.content.strip()

    data = safe_json_parse(content)

    required_fields = [
        "triggers",
        "automatic_thoughts",
        "emotions",
        "behaviors",
    ]

    for f in required_fields:
        if f not in data or data[f] is None:
            data[f] = []

    return data


if __name__ == "__main__":
    text = "I feel calm most of the time, but sometimes when small things pile up, it feels like nothing is going the way it should, and I end up blowing up."
    out = extract_user_schema(text)
    print(json.dumps(out, indent=2))
