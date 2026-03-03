import re
import json
import requests
from datetime import datetime
from cbt_llm.config import OUTPUT_LLM_DIR
from .prompt_templates.prompt import PROMPT_SNOMED
import pandas as pd
import os


MODELS = [
    "gemma3:12b",
    "llama3:latest",
    "mistral:7b-instruct"
]

OLLAMA_URL = "http://localhost:11434/api/generate"


def run_model(user_text: str, model_name: str):
    prompt = PROMPT_SNOMED.format(user_text=user_text)

    try:
        response = requests.post(OLLAMA_URL, json={
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }, timeout=120)
        response.raise_for_status()
        raw_output = response.json().get("response", "").strip()
    except requests.RequestException as e:
        return {"error": f"Ollama API error for model {model_name}: {e}"}

    match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if not match:
        return {
            "error": f"No valid JSON returned for model {model_name}",
            "raw_output": raw_output
        }

    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {
            "error": f"Invalid JSON format for model {model_name}",
            "raw_output": raw_output
        }


def run_all_models(user_text_list: list):
    """
    user_text_list: list of queries to run
    """

    for model in MODELS:
        print(f"\n========== Running Model: {model} ==========\n")
        all_rows = []

        for user_text in user_text_list:
            model_result = run_model(user_text, model)

            row = {"User Query": user_text}

            if "error" in model_result:
                for i in range(1, 6):
                    row[f"Concept {i}"] = model_result.get("error")
            else:
                findings = model_result.get("findings", [])
                for i in range(5):
                    if i < len(findings):
                        row[f"Concept {i+1}"] = findings[i].get("term", "")
                    else:
                        row[f"Concept {i+1}"] = ""

            all_rows.append(row)

        # Create DataFrame
        df = pd.DataFrame(all_rows)

        # Create timestamped filename per model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model.replace(':','_')}_results_{timestamp}.csv"
        filepath = os.path.join(OUTPUT_LLM_DIR, filename)

        # Save to CSV
        df.to_csv(filepath, index=False)
        print(f"Results for model {model} saved to: {filepath}")
