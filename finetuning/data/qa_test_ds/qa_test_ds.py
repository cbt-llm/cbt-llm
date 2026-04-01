from datasets import load_dataset
import os
from dotenv import load_dotenv


if __name__=="__main__":
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
     
    ds = load_dataset("Psychotherapy-LLM/CBT-Bench", "qa_test",token=hf_token)
    ds["train"].to_csv("qa_test.csv")