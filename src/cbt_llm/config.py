import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

SNOMEDCT_DIR = os.getenv("SNOMEDCT_DIR")
SNOMEDCT_CORE_FILE = os.getenv("SNOMEDCT_CORE_FILE")

OUTPUT_NEO4J_DIR = ROOT / "src" / "output_files" / "neo4j_retrival_output"
OUTPUT_LLM_DIR = ROOT / "src" / "output_files" / "llm_retrival_output"

OUTPUT_NEO4J_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_LLM_DIR.mkdir(parents=True, exist_ok=True)
