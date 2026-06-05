import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# The vendored bot reads its system prompt from SYSTEM_PROMPT_PATH; point it at the
# harness's filled prompt so live evals run against known ground-truth hotel data.
os.environ.setdefault(
    "SYSTEM_PROMPT_PATH",
    str(Path(__file__).parent / "data" / "system_prompt.txt"),
)


def has_key(name: str) -> bool:
    return bool(os.environ.get(name, "").strip())
