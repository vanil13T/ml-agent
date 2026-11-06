import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
if not os.getenv("OPENAI_BASE_URL"):
    os.environ.pop("OPENAI_BASE_URL", None)  # пустая строка ломает клиент OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or None
OPENAI_MODEL = os.getenv("OPENAI_MODEL") or "Qwen3.8-27B"
TEMPERATURE = 0.3
MAX_TOKENS = 8000

START_TIME = time.time()
TIME_LIMIT_SEC = float(os.getenv("TIME_LIMIT_SEC", 2 * 3600 + 50 * 60))
RUN_TIMEOUT_SEC = 15 * 60
MAX_EXP = 25
MAX_FAILS_IN_ROW = 6
MAX_NUDGES = 3
MIN_GAIN = 0.001

N_FOLDS = 5
N_JOBS = 8
SEED = 42

KEEP_LAST_MESSAGES = 6
MAX_OUTPUT_CHARS = 3000


def time_left() -> float:
    return TIME_LIMIT_SEC - (time.time() - START_TIME)
