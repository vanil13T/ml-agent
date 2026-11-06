import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

import agent_config as config

WORK_DIR = Path("experiments")


def set_work_dir(path: str) -> None:
    global WORK_DIR
    WORK_DIR = Path(path)
    WORK_DIR.mkdir(parents=True, exist_ok=True)


def _text(value) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def run_experiment(code: str) -> tuple[str, str]:
    exp_id = f"exp_{len(list(WORK_DIR.glob('exp_*'))):03d}"
    exp_dir = WORK_DIR / exp_id
    exp_dir.mkdir(parents=True)
    (exp_dir / "code.py").write_text(code, encoding="utf-8")

    timeout = max(30.0, min(config.RUN_TIMEOUT_SEC, config.time_left() - 60))
    env = {k: v for k, v in os.environ.items() if not k.startswith("OPENAI")}
    env["PYTHONIOENCODING"] = "utf-8"
    start = time.time()
    try:
        result = subprocess.run([sys.executable, "-u", "code.py"], cwd=exp_dir, env=env,
                                capture_output=True, encoding="utf-8", errors="replace", timeout=timeout)
        status = f"код возврата {result.returncode}"
        stdout, stderr = result.stdout, result.stderr
    except subprocess.TimeoutExpired as error:
        status = f"ПРЕВЫШЕН ЛИМИТ ВРЕМЕНИ {timeout / 60:.0f} мин"
        stdout, stderr = _text(error.stdout), _text(error.stderr)

    seconds = time.time() - start
    saved = ", ".join(f"{name}: {'есть' if (exp_dir / name).exists() else 'НЕТ'}"
                      for name in ("submission.csv", "oof.csv"))
    report = (f"Эксперимент {exp_id}: {status}, {seconds:.0f} с. {saved}\n"
              f"STDOUT (конец):\n{stdout[-config.MAX_OUTPUT_CHARS:]}\n"
              f"STDERR (конец):\n{stderr[-config.MAX_OUTPUT_CHARS:]}")
    (exp_dir / "output.txt").write_text(f"{status}\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}", encoding="utf-8")
    return exp_id, report


@tool
def read_csv_head(path: str, n: int = 10) -> str:
    """Показывает первые n строк CSV (не больше 20) и типы колонок."""
    try:
        df = pd.read_csv(path, nrows=min(max(n, 1), 20))
    except Exception as e:
        return f"ERROR: {e}"
    return f"dtypes:\n{df.dtypes.to_string()}\n\nhead:\n{df.to_string()}"[:6000]


@tool
def execute_code(code: str) -> str:
    """Запускает ПОЛНЫЙ Python-скрипт одного эксперимента в новом процессе и своей пустой папке.
    Скрипт обязан сохранить в текущую папку submission.csv и oof.csv.
    Возвращает код возврата и конец stdout/stderr. Таймаут 15 минут."""
    _, report = run_experiment(code)
    return report


tools = [read_csv_head, execute_code]
