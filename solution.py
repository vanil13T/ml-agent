import logging
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from agent_graph import build_graph  # noqa: E402
from agent_task import load_task  # noqa: E402

logger = logging.getLogger("agent")


def main(task_path: Path) -> None:
    task = load_task(task_path)
    os.chdir(task["task_dir"])
    try:
        build_graph().invoke({"task": task, "messages": []}, {"recursion_limit": 1000})
    except Exception:
        logger.exception("agent graph failed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        main(Path(sys.argv[1] if len(sys.argv) > 1 else "task.md").resolve())
    except Exception:
        logger.exception("fatal error")
