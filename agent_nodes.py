import json
import logging
import re
import time
import uuid
from pathlib import Path

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

import agent_config as config
import agent_prompts as prompts
import agent_tools as tools
from agent_task import check_predictions, compute_metric, describe_data, is_better, make_folds, save_submission

logger = logging.getLogger(__name__)


def analyze_task_node(state: dict) -> dict:
    task = state["task"]
    tools.set_work_dir(task["work_dir"])
    train = pd.read_csv(task["train_path"], low_memory=False)
    test = pd.read_csv(task["test_path"], low_memory=False)
    task = {**task, "folds_path": make_folds(task, train).resolve().as_posix()}
    profile = describe_data(task, train, test)
    return {
        "task": task,
        "best_score": None, "best_exp": None, "best_code": None, "last_error": None,
        "experiment_num": 0, "fails_in_row": 0, "nudges": 0, "llm_failures": 0, "history": [],
        "messages": [SystemMessage(prompts.system_prompt(task)),
                     HumanMessage(prompts.first_message(task, profile))],
    }


def trim(messages: list) -> list:
    """Системный промпт, задача и последние шаги. ToolMessage не должен остаться без своего вызова."""
    head, tail = messages[:2], messages[2:][-config.KEEP_LAST_MESSAGES:]
    while tail and isinstance(tail[0], ToolMessage):
        tail = tail[1:]
    return head + tail


def status_text(state: dict) -> str:
    task = state["task"]
    direction = "меньше" if task["lower_is_better"] else "больше"
    lines = [f"Метрика {task['metric']} ({direction} - лучше)."]
    if state.get("best_exp"):
        lines.append(f"Лучший результат: {state['best_exp']}, {task['metric']} = {state['best_score']:.5f}.")
    else:
        lines.append("Пока нет ни одного засчитанного эксперимента.")
    if state.get("last_error"):
        lines.append(f"Последний эксперимент: {state['last_error']}")
    if state.get("history"):
        lines.append("Журнал (последние 8): " + json.dumps(state["history"][-8:], ensure_ascii=False))
    lines.append(f"Осталось {config.time_left() / 60:.0f} мин, экспериментов "
                 f"{state.get('experiment_num', 0)} из {config.MAX_EXP}.")
    if state.get("best_code"):
        lines.append(f"Код лучшего эксперимента:\n```python\n{state['best_code'][:8000]}\n```")
    lines.append("Следующий шаг: вызови execute_code с полным скриптом нового эксперимента или ответь ФИНАЛ.")
    return "\n".join(lines)


def normalize(response: AIMessage) -> AIMessage:
    """Чистит <think>, оставляет один tool call; код из markdown-блока превращает в вызов execute_code."""
    content = response.content if isinstance(response.content, str) else ""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    if response.tool_calls:
        return AIMessage(content=content, tool_calls=response.tool_calls[:1])
    blocks = re.findall(r"```(?:python3?|py)?[ \t]*\n(.*?)```", content, flags=re.DOTALL | re.IGNORECASE)
    code = max(blocks, key=len).strip() if blocks else ""
    if "oof.csv" in code and "submission" in code:
        call = {"name": "execute_code", "args": {"code": code}, "id": f"call_{uuid.uuid4().hex[:12]}"}
        return AIMessage(content="Запускаю эксперимент.", tool_calls=[call])
    return AIMessage(content=content or "(пустой ответ)")


def generate_code_node(state: dict, llm) -> dict:
    messages = trim(state["messages"]) + [HumanMessage(status_text(state))]
    for attempt in range(1, 4):
        try:
            response = llm.invoke(messages)
            break
        except Exception as error:
            logger.warning("LLM error (attempt %d): %s", attempt, str(error)[:300])
            time.sleep(5 * attempt)
    else:
        return {"llm_failures": state.get("llm_failures", 0) + 1}

    return {"messages": [normalize(response)], "llm_failures": 0}


def nudge_node(state: dict) -> dict:
    text = "Ты не вызвал инструмент. Вызови execute_code с ПОЛНЫМ скриптом эксперимента или ответь ФИНАЛ."
    return {"messages": [HumanMessage(text)], "nudges": state.get("nudges", 0) + 1}


def judge(task: dict, exp_id: str, best_score: float | None) -> dict:
    """Проверяет файлы эксперимента и считает метрику по oof.csv."""
    exp_dir = Path(task["work_dir"]) / exp_id
    result = {"ok": False, "score": None, "better": False, "error": None, "warning": None}
    train = pd.read_csv(task["train_path"], usecols=[task["id_col"], task["target"]], low_memory=False)
    test_ids = pd.read_csv(task["test_path"], usecols=[task["id_col"]])[task["id_col"]]

    oof, error = check_predictions(task, exp_dir / "oof.csv", train[task["id_col"]])
    if error is None:
        _, error = check_predictions(task, exp_dir / "submission.csv", test_ids)
    if error is None:
        try:
            result["score"] = compute_metric(task, train[task["target"]], oof)
        except Exception as e:
            error = f"метрика не считается по oof.csv: {e}"
    if error:
        result["error"] = error
        return result

    value = result["score"]
    result.update(ok=True, better=is_better(task, value, best_score))
    if (task["lower_is_better"] and value <= 1e-9) or (not task["lower_is_better"] and value >= 0.99999):
        result["warning"] = "скор идеален - проверь, нет ли утечки таргета (обучение только на folds != k)"
    if result["better"]:
        save_submission(task, exp_dir / "submission.csv")
    return result


def evaluate_node(state: dict) -> dict:
    last_tools = []
    for message in reversed(state["messages"]):
        if not isinstance(message, ToolMessage):
            break
        last_tools.append(message)
    runs = [m for m in last_tools if m.name == "execute_code"]
    match = re.search(r"exp_\d+", runs[0].content) if runs else None
    if not match:
        return {"nudges": 0}

    task, exp_id, best = state["task"], match.group(0), state.get("best_score")
    verdict = judge(task, exp_id, best)
    entry = {"exp": exp_id, "score": verdict["score"], "kept": verdict["better"]}
    if verdict["ok"] and verdict["better"]:
        note = f"{exp_id}: {task['metric']} = {verdict['score']:.5f} - НОВЫЙ ЛУЧШИЙ, ответ обновлён"
        if best is not None:
            note += f" (был {best:.5f})"
    elif verdict["ok"]:
        note = f"{exp_id}: {task['metric']} = {verdict['score']:.5f} - не лучше {best:.5f}"
    else:
        note = f"{exp_id} не засчитан: {verdict['error']}"
        entry["error"] = verdict["error"][:200]
    if verdict["warning"]:
        note += f". ВНИМАНИЕ: {verdict['warning']}"

    update = {
        "experiment_num": state.get("experiment_num", 0) + 1,
        "fails_in_row": 0 if verdict["better"] else state.get("fails_in_row", 0) + 1,
        "last_error": note, "history": [entry], "nudges": 0,
    }
    if verdict["better"]:
        code_path = Path(task["work_dir"]) / exp_id / "code.py"
        update.update(best_score=verdict["score"], best_exp=exp_id,
                      best_code=code_path.read_text(encoding="utf-8"))
    return update
