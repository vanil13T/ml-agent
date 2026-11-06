import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

import agent_config as config

KNOWN_METRICS = {
    "rmse": "regression", "mae": "regression", "mse": "regression",
    "rmsle": "regression", "mape": "regression", "r2": "regression",
    "macro_f1": "classification", "weighted_f1": "classification",
    "accuracy": "classification", "roc_auc": "classification", "logloss": "classification",
}
LOWER_IS_BETTER = {"rmse", "mae", "mse", "rmsle", "mape", "logloss"}

# длинные фразы раньше коротких: "macro f1" раньше "f1"
METRIC_KEYWORDS = [
    ("macro f1", "macro_f1"), ("macro-f1", "macro_f1"), ("weighted f1", "weighted_f1"),
    ("rmsle", "rmsle"), ("rmse", "rmse"), ("mape", "mape"), ("mae", "mae"), ("mse", "mse"),
    ("roc-auc", "roc_auc"), ("roc auc", "roc_auc"), ("auc", "roc_auc"),
    ("logloss", "logloss"), ("log loss", "logloss"),
    ("accuracy", "accuracy"), ("f1", "macro_f1"), ("r2", "r2"),
]

DATA_FILES = ("train.csv", "test.csv", "sample_submission.csv", "data_dictionary.csv")


def find_file(name: str, task_dir: Path, description: str) -> Path | None:
    for match in re.findall(r"`([^`]*" + re.escape(name) + r")`", description):
        if (task_dir / match).exists():
            return task_dir / match
    found = sorted(p for p in task_dir.rglob(name) if "experiments" not in p.parts)
    return found[0] if found else None


def detect_metric(description: str, task_type: str) -> str:
    text = description.lower()
    for keyword, metric in METRIC_KEYWORDS:
        if re.search(r"\b" + re.escape(keyword) + r"\b", text) and KNOWN_METRICS[metric] == task_type:
            return metric
    return "rmse" if task_type == "regression" else "macro_f1"


def find_period_column(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> str | None:
    """Колонка, все значения которой в test больше, чем в train (test - будущий период)."""
    for col in columns:
        if not pd.api.types.is_numeric_dtype(train[col]):
            continue
        if 3 <= train[col].nunique() <= 50 and test[col].min() > train[col].max():
            return col
    return None


def load_task(task_path: Path) -> dict:
    task_dir = task_path.parent
    description = task_path.read_text(encoding="utf-8") if task_path.exists() else ""
    paths = {name: find_file(name, task_dir, description) for name in DATA_FILES}
    for name in DATA_FILES[:3]:
        if paths[name] is None:
            raise FileNotFoundError(f"{name} not found near {task_path}")

    train = pd.read_csv(paths["train.csv"], low_memory=False)
    test = pd.read_csv(paths["test.csv"], low_memory=False)
    sample = pd.read_csv(paths["sample_submission.csv"])
    id_col, target = sample.columns[0], sample.columns[1]

    y = train[target]
    is_cls = (not pd.api.types.is_numeric_dtype(y)) or pd.api.types.is_bool_dtype(y) or (
        pd.api.types.is_integer_dtype(y) and y.nunique() <= 30)
    task_type = "classification" if is_cls else "regression"
    metric = detect_metric(description, task_type)
    features = [c for c in train.columns if c not in (id_col, target) and c in test.columns]
    period_col = find_period_column(train, test, features)
    classes = sorted(y.astype(str).unique().tolist()) if is_cls else []
    match = re.search(r"Файл\s+`([^`]+\.csv)`", description)

    return {
        "task_dir": str(task_dir),
        "description": description,
        "train_path": paths["train.csv"].resolve().as_posix(),
        "test_path": paths["test.csv"].resolve().as_posix(),
        "sample_path": paths["sample_submission.csv"].resolve().as_posix(),
        "dictionary_path": paths["data_dictionary.csv"].resolve().as_posix() if paths["data_dictionary.csv"] else None,
        "output_path": str(task_dir / (match.group(1) if match else "submission.csv")),
        "work_dir": str(task_dir / "experiments"),
        "id_col": id_col, "target": target, "task_type": task_type,
        "metric": metric, "lower_is_better": metric in LOWER_IS_BETTER,
        "proba_output": is_cls and metric in ("roc_auc", "logloss") and len(classes) == 2,
        "classes": classes, "period_col": period_col,
        "validation": "group" if period_col else ("stratified" if is_cls else "kfold"),
    }


def make_folds(task: dict, train: pd.DataFrame) -> Path:
    n, y = len(train), train[task["target"]]
    fold_of = np.zeros(n, dtype=int)
    if task["validation"] == "group":
        groups = train[task["period_col"]]
        splits = GroupKFold(n_splits=min(config.N_FOLDS, groups.nunique())).split(np.zeros(n), y, groups)
    elif task["validation"] == "stratified":
        k = max(2, min(config.N_FOLDS, int(y.astype(str).value_counts().min())))
        splits = StratifiedKFold(n_splits=k, shuffle=True, random_state=config.SEED).split(np.zeros(n), y.astype(str))
    else:
        splits = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED).split(np.zeros(n))
    for k, (_, valid_idx) in enumerate(splits):
        fold_of[valid_idx] = k
    path = Path(task["work_dir"]) / "folds.npy"
    np.save(path, fold_of)
    return path


def describe_data(task: dict, train: pd.DataFrame, test: pd.DataFrame, max_cols: int = 80) -> str:
    descriptions = {}
    if task["dictionary_path"]:
        d = pd.read_csv(task["dictionary_path"])
        text_col = next((c for c in d.columns if "desc" in c.lower()), None)
        if "feature" in d.columns and text_col:
            descriptions = dict(zip(d["feature"], d[text_col].astype(str)))

    y = train[task["target"]]
    if task["task_type"] == "regression":
        q = y.quantile([0, 0.25, 0.5, 0.75, 0.99, 1]).round(3).tolist()
        target_info = f"числа, квантили 0/25/50/75/99/100%: {q}"
    else:
        target_info = f"классы и частоты: {y.value_counts().to_dict()}"
    lines = [f"train: {train.shape[0]} строк, test: {test.shape[0]} строк",
             f"Таргет {task['target']}: {target_info}",
             "Признаки (тип, уникальных, пропусков, описание; [!] - значения test вне диапазона train):"]
    features = [c for c in train.columns if c not in (task["id_col"], task["target"]) and c in test.columns]
    for col in features[:max_cols]:
        s, t = train[col], test[col]
        flag = ""
        if pd.api.types.is_numeric_dtype(s) and s.notna().any() and t.notna().any():
            if t.min() > s.max() or t.max() < s.min():
                flag = f" [!] train {s.min()}..{s.max()}, test {t.min()}..{t.max()}"
        example = "" if pd.api.types.is_numeric_dtype(s) else f", примеры {s.dropna().astype(str).unique()[:4].tolist()}"
        lines.append(f"- {col} ({s.dtype}, уник. {s.nunique()}, пропусков {s.isna().mean():.0%}{example}): "
                     f"{descriptions.get(col, '')[:90]}{flag}")
    if len(features) > max_cols:
        lines.append(f"... и ещё {len(features) - max_cols} колонок")
    return "\n".join(lines)


def check_predictions(task: dict, path: Path, ids: pd.Series) -> tuple[pd.Series | None, str | None]:
    """Проверяет submission.csv / oof.csv. Возвращает (предсказания, None) или (None, ошибка)."""
    if not path.exists():
        return None, f"нет {path.name}"
    try:
        df = pd.read_csv(path)
    except Exception as error:
        return None, f"{path.name} не читается: {error}"
    expected = [task["id_col"], task["target"]]
    if list(df.columns) != expected:
        return None, f"в {path.name} колонки {list(df.columns)}, а нужны {expected}"
    if len(df) != len(ids):
        return None, f"в {path.name} {len(df)} строк, а нужно {len(ids)}"
    if not (df[task["id_col"]].astype(str).values == ids.astype(str).values).all():
        return None, f"в {path.name} id не совпадают или идут в другом порядке"
    pred = df[task["target"]]
    if pred.isna().any():
        return None, f"в {path.name} {int(pred.isna().sum())} пропусков"
    if task["task_type"] == "regression" or task["proba_output"]:
        if not pd.api.types.is_numeric_dtype(pred) or not np.isfinite(pred).all():
            return None, f"в {path.name} должны быть конечные числа"
    else:
        unknown = set(pred.astype(str)) - set(task["classes"])
        if unknown:
            return None, f"в {path.name} неизвестные классы: {sorted(unknown)[:5]}"
    return pred, None


def compute_metric(task: dict, y_true: pd.Series, pred: pd.Series) -> float:
    m = task["metric"]
    if task["task_type"] == "regression":
        y, p = y_true.to_numpy(dtype=float), pred.to_numpy(dtype=float)
        if m == "rmse":
            return float(np.sqrt(metrics.mean_squared_error(y, p)))
        if m == "mse":
            return float(metrics.mean_squared_error(y, p))
        if m == "mae":
            return float(metrics.mean_absolute_error(y, p))
        if m == "rmsle":
            return float(np.sqrt(metrics.mean_squared_log_error(y, np.clip(p, 0, None))))
        if m == "mape":
            return float(metrics.mean_absolute_percentage_error(y, p))
        return float(metrics.r2_score(y, p))
    if task["proba_output"]:
        y = (y_true.astype(str) == task["classes"][1]).astype(int).to_numpy()
        p = np.clip(pred.to_numpy(dtype=float), 1e-15, 1 - 1e-15)
        return float(metrics.roc_auc_score(y, p) if m == "roc_auc" else metrics.log_loss(y, p))
    y, p = y_true.astype(str).to_numpy(), pred.astype(str).to_numpy()
    if m == "accuracy":
        return float(metrics.accuracy_score(y, p))
    if m == "weighted_f1":
        return float(metrics.f1_score(y, p, average="weighted"))
    return float(metrics.f1_score(y, p, average="macro"))


def is_better(task: dict, new: float, old: float | None) -> bool:
    if old is None:
        return True
    if task["lower_is_better"]:
        return new < old - abs(old) * config.MIN_GAIN
    return new > old + abs(old) * config.MIN_GAIN


def save_submission(task: dict, source: Path) -> None:
    tmp = Path(task["output_path"] + ".tmp")
    shutil.copyfile(source, tmp)
    tmp.replace(task["output_path"])
