import agent_config as config

SYSTEM_PROMPT = """Ты - автономный ML-инженер. Строишь лучшую модель за отведённое время.
Ты сам решаешь, какие признаки, модели и параметры пробовать.

ИНСТРУМЕНТЫ:
- read_csv_head(path, n) - посмотреть первые строки CSV.
- execute_code(code) - запустить ПОЛНЫЙ Python-скрипт одного эксперимента.
  Каждый запуск - новый процесс в своей пустой папке.

ПРАВИЛА СКРИПТА (иначе эксперимент не засчитают):
1. Данные читай по абсолютным путям из задания.
2. Валидация - ТОЛЬКО по общим фолдам: folds = np.load(FOLDS_PATH) - номер фолда для каждой
   строки train. Для фолда k модель и всё, что использует таргет (target encoding и т.п.),
   обучай только на строках folds != k. Так скоры всех экспериментов честные и сравнимые.
3. Сохрани в ТЕКУЩУЮ папку два файла с колонками [{id_col}, {target}]:
   - submission.csv - предсказания для test.csv, строки в порядке test.csv;
   - oof.csv - предсказания для train.csv, строки в порядке train.csv; каждое предсказание
     сделано моделью, которая эту строку НЕ видела (фолд k предсказывает строки folds == k).
   {format_hint}
4. Метрику ({metric}, {direction}) по oof.csv считает проверяющий сам. Для себя печатай
   метрику по фолдам, итог, топ важных признаков, для классификации - F1 по классам.
5. Библиотеки: pandas, numpy, scipy, scikit-learn, xgboost, lightgbm, catboost,
   imbalanced-learn, category_encoders, optuna. n_jobs={n_jobs}.
6. Интернет и предобученные модели запрещены.
7. Скрипт должен уложиться в {timeout_min} мин (для бустингов - early stopping).

КАК РАБОТАТЬ:
- Первым запусти пример из задания (можно сразу немного улучшить) - это baseline.
- Один вызов execute_code = один эксперимент = одно изменение. Так понятно, что помогло.
- Если execute_code упал - читай stderr, исправляй, запускай снова.
- Если эксперимент хуже лучшего - пойми почему и попробуй другой подход (смотри журнал).
- Хорошие направления: признаки из предметной области, другие модели (CatBoost, XGBoost),
  работа с дисбалансом классов, параметры.
- Пиши код сразу в вызов execute_code, а не текстом.
- Когда идеи кончились - ответь ТОЛЬКО текстом "ФИНАЛ" без вызова тулов."""

FORMAT_HINT = {
    "regression": "В колонке {target} - число.",
    "labels": "В колонке {target} - метка класса, одна из: {classes}.",
    "proba": "В колонке {target} - вероятность класса {positive}.",
}

FIRST_MESSAGE = """Вот задача:

{description}

# Что известно заранее
- Таргет: `{target}`, id: `{id_col}`, тип: {task_type}
- Метрика: {metric} ({direction})
- Фолды: {validation_text}
- Пути:
  TRAIN_PATH = "{train_path}"
  TEST_PATH = "{test_path}"
  SAMPLE_PATH = "{sample_path}"
  FOLDS_PATH = "{folds_path}"

# Данные
{profile}

# Рабочий пример эксперимента (baseline)
```python
{baseline}
```

Начни с запуска baseline через execute_code, дальше исследуй сам."""

VALIDATION_TEXT = {
    "group": "по колонке `{period_col}`: в test - более поздний период, чем в train, "
             "поэтому модель проверяется на периодах, которых не видела",
    "stratified": "стратифицированные: доля каждого класса в фолдах одинакова",
    "kfold": "случайные",
}

BASELINE_TEMPLATE = '''import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn import metrics

TRAIN_PATH = "__TRAIN__"
TEST_PATH = "__TEST__"
FOLDS_PATH = "__FOLDS__"
TARGET, ID = "__TARGET__", "__ID__"
IS_CLS = __IS_CLS__
PROBA = __PROBA__

train = pd.read_csv(TRAIN_PATH, low_memory=False)
test = pd.read_csv(TEST_PATH, low_memory=False)
folds = np.load(FOLDS_PATH)  # номер фолда для каждой строки train
n_folds = folds.max() + 1

feats = [c for c in train.columns if c not in (ID, TARGET) and c in test.columns]
X, X_test = train[feats].copy(), test[feats].copy()
for col in feats:
    if not pd.api.types.is_numeric_dtype(X[col]):
        cats = pd.concat([X[col], X_test[col]]).astype(str).unique()
        X[col] = pd.Categorical(X[col].astype(str), categories=cats)
        X_test[col] = pd.Categorical(X_test[col].astype(str), categories=cats)

if IS_CLS:
    classes, y = np.unique(train[TARGET].astype(str), return_inverse=True)
    Model = lgb.LGBMClassifier
    oof, pred = np.zeros((len(X), len(classes))), np.zeros((len(X_test), len(classes)))
else:
    y = train[TARGET].to_numpy(dtype=float)
    Model = lgb.LGBMRegressor
    oof, pred = np.zeros(len(X)), np.zeros(len(X_test))

params = dict(n_estimators=3000, learning_rate=0.05, num_leaves=63, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              n_jobs=__N_JOBS__, random_state=42, verbose=-1)
for k in range(n_folds):
    tr, va = folds != k, folds == k
    model = Model(**params)
    model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    if IS_CLS:
        oof[va] = model.predict_proba(X[va])
        pred += model.predict_proba(X_test) / n_folds
    else:
        oof[va] = model.predict(X[va])
        pred += model.predict(X_test) / n_folds
    print(f"fold {k}: деревьев {model.best_iteration_}", flush=True)

importance = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
print("Топ-15 признаков:\\n" + importance.head(15).to_string())

def to_answer(p):
    if not IS_CLS:
        return p
    if PROBA:
        return p[:, 1]
    return classes[p.argmax(1)]

if IS_CLS and not PROBA:
    print("macro_f1:", metrics.f1_score(y, oof.argmax(1), average="macro"))
    f1 = metrics.f1_score(y, oof.argmax(1), average=None, labels=range(len(classes)))
    print("F1 по классам:", dict(zip(classes, f1.round(3))))
elif not IS_CLS:
    print("rmse:", np.sqrt(metrics.mean_squared_error(y, oof)))

pd.DataFrame({ID: train[ID], TARGET: to_answer(oof)}).to_csv("oof.csv", index=False)
pd.DataFrame({ID: test[ID], TARGET: to_answer(pred)}).to_csv("submission.csv", index=False)
'''


def baseline_code(task: dict) -> str:
    replacements = {
        "__TRAIN__": task["train_path"], "__TEST__": task["test_path"], "__FOLDS__": task["folds_path"],
        "__TARGET__": task["target"], "__ID__": task["id_col"], "__N_JOBS__": str(config.N_JOBS),
        "__IS_CLS__": str(task["task_type"] == "classification"), "__PROBA__": str(task["proba_output"]),
    }
    code = BASELINE_TEMPLATE
    for key, value in replacements.items():
        code = code.replace(key, value)
    return code


def _direction(task: dict) -> str:
    return "меньше - лучше" if task["lower_is_better"] else "больше - лучше"


def system_prompt(task: dict) -> str:
    if task["task_type"] == "regression":
        hint = FORMAT_HINT["regression"].format(target=task["target"])
    elif task["proba_output"]:
        hint = FORMAT_HINT["proba"].format(target=task["target"], positive=task["classes"][1])
    else:
        hint = FORMAT_HINT["labels"].format(target=task["target"], classes=task["classes"])
    return SYSTEM_PROMPT.format(id_col=task["id_col"], target=task["target"], format_hint=hint,
                                metric=task["metric"], direction=_direction(task), n_jobs=config.N_JOBS,
                                timeout_min=int(config.RUN_TIMEOUT_SEC // 60))


def first_message(task: dict, profile: str) -> str:
    return FIRST_MESSAGE.format(
        description=task["description"][:6000], target=task["target"], id_col=task["id_col"],
        task_type=task["task_type"], metric=task["metric"], direction=_direction(task),
        validation_text=VALIDATION_TEXT[task["validation"]].format(period_col=task["period_col"]),
        train_path=task["train_path"], test_path=task["test_path"], sample_path=task["sample_path"],
        folds_path=task["folds_path"], profile=profile, baseline=baseline_code(task),
    )
