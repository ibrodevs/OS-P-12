"""Practical work 12: leakage-safe cross-validation and hyperparameter search.

All hyperparameter selection and nested CV are conducted on the development set;
the untouched holdout is used only for final reporting.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import loguniform
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).parent / 'data' / 'healthcare-dataset-stroke-data.csv'
# sklearn >=1.8 deprecated `penalty` in favor of `l1_ratio`.
SKLEARN_USE_L1_RATIO = tuple(int(i) for i in sklearn.__version__.split('.')[:2]) >= (1, 8)
PENALTY_KEY = 'model__l1_ratio' if SKLEARN_USE_L1_RATIO else 'model__penalty'
PENALTY_VALUES = [1.0, 0.0] if SKLEARN_USE_L1_RATIO else ['l1', 'l2']
GRID = {
    PENALTY_KEY: PENALTY_VALUES,
    'model__C': [0.03, 0.3, 3.0],
    'model__class_weight': [None, 'balanced'],
}
RANDOM_SPACE = {
    PENALTY_KEY: PENALTY_VALUES,
    'model__C': loguniform(0.01, 100.0),
    'model__class_weight': [None, 'balanced'],
}


def get_data(path: Path) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f'CSV file not found: {path}. See README.md for data source.')
    frame = pd.read_csv(path)
    if 'stroke' not in frame:
        raise ValueError("Expected the binary target column 'stroke'.")
    if frame['stroke'].isna().any() or set(frame['stroke'].unique()) != {0, 1}:
        raise ValueError('Target stroke must be non-missing and contain both classes 0 and 1.')
    x = frame.drop(columns=['stroke', 'id'], errors='ignore')
    y = frame['stroke'].astype(int)
    if x.columns.duplicated().any() or len(x) < 30:
        raise ValueError('Invalid dataset shape or duplicate feature column names.')
    info = {
        'rows': int(len(frame)),
        'features': int(x.shape[1]),
        'positives': int(y.sum()),
        'prevalence': float(y.mean()),
        'removed_id': 'id' in frame,
    }
    return x, y, info


def make_pipeline(x: pd.DataFrame, *, c: float = 1.0,
                  penalty: str = 'l2', class_weight: str | None = None) -> Pipeline:
    numeric = x.select_dtypes(include='number').columns.tolist()
    categorical = x.select_dtypes(exclude='number').columns.tolist()
    if not numeric and not categorical:
        raise ValueError('No features are available.')
    transformers = []
    if numeric:
        transformers.append(('numerical', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), categorical))
    return Pipeline([
        ('preprocess', ColumnTransformer(transformers, remainder='drop')),
        ('model', LogisticRegression(
            solver='liblinear', max_iter=2000, C=c,
            class_weight=class_weight, random_state=42,
            **({'l1_ratio': 1.0 if penalty == 'l1' else 0.0}
               if SKLEARN_USE_L1_RATIO else {'penalty': penalty}),
        )),
    ])


def cv_splits(folds: int, seed: int) -> StratifiedKFold:
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)


def holdout_metrics(fitted: Pipeline, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    proba = fitted.predict_proba(x)[:, 1]
    predictions = fitted.predict(x)
    return {
        'average_precision': float(average_precision_score(y, proba)),
        'recall': float(recall_score(y, predictions, zero_division=0)),
        'f1': float(f1_score(y, predictions, zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y, predictions)),
    }


def run_searches(x: pd.DataFrame, y: pd.Series, folds: int,
                 seed: int, n_iter: int) -> dict[str, Any]:
    cv = cv_splits(folds, seed)
    pipeline = make_pipeline(x)
    grid = GridSearchCV(pipeline, GRID, scoring='average_precision',
                        cv=cv, n_jobs=1, refit=True, error_score='raise')
    random_search = RandomizedSearchCV(
        pipeline, RANDOM_SPACE, n_iter=n_iter, scoring='average_precision',
        cv=cv, random_state=seed, n_jobs=1, refit=True, error_score='raise',
    )
    summaries = {}
    for name, search in [('grid', grid), ('random', random_search)]:
        started = time.perf_counter()
        search.fit(x, y)
        elapsed = time.perf_counter() - started
        summaries[name] = {
            'best_cv_ap': float(search.best_score_),
            'best_params': search.best_params_,
            'tested_candidates': int(len(search.cv_results_['params'])),
            'total_cv_fits': int(len(search.cv_results_['params']) * folds),
            'elapsed_seconds': float(elapsed),
            '_estimator': search.best_estimator_,
        }
    return summaries


def optuna_search(x: pd.DataFrame, y: pd.Series, folds: int,
                  seed: int, n_trials: int) -> dict[str, Any]:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError('Install dependencies: pip install -r requirements.txt (optuna required).') from exc
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = cv_splits(folds, seed)

    def objective(trial: Any) -> float:
        candidate = make_pipeline(
            x,
            c=trial.suggest_float('C', 0.01, 100.0, log=True),
            penalty=trial.suggest_categorical('penalty', ['l1', 'l2']),
            class_weight=trial.suggest_categorical('class_weight', ['none', 'balanced'])
                .replace('none', '') or None,
        )
        values = cross_val_score(candidate, x, y, cv=cv,
                                 scoring='average_precision', error_score='raise')
        return float(np.mean(values))

    started = time.perf_counter()
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
    elapsed = time.perf_counter() - started
    p = study.best_params
    estimator = make_pipeline(x, c=float(p['C']), penalty=p['penalty'],
                              class_weight=None if p['class_weight'] == 'none' else 'balanced')
    estimator.fit(x, y)
    return {
        'best_cv_ap': float(study.best_value),
        'best_params': {'model__C': float(p['C']),
                        'model__penalty': p['penalty'],
                        'model__class_weight': None if p['class_weight'] == 'none' else 'balanced'},
        'tested_candidates': int(len(study.trials)),
        'total_cv_fits': int(len(study.trials) * folds),
        'elapsed_seconds': float(elapsed),
        '_estimator': estimator,
    }


def nested_cv(x: pd.DataFrame, y: pd.Series, outer_folds: int,
              inner_folds: int, seed: int) -> list[float]:
    scores = []
    outer = cv_splits(outer_folds, seed + 30)
    for train_index, val_index in outer.split(x, y):
        inner_train, outer_validation = x.iloc[train_index], x.iloc[val_index]
        inner_y, outer_y = y.iloc[train_index], y.iloc[val_index]
        search = GridSearchCV(
            make_pipeline(inner_train), GRID, scoring='average_precision',
            cv=cv_splits(inner_folds, seed + 31), n_jobs=1,
            refit=True, error_score='raise',
        )
        search.fit(inner_train, inner_y)
        probability = search.best_estimator_.predict_proba(outer_validation)[:, 1]
        scores.append(float(average_precision_score(outer_y, probability)))
    return scores


def regularization_study(x: pd.DataFrame, y: pd.Series,
                         folds: int, seed: int) -> dict[str, Any]:
    results = {}
    for penalty in ('l1', 'l2'):
        search = GridSearchCV(
            make_pipeline(x), {
                PENALTY_KEY: [1.0 if penalty == 'l1' else 0.0] if SKLEARN_USE_L1_RATIO else [penalty],
                'model__C': GRID['model__C'],
                'model__class_weight': GRID['model__class_weight'],
            }, cv=cv_splits(folds, seed), scoring='average_precision',
            n_jobs=1, refit=True, error_score='raise',
        )
        search.fit(x, y)
        coef = search.best_estimator_.named_steps['model'].coef_.ravel()
        results[penalty] = {
            'best_cv_ap': float(search.best_score_),
            'best_params': search.best_params_,
            'nonzero_coefficients': int(np.count_nonzero(np.abs(coef) > 1e-8)),
            'total_coefficients': int(coef.size),
        }
    return results


def markdown_report(r: dict[str, Any]) -> str:
    a, b, c = r['A'], r['B'], r['C']
    fmt = lambda v: f'{v:.4f}'
    timings = '\n'.join(
        f"| {name} | {s['tested_candidates']} | {s['total_cv_fits']} | {fmt(s['best_cv_ap'])} | {s['elapsed_seconds']:.2f} |"
        for name, s in c['methods'].items()
    )
    reg = '\n'.join(
        f"| {name.upper()} | {fmt(s['best_cv_ap'])} | {s['nonzero_coefficients']} / {s['total_coefficients']} | `{s['best_params']}` |"
        for name, s in c['regularization'].items()
    )
    return f'''# Отчёт — практическая работа № 12

**Задача:** классификация `stroke` на историческом датасете; это учебный пример, **не инструмент медицинской диагностики**.

## Данные и разделение

- Наблюдений: **{r['data']['rows']}**, признаков после удаления `id`: **{r['data']['features']}**.
- Положительный класс: **{r['data']['positives']}** ({r['data']['prevalence']:.2%}).
- Train/development: **{r['split']['train_rows']}**, test/holdout: **{r['split']['test_rows']}**. Стратификация 80/20, seed={r['split']['seed']}.
- **Основная метрика — Average Precision (AP, площадь под precision–recall кривой в реализации sklearn)**; accuracy при редком положительном классе неинформативна.
- Imputer, StandardScaler и OneHotEncoder обучаются **только внутри Pipeline в каждой обучающей части**.
- Подбор модели и все внутренние/внешние фолды уровня C используют **только development**. Тест не участвовал в подборе.

## Уровень A — одиночное разбиение против пяти фолдов

- Holdout AP, seed={a['seed_1']}: **{fmt(a['holdout_ap_1'])}**.
- Holdout AP, seed={a['seed_2']}: **{fmt(a['holdout_ap_2'])}** (второе разбиение приведено **только для учебного сравнения**).
- CV AP по фолдам **только на development**: {', '.join(fmt(x) for x in a['cv_scores'])}.
- Среднее: **{fmt(a['cv_mean'])}**, выборочное стандартное отклонение: **{fmt(a['cv_std'])}** (размах **{fmt(a['cv_range'])}**).

**Вывод:** оценки на фолдах колеблются из-за состава наблюдений и редкости положительного класса. Один train/test split может дать слишком оптимистичную или пессимистичную оценку; CV показывает разброс и устойчивость. При другом random_state метрика изменяется, поэтому нельзя делать вывод только по одному разбиению.

## Уровень B — поиск гиперпараметров

- CV AP базовой модели (до поиска): **{fmt(b['baseline_cv_ap'])}**.
- Лучший GridSearchCV: **{fmt(b['grid_cv_ap'])}**; параметры: `{b['grid_params']}`.
- Лучший RandomizedSearchCV: **{fmt(b['random_cv_ap'])}**; параметры: `{b['random_params']}`.
- Модель выбрана **по development CV**, победитель: **{b['winner']}**.
- AP на untouched holdout до подбора: **{fmt(b['baseline_test']['average_precision'])}**, после: **{fmt(b['tuned_test']['average_precision'])}**.
- Recall: до **{fmt(b['baseline_test']['recall'])}**, после **{fmt(b['tuned_test']['recall'])}**. Дополнительно: F1, balanced accuracy в `metrics.json`.

**Почему три оценки отличаются?** Среднее CV базовой модели измеряет работу без подбора; максимум CV поиска зависит от выбора лучшего из нескольких кандидатов и потому может быть оптимистичным; независимый holdout проверяет обобщение на ранее не использованных при выборе данных. Оптимизм — ожидаемая тенденция, но разница не обязана быть положительной на каждом эксперименте.

## Уровень C — nested CV и регуляризация

- Nested CV: внешний цикл **{c['outer_folds']}** фолдов, внутренний **{c['inner_folds']}** фолдов; оценка AP по внешним фолдам: {', '.join(fmt(v) for v in c['outer_scores'])}.
- Средняя независимая nested CV AP: **{fmt(c['nested_mean_ap'])}** (std={fmt(c['nested_std_ap'])}).
- Обычный лучший GridSearchCV AP: **{fmt(c['ordinary_grid_cv_ap'])}**.
- **Оценка оптимизма** (ordinary best CV − nested CV): **{fmt(c['optimism_gap'])}**. Может быть отрицательна из-за вариации выборок; это не математически гарантированная положительная величина.

### Grid Search / Random Search / Bayesian (Optuna)

| Метод | Кандидатов / итераций | CV fit | Лучший CV AP | Время, с |
|---|---:|---:|---:|---:|
{timings}

Все методы использовали одну метрику, одни development-фолды и сопоставимое число кандидатов. Время зависит от среды исполнения; число fit также приведено. Сравнение лучших CV AP служит характеристикой поискового процесса, а **не независимой оценкой** обобщения. Optuna использует TPE — последовательную модельно-ориентированную оптимизацию.

### L1 и L2

| Регуляризация | Лучший CV AP | Ненулевых коэффициентов | Параметры |
|---|---:|---:|---|
{reg}

L1 может обнулить некоторые веса и даёт разреженную модель; L2 сжимает веса, но почти никогда не зануляет их точно. Отбор силы регуляризации и сравнение велись на development CV, без использования теста.

### Практическая рекомендация

**Nested CV оправдана** на малых или средних датасетах, при большом поиске и когда нужна менее смещённая оценка качества выбранной *процедуры* обучения. **Часто не окупается** на очень больших датасетах и дорогих моделях, если уже есть достаточно большой независимый test/holdout и строгая дисциплина не использовать его для настройки. Однократный holdout после обучения по train-CV остаётся допустимым практическим решением.

**Ограничения:** AP зависит от распространённости положительного класса. Данные — ретроспективные, не содержат временной схемы наблюдений; оценка по случайным фолдам не доказывает пригодности для прогнозирования инсульта в будущем.
'''


def run_experiment(data_path: Path = DATA_PATH, output_dir: Path = Path('results'),
                   seed: int = 42, quick: bool = False) -> dict[str, Any]:
    x, y, data_info = get_data(data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=seed,
    )
    outer_folds, inner_folds = (3, 2) if quick else (5, 3)
    b_folds = 2 if quick else 5
    iterations = 4 if quick else 12
    baseline = make_pipeline(x_train)
    baseline.fit(x_train, y_train)
    base_test = holdout_metrics(baseline, x_test, y_test)

    # Level A: second holdout uses a different fixed split solely to show variability.
    x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=seed + 7,
    )
    baseline2 = make_pipeline(x_train_2)
    baseline2.fit(x_train_2, y_train_2)
    cv_a = cross_val_score(make_pipeline(x_train), x_train, y_train,
                          cv=cv_splits(5 if not quick else 3, seed),
                          scoring='average_precision', error_score='raise')
    a = {
        'seed_1': seed, 'seed_2': seed + 7,
        'holdout_ap_1': base_test['average_precision'],
        'holdout_ap_2': holdout_metrics(baseline2, x_test_2, y_test_2)['average_precision'],
        'cv_scores': cv_a.tolist(), 'cv_mean': float(np.mean(cv_a)),
        'cv_std': float(np.std(cv_a, ddof=1)),
        'cv_range': float(np.max(cv_a) - np.min(cv_a)),
    }

    # Level B: tune only on development/train. Test is never passed to the search.
    base_cv = cross_val_score(make_pipeline(x_train), x_train, y_train,
                              cv=cv_splits(b_folds, seed), scoring='average_precision',
                              error_score='raise')
    searches = run_searches(x_train, y_train, b_folds, seed, iterations)
    best_method = max(('grid', 'random'), key=lambda name: searches[name]['best_cv_ap'])
    tuned_test = holdout_metrics(searches[best_method]['_estimator'], x_test, y_test)
    b = {
        'baseline_cv_ap': float(np.mean(base_cv)),
        'grid_cv_ap': searches['grid']['best_cv_ap'],
        'grid_params': searches['grid']['best_params'],
        'random_cv_ap': searches['random']['best_cv_ap'],
        'random_params': searches['random']['best_params'],
        'winner': best_method,
        'baseline_test': base_test, 'tuned_test': tuned_test,
    }

    # Level C: inner folds choose params, outer folds score that choice.
    nested_scores = nested_cv(x_train, y_train, outer_folds, inner_folds, seed)
    optuna_stats = optuna_search(x_train, y_train, b_folds, seed, iterations)
    regularization = regularization_study(x_train, y_train, b_folds, seed)
    grid_cv_ap = searches['grid']['best_cv_ap']
    c = {
        'outer_folds': outer_folds, 'inner_folds': inner_folds,
        'outer_scores': nested_scores,
        'nested_mean_ap': float(np.mean(nested_scores)),
        'nested_std_ap': float(np.std(nested_scores, ddof=1)),
        'ordinary_grid_cv_ap': grid_cv_ap,
        'optimism_gap': float(grid_cv_ap - np.mean(nested_scores)),
        'methods': {name: {k: v for k, v in summary.items() if k != '_estimator'}
                    for name, summary in {**searches, 'optuna': optuna_stats}.items()},
        'regularization': regularization,
    }
    result = {
        'data': data_info,
        'split': {'seed': seed, 'train_rows': int(len(y_train)), 'test_rows': int(len(y_test)),
                  'train_positives': int(y_train.sum()), 'test_positives': int(y_test.sum())},
        'metric': 'average_precision',
        'A': a, 'B': b, 'C': c,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                             encoding='utf-8')
    (output_dir / 'REPORT.md').write_text(markdown_report(result), encoding='utf-8')
    print('Level A fold AP:', ', '.join(f'{v:.4f}' for v in a['cv_scores']))
    print('Level A mean/std:', f"{a['cv_mean']:.4f}", f"{a['cv_std']:.4f}")
    print('Level B baseline/best grid/best random/test AP:',
          *(f'{v:.4f}' for v in (b['baseline_cv_ap'], b['grid_cv_ap'],
                                 b['random_cv_ap'], b['tuned_test']['average_precision'])))
    print('Level C nested AP:', f"{c['nested_mean_ap']:.4f}",
          'optimism gap:', f"{c['optimism_gap']:+.4f}")
    print('Reports:', output_dir / 'metrics.json', output_dir / 'REPORT.md')
    return result


def main() -> None:
    p = argparse.ArgumentParser(description='Practical work 12: CV, hyperparameter search and nested CV')
    p.add_argument('--data', type=Path, default=DATA_PATH)
    p.add_argument('--output-dir', type=Path, default=Path('results'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--quick', action='store_true', help='Smaller budgets for a smoke test, not a full report')
    args = p.parse_args()
    run_experiment(args.data, args.output_dir, args.seed, args.quick)


if __name__ == '__main__':
    main()
