# Практическая работа № 12 — Валидация и подбор гиперпараметров

**Выполнены уровни A, B и C.** Тема: честная оценка качества, `Pipeline`, `StratifiedKFold`, `GridSearchCV`, `RandomizedSearchCV`, nested CV, байесовская оптимизация `Optuna` и сравнение L1/L2.

## Данные

Используется тот же основной датасет курса о зарегистрированных инсультах: **Stroke Prediction Dataset**, 5 110 наблюдений, 12 исходных столбцов, положительный класс `stroke=1` встречается 249 раз (4,87%). Признак `id` исключён из обучения. В `bmi` пропуски, категориальный признак `smoking_status` может содержать категорию `Unknown`.

- Оригинальный датасет: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
- Открытая копия используемого CSV: https://github.com/chandanverma07/DataSets/blob/master/healthcare-dataset-stroke-data.csv
- Исходный CSV **не хранится** в репозитории. Команда `python download_data.py` скачивает открытую копию и проверяет её Git blob SHA-1. Для первого запуска нужен интернет; после скачивания `experiment.py` работает офлайн.

⚠️ Это учебная классификация ретроспективных данных, **не медицинский прогноз или диагностический инструмент**.

## Проверенные результаты

Полный опыт успешно выполнен в GitHub Actions. Читайте **[REPORT.md](REPORT.md)** (выводы по всем трём уровням) и **[metrics.json](metrics.json)** (точные значения метрик, параметры и времена). Строки `results/` формируются заново при каждом локальном запуске и сохраняются в Actions artifacts.

## Установка и запуск

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python download_data.py                  # один раз, скачает проверенный CSV
python experiment.py                   # полноценный опыт для всех уровней
python -m pytest -q                    # тесты
```

После выполнения формируются `results/metrics.json` (все значения) и `results/REPORT.md` (готовый отчёт для сдачи). Для короткой проверки: `python experiment.py --quick` (меньшее число фолдов/итераций; не использовать вместо полного отчёта).

## Схема эксперимента

1. **80% development/train, 20% test** — `train_test_split(..., stratify=y, random_state=42)`; test не подаётся ни в один поиск гиперпараметров. При 5 110 строках получается 4 088 train / 1 022 test.
2. **Предобработка внутри `Pipeline`** — медианная импутация + `StandardScaler` для числовых столбцов; импутация + `OneHotEncoder(handle_unknown='ignore')` для категориальных. Статистики каждого фолда обучаются только на его train-части.
3. **Метрика `average_precision` (AP)** вместо accuracy: класс 1 встречается менее чем в 5% случаев. Дополнительно на test выводятся recall, F1, balanced accuracy.
4. **Уровень A:** baseline Logistic Regression, holdout с двумя random_state, 5-fold stratified CV на development, среднее и выборочное стандартное отклонение.
5. **Уровень B:** GridSearchCV (12 вариантов) vs RandomizedSearchCV (12 итераций) на 5-fold stratified development CV. Выбор лучшего — только по CV; затем однократная независимая проверка на test. Параметры: `C`, `penalty`, `class_weight`.
6. **Уровень C:** внешняя 5-fold CV (независимая оценка всей процедуры Grid Search) + внутренняя 3-fold CV (подбор), **только на development**. Разница обычного лучшего CV AP и nested CV AP оценивает смещение выбора максимума. Grid, Random и Optuna сравниваются на одинаковых development-фолдах с бюджетом 12 вариантов/итераций; L1 и L2 сопоставляются по CV AP и числу ненулевых коэффициентов.

### Почему оценки могут отличаться

Обычный best-score поиска выбран как максимум нескольких шумных CV-оценок и обычно оптимистичен. Nested CV оценивает всю процедуру подбора на внешних валидационных фолдах. Test остаётся независимым, пока его не используют для выбора модели/порога. Разница обычной и вложенной CV может быть отрицательной на конкретных данных из-за дисперсии оценок — это не ошибка.

### Когда выбирать nested CV?

Nested CV полезна на небольших/средних датасетах при интенсивном подборе и когда требуется качественная оценка самой процедуры выбора. На больших данных с дорогими моделями и достаточно большим untouched test часто достаточно CV для подбора + единственной отложенной проверки на test.

## Структура

```text
OS-P-12/
├── experiment.py                       # Все три уровня + автоматический отчёт
├── download_data.py                    # Проверенная загрузка исходного CSV
├── data/healthcare-dataset-stroke-data.csv  # После скачивания (в .gitignore)
├── tests/test_experiment.py           # Проверки и smoke-test
├── .github/workflows/test.yml         # CI
├── requirements.txt
├── .gitignore
└── README.md
```

Время работы поиска зависит от компьютера. Все эксперименты воспроизводимы с фиксированным seed; между машинами небольшие отличия времени допустимы.
