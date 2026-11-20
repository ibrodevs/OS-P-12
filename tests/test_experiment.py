from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_val_score

from experiment import get_data, make_pipeline, run_experiment


def make_sample(tmp_path: Path) -> Path:
    x, y = make_classification(n_samples=240, n_features=3, n_informative=2,
                               n_redundant=0, weights=[0.8, 0.2], random_state=2)
    df = pd.DataFrame(x, columns=['age', 'bmi', 'avg_glucose_level'])
    df['gender'] = np.where(df['age'] > 0, 'Female', 'Male')
    df.loc[df.index[::9], 'bmi'] = np.nan
    df['id'] = np.arange(len(df))
    df['stroke'] = y
    path = tmp_path / 'example.csv'
    df.to_csv(path, index=False)
    return path


def test_load_drops_identifier(tmp_path):
    x, y, info = get_data(make_sample(tmp_path))
    assert x.shape == (240, 4)
    assert 'id' not in x.columns
    assert info['positives'] == int(y.sum())


def test_pipeline_handles_missing_and_unseen_categories(tmp_path):
    x, y, _ = get_data(make_sample(tmp_path))
    model = make_pipeline(x)
    model.fit(x.iloc[:190], y.iloc[:190])
    other = x.iloc[190:].copy()
    other.loc[other.index[0], 'gender'] = 'Other'
    assert model.predict_proba(other).shape == (50, 2)


def test_repeated_stratified_cross_val_is_deterministic(tmp_path):
    x, y, _ = get_data(make_sample(tmp_path))
    cv = StratifiedKFold(3, shuffle=True, random_state=42)
    first = cross_val_score(make_pipeline(x), x, y, cv=cv, scoring='average_precision')
    second = cross_val_score(make_pipeline(x), x, y, cv=cv, scoring='average_precision')
    assert np.allclose(first, second)


def test_all_levels_smoke(tmp_path):
    r = run_experiment(make_sample(tmp_path), tmp_path / 'reports', quick=True)
    assert len(r['A']['cv_scores']) == 3
    assert len(r['C']['outer_scores']) == 3
    assert r['C']['methods']['grid']['tested_candidates'] == 12
    assert r['C']['methods']['random']['tested_candidates'] == 4
    assert r['C']['methods']['optuna']['tested_candidates'] == 4
    assert (tmp_path / 'reports' / 'metrics.json').is_file()
    assert (tmp_path / 'reports' / 'REPORT.md').is_file()
    assert 0 <= r['B']['tuned_test']['average_precision'] <= 1


def test_invalid_target_is_rejected(tmp_path):
    path = make_sample(tmp_path)
    frame = pd.read_csv(path)
    frame['stroke'] = 0
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='both classes'):
        get_data(path)
