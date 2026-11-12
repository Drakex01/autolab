import sys
from pathlib import Path
import pytest
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.experiment_service import (
    ClinicalZeroCleaner,
    load_dataset,
    run_classification_experiment,
    run_regression_experiment,
    run_clustering_experiment,
    run_parameter_sweep
)
from services.history_service import format_runs_for_comparison
import numpy as np

def test_load_builtin_datasets():
    # Iris
    df_iris, target_iris = load_dataset("iris")
    assert not df_iris.empty
    assert target_iris == "species"
    assert "species" in df_iris.columns

    # Diabetes
    df_diab, target_diab = load_dataset("diabetes")
    assert not df_diab.empty
    assert target_diab == "target"

    # Synthetic
    df_clf, t_clf = load_dataset("synth_clf")
    assert len(df_clf) == 300
    assert t_clf == "target"

def test_classification_iris():
    df, target = load_dataset("iris")

    # Logistic Regression
    res_lr = run_classification_experiment(df, target_col=target, model_type="Logistic Regression")
    assert res_lr["status"] == "success"
    assert "accuracy" in res_lr["metrics"]
    assert 0.8 <= res_lr["metrics"]["accuracy"] <= 1.0
    assert len(res_lr["classes"]) == 3
    assert len(res_lr["confusion_matrix"]) == 3

    # Random Forest
    res_rf = run_classification_experiment(df, target_col=target, model_type="Random Forest")
    assert res_rf["status"] == "success"
    assert "accuracy" in res_rf["metrics"]
    assert 0.8 <= res_rf["metrics"]["accuracy"] <= 1.0
    assert len(res_rf["feature_importance"]) > 0

def test_regression_diabetes():
    df, target = load_dataset("diabetes")

    # Linear Regression
    res_lin = run_regression_experiment(df, target_col=target, model_type="Linear Regression")
    assert res_lin["status"] == "success"
    assert "rmse" in res_lin["metrics"]
    assert "r2_score" in res_lin["metrics"]

    # Random Forest Regressor
    res_rf = run_regression_experiment(df, target_col=target, model_type="Random Forest Regressor")
    assert res_rf["status"] == "success"
    assert "rmse" in res_rf["metrics"]
    assert len(res_rf["feature_importance"]) > 0

def test_clustering():
    df, _ = load_dataset("iris")
    res_km = run_clustering_experiment(df, n_clusters=3)
    assert res_km["status"] == "success"
    assert "silhouette_score" in res_km["metrics"]
    assert len(res_km["pca_data"]["x"]) == len(df)

def test_parameter_sweep():
    df, target = load_dataset("iris")
    sweep_res = run_parameter_sweep(
        df, target_col=target, model_type="Random Forest",
        param_name="n_estimators", param_values=[10, 50, 100]
    )
    assert sweep_res["status"] == "success"
    assert len(sweep_res["sweep_data"]) == 3


def test_clinical_zero_cleaner_and_standardized_safety():
    # Clinical biometric dataset: Glucose=0 and Insulin=0 must become NaN, Pregnancies=0 must stay 0
    df_clin = pd.DataFrame({
        "Pregnancies": [0, 2, 4, 1],
        "Glucose": [120, 0, 145, 98],
        "Insulin": [0, 85, 0, 110],
        "BMI": [28.4, 33.1, 0.0, 25.0],
    })
    cleaner = ClinicalZeroCleaner().fit(df_clin)
    transformed = cleaner.transform(df_clin)
    assert transformed["Pregnancies"].iloc[0] == 0
    assert pd.isna(transformed["Glucose"].iloc[1])
    assert pd.isna(transformed["Insulin"].iloc[0])
    assert pd.isna(transformed["BMI"].iloc[2])

    # Standardized regression dataset (e.g. sklearn load_diabetes has negative values in 'bmi'): must stay untouched
    df_std = pd.DataFrame({
        "bmi": [-0.04, 0.0, 0.06, -0.02],
        "s1": [0.01, -0.03, 0.04, 0.0],
    })
    cleaner_std = ClinicalZeroCleaner().fit(df_std)
    out_std = cleaner_std.transform(df_std)
    assert out_std["bmi"].isna().sum() == 0
    assert (out_std["bmi"] == df_std["bmi"]).all()


def test_improved_classification_algorithms_and_metrics():
    # Imbalanced clinical-style binary classification (2:1 imbalance + clinical zeros)
    rng = np.random.default_rng(42)
    n = 180
    outcome = np.array([0] * 120 + [1] * 60)
    glucose = np.where(outcome == 1, rng.normal(148, 22, n), rng.normal(105, 18, n)).clip(60, 220)
    bmi = np.where(outcome == 1, rng.normal(34, 5, n), rng.normal(27, 4.5, n)).clip(18, 55)
    insulin = rng.normal(120, 40, n).clip(20, 350)
    # Inject biological zeros
    glucose[0] = 0
    insulin[::4] = 0

    df = pd.DataFrame({
        "Pregnancies": rng.integers(0, 8, size=n),
        "Glucose": glucose,
        "BMI": bmi,
        "Insulin": insulin,
        "Outcome": outcome,
    })

    for model_name in ("Random Forest", "Logistic Regression", "HistGradientBoosting", "Support Vector Machine"):
        res = run_classification_experiment(df, target_col="Outcome", model_type=model_name, test_size=0.2, random_state=42)
        assert res["status"] == "success"
        m = res["metrics"]
        for req_key in (
            "accuracy",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
            "balanced_accuracy",
            "roc_auc",
            "positive_class_recall",
            "false_negatives",
            "cv_score_mean",
            "train_accuracy",
            "generalization_gap",
            "training_time_seconds",
            "per_class_metrics",
            "confusion_matrix",
            "classes",
        ):
            assert req_key in m, f"Missing {req_key} for {model_name}"
        assert res["config"]["class_weight"] in ("balanced", "balanced_subsample")
        assert res["config"]["cv_folds"] == 5
        assert m["roc_auc"] >= 0.75

        # History table flattening must exclude nested dicts/lists cleanly
        flat = format_runs_for_comparison([{
            "id": 1,
            "dataset_name": "synthetic_clin.csv",
            "experiment_type": "classification",
            "config": res["config"],
            "metrics": res["metrics"],
            "created_at": "2026-11-12T20:00:00",
        }])
        assert "Metric: confusion_matrix" not in flat.columns
        assert "Metric: per_class_metrics" not in flat.columns
        assert "Metric: accuracy" in flat.columns


def test_improved_regression_algorithms_and_diagnostics():
    df, target = load_dataset("diabetes")
    res_hgb = run_regression_experiment(df, target_col=target, model_type="HistGradientBoosting Regressor")
    assert res_hgb["status"] == "success"
    for req_key in ("mae", "rmse", "r2_score", "train_r2", "generalization_gap", "cv_r2_mean", "training_time_seconds"):
        assert req_key in res_hgb["metrics"]
    assert res_hgb["config"]["cv_folds"] == 5
