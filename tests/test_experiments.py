import sys
from pathlib import Path
import pytest
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.experiment_service import (
    load_dataset,
    run_classification_experiment,
    run_regression_experiment,
    run_clustering_experiment,
    run_parameter_sweep
)

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
