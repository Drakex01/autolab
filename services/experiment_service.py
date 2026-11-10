import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.datasets import load_iris, load_diabetes, make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score, silhouette_score
)
import plotly.express as px
import plotly.graph_objects as go
from core.config import DEFAULT_RANDOM_SEED
from core.schemas import ExperimentConfig

def get_builtin_datasets() -> Dict[str, str]:
    return {
        "Iris Species (Classification)": "iris",
        "Diabetes Progression (Regression)": "diabetes",
        "Synthetic Binary Classification": "synth_clf",
        "Synthetic Continuous Regression": "synth_reg"
    }

def load_dataset(dataset_key: str, uploaded_file=None) -> Tuple[pd.DataFrame, Optional[str]]:
    """Loads a dataframe from a built-in identifier or uploaded CSV."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df, None

    if dataset_key == "iris":
        data = load_iris(as_frame=True)
        df = data.frame.copy()
        # Ensure target column is human-readable or named 'target'
        df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        df = df.drop(columns=['target'])
        return df, "species"

    elif dataset_key == "diabetes":
        data = load_diabetes(as_frame=True)
        df = data.frame.copy()
        return df, "target"

    elif dataset_key == "synth_clf":
        X, y = make_classification(
            n_samples=300, n_features=6, n_informative=4, n_redundant=1,
            random_state=DEFAULT_RANDOM_SEED
        )
        cols = [f"feature_{i+1}" for i in range(6)]
        df = pd.DataFrame(X, columns=cols)
        df['target'] = y
        return df, "target"

    elif dataset_key == "synth_reg":
        X, y = make_regression(
            n_samples=300, n_features=6, n_informative=4, noise=15.0,
            random_state=DEFAULT_RANDOM_SEED
        )
        cols = [f"feature_{i+1}" for i in range(6)]
        df = pd.DataFrame(X, columns=cols)
        df['target'] = y
        return df, "target"

    else:
        raise ValueError(f"Unknown dataset key: {dataset_key}")

def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str]]:
    """Builds a safe imputation and scaling preprocessor for mixed features."""
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    transformers = []
    if num_cols:
        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', num_pipe, num_cols))

    if cat_cols:
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', cat_pipe, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor, num_cols + cat_cols

def run_classification_experiment(df: pd.DataFrame, target_col: str, model_type: str = "Random Forest",
                                  test_size: float = 0.2, random_state: int = DEFAULT_RANDOM_SEED,
                                  model_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Safe, reproducible supervised classification pipeline."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    if X.empty:
        raise ValueError("No feature columns available after removing target.")

    # Validate target classes
    unique_classes = np.unique(y.dropna())
    if len(unique_classes) < 2:
        raise ValueError(f"Target column must have at least 2 distinct classes, found {len(unique_classes)}.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(unique_classes) < 10 else None
    )

    preprocessor, feat_cols = build_preprocessor(X)
    params = model_params or {}

    if model_type == "Logistic Regression":
        clf = LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=params.get("max_iter", 1000),
            random_state=random_state
        )
    elif model_type == "Random Forest":
        clf = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None),
            random_state=random_state
        )
    else:
        raise ValueError(f"Unsupported classification model: {model_type}")

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
    rec = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))

    labels = sorted(list(set(unique_classes.tolist())))
    cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()

    # Feature Importance (if Random Forest)
    feature_importance = {}
    if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
        importances = pipeline.named_steps['classifier'].feature_importances_
        # Match with feature names if possible
        try:
            num_names = X.select_dtypes(include=[np.number]).columns.tolist()
            feature_importance = {name: float(val) for name, val in zip(num_names, importances[:len(num_names)])}
        except Exception:
            pass

    return {
        "status": "success",
        "experiment_type": "classification",
        "model_name": model_type,
        "metrics": {
            "accuracy": round(acc, 4),
            "precision_weighted": round(prec, 4),
            "recall_weighted": round(rec, 4),
            "f1_weighted": round(f1, 4)
        },
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "target_column": target_col,
        "classes": [str(c) for c in labels],
        "confusion_matrix": cm,
        "feature_importance": feature_importance,
        "y_true": [str(v) for v in y_test.tolist()[:30]],
        "y_pred": [str(v) for v in y_pred.tolist()[:30]],
        "config": {
            "model_type": model_type,
            "test_size": test_size,
            "random_state": random_state,
            "params": params
        }
    }

def run_regression_experiment(df: pd.DataFrame, target_col: str, model_type: str = "Random Forest Regressor",
                              test_size: float = 0.2, random_state: int = DEFAULT_RANDOM_SEED,
                              model_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Safe, reproducible supervised regression pipeline."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    if not np.issubdtype(y.dtype, np.number):
        raise ValueError(f"Target column '{target_col}' must be numeric for regression.")

    if X.empty:
        raise ValueError("No feature columns available after removing target.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocessor, feat_cols = build_preprocessor(X)
    params = model_params or {}

    if model_type == "Linear Regression":
        reg = LinearRegression()
    elif model_type == "Random Forest Regressor":
        reg = RandomForestRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None),
            random_state=random_state
        )
    else:
        raise ValueError(f"Unsupported regression model: {model_type}")

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', reg)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Metrics
    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))

    feature_importance = {}
    if hasattr(pipeline.named_steps['regressor'], 'feature_importances_'):
        importances = pipeline.named_steps['regressor'].feature_importances_
        num_names = X.select_dtypes(include=[np.number]).columns.tolist()
        feature_importance = {name: float(val) for name, val in zip(num_names, importances[:len(num_names)])}

    residuals = (y_test - y_pred).tolist()

    return {
        "status": "success",
        "experiment_type": "regression",
        "model_name": model_type,
        "metrics": {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2_score": round(r2, 4)
        },
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "target_column": target_col,
        "feature_importance": feature_importance,
        "y_true": [float(v) for v in y_test.tolist()[:40]],
        "y_pred": [float(v) for v in y_pred.tolist()[:40]],
        "residuals": [float(v) for v in residuals[:40]],
        "config": {
            "model_type": model_type,
            "test_size": test_size,
            "random_state": random_state,
            "params": params
        }
    }

def run_clustering_experiment(df: pd.DataFrame, n_clusters: int = 3,
                              random_state: int = DEFAULT_RANDOM_SEED) -> Dict[str, Any]:
    """Unsupervised K-Means clustering with 2D PCA projection."""
    # Select numeric features
    X_num = df.select_dtypes(include=[np.number]).dropna()
    if X_num.shape[1] < 2:
        raise ValueError("At least 2 numeric features required for clustering.")
    if len(X_num) < n_clusters:
        raise ValueError(f"Dataset has fewer rows ({len(X_num)}) than requested clusters ({n_clusters}).")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Calculate silhouette score if valid
    sil_score = None
    if 1 < n_clusters < len(X_scaled):
        sil_score = float(silhouette_score(X_scaled, cluster_labels))

    # 2D PCA projection for visualization
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(X_scaled)

    return {
        "status": "success",
        "experiment_type": "clustering",
        "model_name": "K-Means",
        "metrics": {
            "n_clusters": n_clusters,
            "inertia": round(float(kmeans.inertia_), 2),
            "silhouette_score": round(sil_score, 4) if sil_score is not None else None,
            "pca_explained_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_]
        },
        "sample_count": len(X_num),
        "cluster_counts": {f"Cluster {k}": int(v) for k, v in zip(*np.unique(cluster_labels, return_counts=True))},
        "pca_data": {
            "x": [round(float(c[0]), 3) for c in coords],
            "y": [round(float(c[1]), 3) for c in coords],
            "cluster": [int(l) for l in cluster_labels]
        },
        "config": {
            "n_clusters": n_clusters,
            "random_state": random_state
        }
    }

def run_parameter_sweep(df: pd.DataFrame, target_col: str, model_type: str,
                        param_name: str, param_values: List[Any],
                        test_size: float = 0.2, random_state: int = DEFAULT_RANDOM_SEED) -> Dict[str, Any]:
    """Runs a controlled parameter sweep comparing a range of values for a single parameter."""
    sweep_results = []
    is_classification = (model_type in ["Random Forest", "Logistic Regression"])

    for val in param_values:
        params = {param_name: val}
        if is_classification:
            res = run_classification_experiment(
                df, target_col=target_col, model_type=model_type,
                test_size=test_size, random_state=random_state, model_params=params
            )
            score = res["metrics"]["f1_weighted"]
            metric_name = "F1-Score (Weighted)"
        else:
            res = run_regression_experiment(
                df, target_col=target_col, model_type=model_type,
                test_size=test_size, random_state=random_state, model_params=params
            )
            score = res["metrics"]["r2_score"]
            metric_name = "R² Score"

        sweep_results.append({
            "param_value": val,
            "score": score,
            "all_metrics": res["metrics"]
        })

    return {
        "status": "success",
        "experiment_type": "parameter_sweep",
        "model_name": model_type,
        "param_name": param_name,
        "metric_name": metric_name,
        "sweep_data": sweep_results,
        "config": {
            "model_type": model_type,
            "param_name": param_name,
            "param_values": param_values,
            "test_size": test_size,
            "random_state": random_state
        }
    }

# ----------------- Plotly Visualizations ----------------- #

def plot_confusion_matrix(cm: List[List[int]], classes: List[str]):
    fig = px.imshow(
        cm,
        x=classes,
        y=classes,
        labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
        color_continuous_scale="Blues",
        text_auto=True,
        title="Confusion Matrix"
    )
    fig.update_layout(height=450, margin=dict(l=40, r=40, t=50, b=40))
    return fig

def plot_regression_actual_vs_predicted(y_true: List[float], y_pred: List[float]):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred, mode='markers',
        marker=dict(size=8, color='#1E88E5', opacity=0.8),
        name='Predictions'
    ))
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines', line=dict(color='#E53935', dash='dash'),
        name='Ideal 1:1 Reference'
    ))
    fig.update_layout(
        title="Actual vs Predicted Values",
        xaxis_title="Actual Value",
        yaxis_title="Predicted Value",
        height=450,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

def plot_clustering_2d_pca(pca_data: Dict[str, Any]):
    df_pca = pd.DataFrame(pca_data)
    df_pca['cluster'] = df_pca['cluster'].astype(str)
    fig = px.scatter(
        df_pca, x='x', y='y', color='cluster',
        title="2D PCA Projection of Clusters",
        labels={'x': 'Principal Component 1', 'y': 'Principal Component 2', 'cluster': 'Cluster'},
        height=450
    )
    fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
    return fig

def plot_parameter_sweep(sweep_data: List[Dict[str, Any]], param_name: str, metric_name: str):
    df_sweep = pd.DataFrame({
        param_name: [d["param_value"] for d in sweep_data],
        metric_name: [d["score"] for d in sweep_data]
    })
    fig = px.line(
        df_sweep, x=param_name, y=metric_name, markers=True,
        title=f"Parameter Sweep: Effect of '{param_name}' on {metric_name}",
        height=420
    )
    fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
    return fig

def plot_feature_importance(feat_dict: Dict[str, float]):
    if not feat_dict:
        return None
    sorted_feats = sorted(feat_dict.items(), key=lambda x: x[1], reverse=True)
    df_fi = pd.DataFrame(sorted_feats, columns=["Feature", "Importance"])
    fig = px.bar(
        df_fi, x="Importance", y="Feature", orientation="h",
        title="Feature Importance",
        height=350,
        color="Importance",
        color_continuous_scale="Viridis"
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=40, r=40, t=50, b=40))
    return fig
