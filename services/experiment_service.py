import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_iris, load_diabetes, make_classification, make_regression
from sklearn.model_selection import (
    GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
)
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    precision_score, recall_score, f1_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, silhouette_score
)
import plotly.express as px
import plotly.graph_objects as go
from core.config import DEFAULT_RANDOM_SEED
from core.schemas import ExperimentConfig

CLASSIFICATION_MODELS = [
    "Random Forest",
    "Logistic Regression",
    "HistGradientBoosting",
    "Support Vector Machine",
]

REGRESSION_MODELS = [
    "Random Forest Regressor",
    "Linear Regression",
    "HistGradientBoosting Regressor",
]

# Biometric columns where <= 0 is physiologically impossible in raw clinical units
_CLINICAL_ZERO_AS_NAN_COLS = {"glucose", "bloodpressure", "skinthickness", "insulin", "bmi"}


class ClinicalZeroCleaner(BaseEstimator, TransformerMixin):
    """
    Replaces physiologically impossible <= 0 entries in raw clinical biometric columns with NaN.
    Fits purely on training data and leaves standardized/unrelated datasets untouched.
    """

    def fit(self, X: pd.DataFrame, y: Any = None):
        self.columns_ = list(X.columns) if isinstance(X, pd.DataFrame) else [f"f{i}" for i in range(X.shape[1])]
        self.zero_cols_: List[Any] = []
        if isinstance(X, pd.DataFrame):
            for col in self.columns_:
                if str(col).strip().lower() not in _CLINICAL_ZERO_AS_NAN_COLS:
                    continue
                vals = pd.to_numeric(X[col], errors="coerce").dropna()
                # Only treat 0 as missing when column uses non-negative raw clinical units (never standardized floats)
                if not vals.empty and (vals >= 0).all() and vals.max() > 5.0 and (vals == 0).any():
                    self.zero_cols_.append(col)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.columns_)
        for col in self.zero_cols_:
            s = pd.to_numeric(df[col], errors="coerce")
            df[col] = s.where(s > 0, np.nan)
        return df

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

def build_preprocessor(
    X: pd.DataFrame,
    use_power_transform: bool = False,
    impute_numeric: bool = True,
    scale_numeric: bool = True,
) -> Tuple[ColumnTransformer, List[str]]:
    """Builds a leak-free imputation, encoding, and scaling preprocessor for mixed features."""
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    transformers = []
    if num_cols:
        num_steps = []
        if impute_numeric:
            num_steps.append(('imputer', SimpleImputer(strategy='median', add_indicator=True)))
        if use_power_transform:
            num_steps.append(('scaler', PowerTransformer(method='yeo-johnson', standardize=True)))
        elif scale_numeric:
            num_steps.append(('scaler', StandardScaler()))
        num_pipe = Pipeline(num_steps) if num_steps else 'passthrough'
        transformers.append(('num', num_pipe, num_cols))

    if cat_cols:
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', cat_pipe, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, verbose_feature_names_out=False)
    return preprocessor, num_cols + cat_cols


def _resolve_imbalance_weight(y_train: pd.Series, model_type: str) -> Optional[str]:
    """Enables balanced class weighting on training folds when class imbalance >= 1.5:1."""
    _, counts = np.unique(y_train.dropna(), return_counts=True)
    if len(counts) < 2 or counts.min() == 0:
        return None
    if (counts.max() / counts.min()) < 1.5:
        return None
    return "balanced_subsample" if model_type == "Random Forest" else "balanced"


def _clean_feature_names(pipeline: Pipeline, fallback_names: List[str]) -> List[str]:
    try:
        names = list(pipeline.named_steps['preprocessor'].get_feature_names_out())
        return [str(n) for n in names]
    except Exception:
        return list(fallback_names)


def _extract_feature_importance(
    pipeline: Pipeline,
    step_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    fallback_names: List[str],
    random_state: int,
    scoring: str,
) -> Dict[str, float]:
    """Extracts model-native importances, normalized linear magnitudes, or training-set permutation importances."""
    estimator = pipeline.named_steps[step_name]
    feat_names = _clean_feature_names(pipeline, fallback_names)

    if hasattr(estimator, 'feature_importances_'):
        vals = np.asarray(estimator.feature_importances_, dtype=float)
        return {name: round(float(val), 6) for name, val in zip(feat_names, vals)}

    if hasattr(estimator, 'coef_'):
        coef = np.asarray(estimator.coef_, dtype=float)
        mag = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
        total = float(mag.sum()) or 1.0
        norm = mag / total
        return {name: round(float(val), 6) for name, val in zip(feat_names, norm)}

    try:
        perm = permutation_importance(
            pipeline, X_train, y_train, n_repeats=3, random_state=random_state, scoring=scoring, n_jobs=1
        )
        raw_names = list(X_train.columns)
        return {name: round(max(0.0, float(val)), 6) for name, val in zip(raw_names, perm.importances_mean)}
    except Exception:
        return {}


def _compute_roc_auc(estimator: Pipeline, X_eval: pd.DataFrame, y_eval: pd.Series, labels: List[Any]) -> Optional[float]:
    try:
        if hasattr(estimator, 'predict_proba'):
            probs = estimator.predict_proba(X_eval)
            if len(labels) == 2:
                pos_idx = list(estimator.classes_).index(labels[1]) if labels[1] in estimator.classes_ else 1
                return round(float(roc_auc_score(y_eval, probs[:, pos_idx])), 4)
            return round(float(roc_auc_score(y_eval, probs, multi_class='ovr', average='weighted', labels=estimator.classes_)), 4)
        if hasattr(estimator, 'decision_function') and len(labels) == 2:
            scores = estimator.decision_function(X_eval)
            return round(float(roc_auc_score(y_eval, scores)), 4)
    except Exception:
        return None
    return None


def _build_classification_estimator_and_grid(
    model_type: str,
    X: pd.DataFrame,
    y_train: pd.Series,
    class_weight: Optional[str],
    random_state: int,
    params: Dict[str, Any],
) -> Tuple[Pipeline, Dict[str, Any], bool]:
    use_power = (model_type == "Logistic Regression")
    is_hgb = (model_type == "HistGradientBoosting")
    preprocessor, feat_cols = build_preprocessor(
        X,
        use_power_transform=use_power,
        impute_numeric=not is_hgb,
        scale_numeric=(model_type == "Support Vector Machine"),
    )

    if model_type == "Logistic Regression":
        clf = LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=params.get("max_iter", 2000),
            class_weight=params.get("class_weight", class_weight),
            random_state=random_state,
        )
        grid = {"classifier__C": [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]}
        use_grid_exact = True
    elif model_type == "Random Forest":
        clf = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 180),
            max_depth=params.get("max_depth", 6),
            min_samples_leaf=params.get("min_samples_leaf", 4),
            min_samples_split=params.get("min_samples_split", 2),
            max_features=params.get("max_features", "sqrt"),
            class_weight=params.get("class_weight", class_weight),
            random_state=random_state,
            n_jobs=1,
        )
        grid = {
            "classifier__n_estimators": [120, 200],
            "classifier__max_depth": [4, 5, 6, 8],
            "classifier__min_samples_leaf": [3, 5, 8, 12],
            "classifier__min_samples_split": [2, 6, 10],
            "classifier__max_features": ["sqrt", 0.5, 0.7],
        }
        use_grid_exact = False
    elif model_type == "HistGradientBoosting":
        clf = HistGradientBoostingClassifier(
            max_iter=params.get("max_iter", 120),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 4),
            min_samples_leaf=params.get("min_samples_leaf", 20),
            l2_regularization=params.get("l2_regularization", 1.0),
            class_weight=params.get("class_weight", class_weight),
            random_state=random_state,
        )
        grid = {
            "classifier__max_iter": [80, 120, 180],
            "classifier__learning_rate": [0.02, 0.03, 0.05, 0.08],
            "classifier__max_depth": [3, 4, 5],
            "classifier__min_samples_leaf": [15, 20, 30],
            "classifier__l2_regularization": [0.5, 1.0, 3.0, 8.0],
        }
        use_grid_exact = False
    elif model_type == "Support Vector Machine":
        base_svc = SVC(
            C=params.get("C", 1.0),
            gamma=params.get("gamma", "scale"),
            kernel=params.get("kernel", "rbf"),
            class_weight=params.get("class_weight", class_weight),
            random_state=random_state,
        )
        n_classes = int(y_train.nunique())
        if n_classes <= 2:
            clf = base_svc
            grid = {
                "classifier__C": [0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
                "classifier__gamma": ["scale", 0.01, 0.05, 0.1],
            }
        else:
            min_cls = int(y_train.value_counts().min())
            clf = CalibratedClassifierCV(base_svc, cv=min(3, min_cls), ensemble=False)
            grid = {
                "classifier__estimator__C": [0.1, 0.5, 1.0, 2.0, 5.0],
                "classifier__estimator__gamma": ["scale", 0.01, 0.05, 0.1],
            }
        use_grid_exact = False
    else:
        raise ValueError(f"Unsupported classification model: {model_type}")

    pipeline = Pipeline([
        ('cleaner', ClinicalZeroCleaner()),
        ('preprocessor', preprocessor),
        ('classifier', clf),
    ])
    return pipeline, grid, use_grid_exact


def run_classification_experiment(df: pd.DataFrame, target_col: str, model_type: str = "Random Forest",
                                  test_size: float = 0.2, random_state: int = DEFAULT_RANDOM_SEED,
                                  model_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Safe, reproducible, CV-tuned supervised classification pipeline without test leakage."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    valid_mask = df[target_col].notna()
    clean_df = df.loc[valid_mask]
    X = clean_df.drop(columns=[target_col])
    y = clean_df[target_col]

    if X.empty:
        raise ValueError("No feature columns available after removing target.")

    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        raise ValueError(f"Target column must have at least 2 distinct classes, found {len(unique_classes)}.")

    stratify_y = y if (y.value_counts().min() >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_y
    )

    params = dict(model_params or {})
    class_weight = _resolve_imbalance_weight(y_train, model_type)
    pipeline, param_grid, use_exact_grid = _build_classification_estimator_and_grid(
        model_type, X_train, y_train, class_weight, random_state, params
    )

    min_train_cls = int(y_train.value_counts().min())
    n_splits = min(5, min_train_cls)
    cv_scoring = "roc_auc" if (model_type == "Support Vector Machine" and len(unique_classes) == 2) else "balanced_accuracy"

    t_start = time.perf_counter()
    tuned_params: Dict[str, Any] = {}
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    cv_auc_mean: Optional[float] = None

    if not params and n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        if use_exact_grid:
            search = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring=cv_scoring, n_jobs=4)
        else:
            total_combos = int(np.prod([len(v) for v in param_grid.values()]))
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=min(12, total_combos),
                cv=cv,
                scoring=cv_scoring,
                n_jobs=4,
                random_state=random_state,
            )
        search.fit(X_train, y_train)
        fitted_pipeline = search.best_estimator_
        cv_mean = round(float(search.best_score_), 4)
        cv_std = round(float(search.cv_results_["std_test_score"][search.best_index_]), 4)
        tuned_params = {
            k.replace("classifier__estimator__", "").replace("classifier__", ""): v
            for k, v in search.best_params_.items()
        }
    else:
        fitted_pipeline = pipeline.fit(X_train, y_train)
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            try:
                scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=cv_scoring, n_jobs=4)
                cv_mean = round(float(scores.mean()), 4)
                cv_std = round(float(scores.std()), 4)
            except Exception:
                pass

    if cv_scoring == "roc_auc" and cv_mean is not None:
        cv_auc_mean = cv_mean
    elif n_splits >= 2:
        try:
            cv_auc_scores = cross_val_score(
                fitted_pipeline,
                X_train,
                y_train,
                cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state),
                scoring="roc_auc" if len(unique_classes) == 2 else "roc_auc_ovr_weighted",
                n_jobs=4,
            )
            cv_auc_mean = round(float(cv_auc_scores.mean()), 4)
        except Exception:
            pass

    training_time_s = round(time.perf_counter() - t_start, 3)

    y_pred = fitted_pipeline.predict(X_test)
    y_train_pred = fitted_pipeline.predict(X_train)

    acc = float(accuracy_score(y_test, y_pred))
    train_acc = float(accuracy_score(y_train, y_train_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
    rec = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
    f1_w = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
    f1_m = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
    roc_auc = _compute_roc_auc(fitted_pipeline, X_test, y_test, sorted(list(set(unique_classes.tolist()))))

    labels = sorted(list(set(unique_classes.tolist())))
    str_labels = [str(c) for c in labels]
    cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()

    report_dict = classification_report(y_test, y_pred, labels=labels, target_names=str_labels, output_dict=True, zero_division=0)
    per_class_metrics = {
        lbl: {
            "precision": round(float(report_dict[lbl]["precision"]), 4),
            "recall": round(float(report_dict[lbl]["recall"]), 4),
            "f1_score": round(float(report_dict[lbl]["f1-score"]), 4),
            "support": int(report_dict[lbl]["support"]),
        }
        for lbl in str_labels
        if lbl in report_dict
    }

    metrics_payload: Dict[str, Any] = {
        "accuracy": round(acc, 4),
        "precision_weighted": round(prec, 4),
        "recall_weighted": round(rec, 4),
        "f1_weighted": round(f1_w, 4),
        "f1_macro": round(f1_m, 4),
        "balanced_accuracy": round(bal_acc, 4),
    }
    if roc_auc is not None:
        metrics_payload["roc_auc"] = roc_auc
    if len(labels) == 2:
        pos_lbl = str_labels[1]
        pos_stats = per_class_metrics.get(pos_lbl, {})
        metrics_payload["positive_class_recall"] = pos_stats.get("recall")
        metrics_payload["positive_class_precision"] = pos_stats.get("precision")
        metrics_payload["positive_class_f1"] = pos_stats.get("f1_score")
        metrics_payload["false_negatives"] = int(cm[1][0])
        metrics_payload["false_positives"] = int(cm[0][1])
    if cv_mean is not None:
        metrics_payload["cv_score_mean"] = cv_mean
        metrics_payload["cv_score_std"] = cv_std
    if cv_auc_mean is not None:
        metrics_payload["cv_roc_auc_mean"] = cv_auc_mean

    metrics_payload["train_accuracy"] = round(train_acc, 4)
    metrics_payload["generalization_gap"] = round(train_acc - acc, 4)
    metrics_payload["training_time_seconds"] = training_time_s
    metrics_payload["classes"] = str_labels
    metrics_payload["confusion_matrix"] = cm
    metrics_payload["per_class_metrics"] = per_class_metrics

    feature_importance = _extract_feature_importance(
        fitted_pipeline, 'classifier', X_train, y_train, list(X.columns), random_state, cv_scoring
    )

    return {
        "status": "success",
        "experiment_type": "classification",
        "model_name": model_type,
        "metrics": metrics_payload,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "target_column": target_col,
        "classes": str_labels,
        "confusion_matrix": cm,
        "per_class_metrics": per_class_metrics,
        "feature_importance": feature_importance,
        "y_true": [str(v) for v in y_test.tolist()[:30]],
        "y_pred": [str(v) for v in y_pred.tolist()[:30]],
        "config": {
            "model_type": model_type,
            "target_column": target_col,
            "test_size": test_size,
            "random_state": random_state,
            "params": params,
            "tuned_params": tuned_params,
            "class_weight": class_weight,
            "cv_folds": n_splits if n_splits >= 2 else 0,
            "cv_scoring": cv_scoring if n_splits >= 2 else None,
        },
    }

def run_regression_experiment(df: pd.DataFrame, target_col: str, model_type: str = "Random Forest Regressor",
                              test_size: float = 0.2, random_state: int = DEFAULT_RANDOM_SEED,
                              model_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Safe, reproducible, CV-tuned supervised regression pipeline."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    clean_df = df.loc[df[target_col].notna()]
    X = clean_df.drop(columns=[target_col])
    y = clean_df[target_col]

    if not np.issubdtype(y.dtype, np.number):
        raise ValueError(f"Target column '{target_col}' must be numeric for regression.")

    if X.empty:
        raise ValueError("No feature columns available after removing target.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    params = dict(model_params or {})
    is_hgb = (model_type == "HistGradientBoosting Regressor")
    preprocessor, feat_cols = build_preprocessor(
        X,
        use_power_transform=False,
        impute_numeric=not is_hgb,
        scale_numeric=(model_type == "Linear Regression"),
    )

    if model_type == "Linear Regression":
        reg = Ridge(alpha=params.get("alpha", 1.0), random_state=random_state)
        grid = {"regressor__alpha": [0.01, 0.1, 1.0, 5.0, 10.0, 25.0]}
        exact_grid = True
    elif model_type == "Random Forest Regressor":
        reg = RandomForestRegressor(
            n_estimators=params.get("n_estimators", 150),
            max_depth=params.get("max_depth", 6),
            min_samples_leaf=params.get("min_samples_leaf", 2),
            max_features=params.get("max_features", "sqrt"),
            random_state=random_state,
            n_jobs=1,
        )
        grid = {
            "regressor__n_estimators": [150, 250],
            "regressor__max_depth": [4, 6, 8, None],
            "regressor__min_samples_leaf": [2, 4, 8],
            "regressor__max_features": ["sqrt", 0.6, 1.0],
        }
        exact_grid = False
    elif model_type == "HistGradientBoosting Regressor":
        reg = HistGradientBoostingRegressor(
            max_iter=params.get("max_iter", 120),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 4),
            min_samples_leaf=params.get("min_samples_leaf", 15),
            l2_regularization=params.get("l2_regularization", 1.0),
            random_state=random_state,
        )
        grid = {
            "regressor__max_iter": [80, 120, 180],
            "regressor__learning_rate": [0.03, 0.05, 0.08],
            "regressor__max_depth": [3, 4, 5],
            "regressor__min_samples_leaf": [10, 15, 25],
            "regressor__l2_regularization": [0.5, 1.0, 3.0],
        }
        exact_grid = False
    else:
        raise ValueError(f"Unsupported regression model: {model_type}")

    pipeline = Pipeline([
        ('cleaner', ClinicalZeroCleaner()),
        ('preprocessor', preprocessor),
        ('regressor', reg),
    ])

    n_splits = min(5, len(X_train))
    t_start = time.perf_counter()
    tuned_params: Dict[str, Any] = {}
    cv_r2_mean: Optional[float] = None
    cv_r2_std: Optional[float] = None

    if not params and n_splits >= 2:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        if exact_grid:
            search = GridSearchCV(pipeline, param_grid=grid, cv=cv, scoring="r2", n_jobs=4)
        else:
            total_combos = int(np.prod([len(v) for v in grid.values()]))
            search = RandomizedSearchCV(
                pipeline, param_distributions=grid, n_iter=min(12, total_combos),
                cv=cv, scoring="r2", n_jobs=4, random_state=random_state
            )
        search.fit(X_train, y_train)
        fitted_pipeline = search.best_estimator_
        cv_r2_mean = round(float(search.best_score_), 4)
        cv_r2_std = round(float(search.cv_results_["std_test_score"][search.best_index_]), 4)
        tuned_params = {k.replace("regressor__", ""): v for k, v in search.best_params_.items()}
    else:
        fitted_pipeline = pipeline.fit(X_train, y_train)
        if n_splits >= 2:
            try:
                scores = cross_val_score(
                    pipeline, X_train, y_train,
                    cv=KFold(n_splits=n_splits, shuffle=True, random_state=random_state),
                    scoring="r2", n_jobs=4
                )
                cv_r2_mean = round(float(scores.mean()), 4)
                cv_r2_std = round(float(scores.std()), 4)
            except Exception:
                pass

    training_time_s = round(time.perf_counter() - t_start, 3)

    y_pred = fitted_pipeline.predict(X_test)
    y_train_pred = fitted_pipeline.predict(X_train)

    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))
    train_r2 = float(r2_score(y_train, y_train_pred))

    feature_importance = _extract_feature_importance(
        fitted_pipeline, 'regressor', X_train, y_train, list(X.columns), random_state, "r2"
    )
    residuals = (y_test - y_pred).tolist()

    metrics_payload: Dict[str, Any] = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2_score": round(r2, 4),
        "train_r2": round(train_r2, 4),
        "generalization_gap": round(train_r2 - r2, 4),
        "training_time_seconds": training_time_s,
    }
    if cv_r2_mean is not None:
        metrics_payload["cv_r2_mean"] = cv_r2_mean
        metrics_payload["cv_r2_std"] = cv_r2_std

    return {
        "status": "success",
        "experiment_type": "regression",
        "model_name": model_type,
        "metrics": metrics_payload,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "target_column": target_col,
        "feature_importance": feature_importance,
        "y_true": [float(v) for v in y_test.tolist()[:40]],
        "y_pred": [float(v) for v in y_pred.tolist()[:40]],
        "residuals": [float(v) for v in residuals[:40]],
        "config": {
            "model_type": model_type,
            "target_column": target_col,
            "test_size": test_size,
            "random_state": random_state,
            "params": params,
            "tuned_params": tuned_params,
            "cv_folds": n_splits if n_splits >= 2 else 0,
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
    is_classification = (model_type in CLASSIFICATION_MODELS)

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
