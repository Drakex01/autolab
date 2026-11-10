from typing import Dict, Any, List, Optional
from core.schemas import InterpretationResult, NextExperimentRecommendation
from services.llm_service import is_gemini_configured, interpret_results_llm, recommend_next_experiments_llm
from services.retrieval_service import retrieve_relevant_chunks

def analyze_and_interpret_run(
    workspace: Dict[str, Any],
    run_record: Dict[str, Any],
    plan_dict: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generates rigorous scientific interpretation and follow-up recommendations for a run."""
    ws_id = workspace["id"]
    rq = workspace["research_question"]
    hyp = workspace.get("hypothesis", "")

    exp_type = run_record["experiment_type"]
    model_name = run_record["config"].get("model_type", run_record.get("model_name", "Model"))
    metrics = run_record["metrics"]
    config = run_record["config"]

    configured, _ = is_gemini_configured()
    if configured:
        # 1. Interpret
        interp = interpret_results_llm(
            research_question=rq,
            hypothesis=hyp,
            experiment_type=exp_type,
            model_name=model_name,
            config=config,
            metrics=metrics,
            plan_context=str(plan_dict.get("objective") if plan_dict else "")
        )

        # 2. Recommendations
        # Retrieve literature context for grounding recommendation
        evidence_chunks = retrieve_relevant_chunks(ws_id, query=rq, top_k=3)
        ev_text = "\n".join([f"[{c.source_filename} p.{c.page_number}]: {c.content}" for c in evidence_chunks])

        recs = recommend_next_experiments_llm(
            research_question=rq,
            hypothesis=hyp,
            current_model=model_name,
            current_metrics=metrics,
            interpretation_summary=interp.summary,
            evidence_context=ev_text
        )

        return {
            "interpretation": interp.model_dump(),
            "recommendations": [r.model_dump() for r in recs]
        }

    # Deterministic fallback when API key is not configured
    # Calculate factual summary based on actual experiment numbers
    if exp_type == "classification":
        acc = metrics.get("accuracy", 0.0)
        f1 = metrics.get("f1_weighted", 0.0)
        status = "Supported" if acc >= 0.90 else "Inconclusive"
        summary = (
            f"The {model_name} model achieved an empirical test accuracy of {acc*100:.1f}% "
            f"and a weighted F1-score of {f1:.4f} on the held-out test split."
        )
        what_suggests = (
            f"High test accuracy ({acc*100:.1f}%) suggests that the input feature set contains "
            f"strong discriminant signals for distinguishing between target classes."
        )
        reasons = [
            f"Effective feature separability in botanical morphology (e.g. petal dimensions).",
            f"Ensemble voting in {model_name} reduces individual tree variance and decision noise.",
            f"Standard scaling and stratified train-test splitting preserved class distributions."
        ]
        recs = [
            NextExperimentRecommendation(
                title=f"{model_name} Hyperparameter Sweep",
                what_to_change="Vary ensemble size n_estimators across [10, 25, 50, 100, 200]",
                what_to_keep_constant="Fixed random seed (42), test split ratio (0.2), feature scaling",
                what_to_measure="Weighted F1-score and generalization gap between train and test accuracy",
                why_useful="Determines the minimum number of estimators needed before asymptotic performance plateaus, optimizing compute efficiency.",
                expected_support_or_weakening="Performance stabilizing above n=50 will confirm sample efficiency of tree ensembles.",
                can_run_predefined=True,
                recommended_runner_config={
                    "experiment_type": "Parameter Sweep",
                    "model_choice": model_name,
                    "sweep_param": "n_estimators"
                }
            ),
            NextExperimentRecommendation(
                title="Linear vs Non-Linear Model Contrast",
                what_to_change="Switch algorithm from Random Forest to Logistic Regression (or vice-versa)",
                what_to_keep_constant="Exact same train/test split index, random state (42), evaluation metrics",
                what_to_measure="Class-by-class confusion matrix misclassifications on borderline samples",
                why_useful="Tests whether non-linear boundary flexibilities provide statistically significant gains over linear hyperplanes.",
                expected_support_or_weakening="Higher F1 on overlapping classes directly supports non-linear hypothesis.",
                can_run_predefined=True,
                recommended_runner_config={
                    "experiment_type": "Classification",
                    "model_choice": "Logistic Regression" if "Random Forest" in model_name else "Random Forest"
                }
            )
        ]

    elif exp_type == "regression":
        rmse = metrics.get("rmse", 0.0)
        r2 = metrics.get("r2_score", 0.0)
        status = "Supported" if r2 > 0.4 else "Inconclusive"
        summary = (
            f"The {model_name} regression model achieved an RMSE of {rmse:.4f} and R² of {r2:.4f} "
            f"on the held-out evaluation split."
        )
        what_suggests = (
            f"An R² score of {r2:.4f} indicates that the model accounts for approximately "
            f"{max(0.0, r2)*100:.1f}% of the observed variance in the target variable."
        )
        reasons = [
            "Continuous features capture linear and non-linear trends in the progression target.",
            "Z-score normalization improved gradient stability across disparate physical scales."
        ]
        recs = [
            NextExperimentRecommendation(
                title="Regularization & Feature Depth Sweep",
                what_to_change="Sweep max_depth across [2, 4, 6, 8, 12, 16]",
                what_to_keep_constant="Dataset split, target column, evaluation metrics (RMSE, R²)",
                what_to_measure="Out-of-sample RMSE to identify onset of overfitting",
                why_useful="Balances bias-variance tradeoff in high-dimensional continuous estimation.",
                expected_support_or_weakening="Lower test RMSE at moderate depth supports controlled complexity.",
                can_run_predefined=True,
                recommended_runner_config={
                    "experiment_type": "Parameter Sweep",
                    "model_choice": model_name,
                    "sweep_param": "max_depth"
                }
            )
        ]

    else:
        # Clustering or parameter sweep
        summary = f"Workflow completed for {model_name}."
        what_suggests = "Observed distributions conform to standard algorithmic clustering / sweep behavior."
        status = "Inconclusive"
        reasons = ["Sample topology reflects selected geometric distance metric."]
        recs = [
            NextExperimentRecommendation(
                title="Cluster Count Sensitivity Evaluation",
                what_to_change="Vary cluster count k from 2 to 6",
                what_to_keep_constant="Numeric features, random seed, PCA projection space",
                what_to_measure="Silhouette coefficient score",
                why_useful="Identifies the natural clustering inflection point (elbow method).",
                expected_support_or_weakening="Peak silhouette score indicates optimal structural partition.",
                can_run_predefined=True
            )
        ]

    interp = InterpretationResult(
        summary=summary,
        what_results_suggest=what_suggests,
        hypothesis_status=status,
        possible_reasons=reasons,
        limitations=[
            "Sample size limitation: Metrics computed on a single partition without nested cross-validation.",
            "Absence of external domain validation or clinical test sets.",
            "Correlation does not imply causal biological or physical mechanisms."
        ]
    )

    return {
        "interpretation": interp.model_dump(),
        "recommendations": [r.model_dump() for r in recs]
    }
