import os
import sys
import tempfile
import pytest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Override DB path for tests before importing core modules
temp_dir = tempfile.mkdtemp()
temp_db = Path(temp_dir) / "test_autolab.db"
os.environ["AUTOLAB_TEST_DB"] = str(temp_db)

import core.config
core.config.DB_PATH = temp_db

from core.database import (
    init_db, create_workspace, get_workspace, list_workspaces,
    update_workspace, delete_workspace, add_document, list_documents,
    save_experiment_plan, get_latest_plan, save_experiment_run,
    get_experiment_run, list_experiment_runs
)

def test_database_lifecycle():
    init_db()

    # 1. Create Workspace
    ws_id = create_workspace(
        title="Iris Benchmark",
        research_question="Can Random Forest outperform Logistic Regression?",
        hypothesis="Random Forest will achieve higher F1 score on Iris dataset.",
        research_area="Machine Learning",
        objective="Compare multi-class classifiers",
        constraints="Train size 80%"
    )
    assert ws_id > 0

    # 2. Get Workspace
    ws = get_workspace(ws_id)
    assert ws is not None
    assert ws["title"] == "Iris Benchmark"
    assert ws["research_question"] == "Can Random Forest outperform Logistic Regression?"

    # 3. Update Workspace
    update_workspace(ws_id, objective="Updated objective")
    ws_updated = get_workspace(ws_id)
    assert ws_updated["objective"] == "Updated objective"

    # 4. Add Document
    doc_id = add_document(ws_id, "iris_study.pdf", "pdf", "/tmp/iris.pdf", chunk_count=12)
    assert doc_id > 0
    docs = list_documents(ws_id)
    assert len(docs) == 1
    assert docs[0]["filename"] == "iris_study.pdf"

    # 5. Save Experiment Plan
    plan_dict = {
        "objective": "Compare models",
        "testable_hypothesis": "RF > LR",
        "independent_variables": ["Classifier type"],
        "dependent_variables": ["Accuracy", "F1"],
        "controlled_variables": ["Dataset split 80/20"],
        "dataset_and_required_columns": "Iris (4 features, 1 target)",
        "methodology": "Train models and evaluate on test set",
        "baseline": "Logistic Regression",
        "evaluation_metrics": ["accuracy", "f1_macro"],
        "expected_outputs": "Confusion matrix and classification report",
        "risks_and_limitations": "Small dataset size (150 samples)",
        "next_steps": ["Execute runner", "Inspect confusion matrix"],
        "evidence_citations": [],
        "assumptions": ["Default hyperparameters"],
        "is_grounded_in_sources": False
    }
    plan_id = save_experiment_plan(ws_id, plan_dict)
    assert plan_id > 0

    latest_plan = get_latest_plan(ws_id)
    assert latest_plan is not None
    assert latest_plan["plan"]["baseline"] == "Logistic Regression"

    # 6. Save Experiment Run
    run_id = save_experiment_run(
        workspace_id=ws_id,
        dataset_name="iris",
        experiment_type="classification",
        config_dict={"model": "RandomForestClassifier", "n_estimators": 100},
        metrics_dict={"accuracy": 0.9667, "f1_macro": 0.9667},
        plan_id=plan_id,
        interpretation="Model demonstrated high accuracy.",
        next_recommendation="Perform hyperparameter sweep on n_estimators."
    )
    assert run_id > 0

    run = get_experiment_run(run_id)
    assert run is not None
    assert run["metrics"]["accuracy"] == 0.9667

    runs = list_experiment_runs(ws_id)
    assert len(runs) == 1

    # 7. Delete Workspace (cascades)
    delete_workspace(ws_id)
    assert get_workspace(ws_id) is None
    assert len(list_documents(ws_id)) == 0
