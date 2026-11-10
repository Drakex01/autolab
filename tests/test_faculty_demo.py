import sys
from pathlib import Path
import json
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.database import (
    init_db, create_workspace, get_workspace, add_document,
    save_experiment_plan, save_experiment_run, get_experiment_run,
    list_experiment_runs
)
from services.document_service import extract_text_from_file
from services.chunking_service import split_text_into_chunks
from services.retrieval_service import add_chunks_to_index, retrieve_relevant_chunks
from services.experiment_service import (
    load_dataset, run_classification_experiment, run_parameter_sweep
)
from services.planning_service import create_experiment_plan
from services.analysis_service import analyze_and_interpret_run
from services.history_service import format_runs_for_comparison

def test_complete_faculty_demo_workflow():
    init_db()

    # Step 1: Create Faculty Demo Workspace
    ws_id = create_workspace(
        title="Iris Species Predictability Benchmark",
        research_question="Can a Random Forest classifier predict Iris species with higher accuracy and F1-score than Logistic Regression on standardized botanical measurements?",
        hypothesis="Nonlinear decision boundaries in Random Forest will yield superior multi-class classification metrics compared to linear Logistic Regression.",
        research_area="Machine Learning / Data Science",
        objective="Empirically compare Random Forest vs Logistic Regression on the Fisher Iris dataset under reproducible train/test splits.",
        constraints="Fixed 80/20 train-test split, seed 42, no external data leakage."
    )
    ws = get_workspace(ws_id)
    assert ws is not None
    assert "Random Forest" in ws["research_question"]

    # Step 2: Upload and Ingest Research PDF
    sample_pdf = PROJECT_ROOT / "data" / "sample_docs" / "iris_classification_study.pdf"
    assert sample_pdf.exists()

    pages_data = extract_text_from_file(sample_pdf, "iris_classification_study.pdf")
    assert len(pages_data) == 2  # 2 pages

    doc_id = add_document(
        workspace_id=ws_id,
        filename="iris_classification_study.pdf",
        file_type="pdf",
        file_path=str(sample_pdf),
        chunk_count=0
    )
    chunks = split_text_into_chunks(
        pages_data=pages_data,
        source_filename="iris_classification_study.pdf",
        document_id=doc_id,
        workspace_id=ws_id,
        chunk_size=800,
        chunk_overlap=120
    )
    assert len(chunks) >= 2
    add_chunks_to_index(ws_id, chunks)

    # Step 3: Retrieve Evidence from Knowledge Library
    evidence = retrieve_relevant_chunks(ws_id, query="Random Forest and Logistic Regression comparison", top_k=3)
    assert len(evidence) > 0
    assert any("iris_classification_study.pdf" == e.source_filename for e in evidence)
    assert all(e.page_number in [1, 2] for e in evidence)

    # Step 4: Create Grounded Experiment Plan
    plan_schema = create_experiment_plan(ws)
    assert plan_schema.objective is not None
    assert plan_schema.testable_hypothesis is not None
    assert len(plan_schema.evaluation_metrics) > 0
    plan_id = save_experiment_plan(ws_id, plan_schema.model_dump())
    assert plan_id > 0

    # Step 5: Load Iris Dataset
    df, target_col = load_dataset("iris")
    assert not df.empty
    assert target_col == "species"

    # Step 6: Run Logistic Regression Baseline
    res_lr = run_classification_experiment(
        df=df, target_col=target_col, model_type="Logistic Regression",
        test_size=0.2, random_state=42
    )
    assert res_lr["status"] == "success"
    lr_acc = res_lr["metrics"]["accuracy"]
    lr_f1 = res_lr["metrics"]["f1_weighted"]
    assert lr_acc > 0.85

    run_lr_id = save_experiment_run(
        workspace_id=ws_id,
        dataset_name="iris",
        experiment_type="classification",
        config_dict=res_lr["config"],
        metrics_dict=res_lr["metrics"],
        plan_id=plan_id
    )
    assert run_lr_id > 0

    # Step 7: Run Random Forest Model
    res_rf = run_classification_experiment(
        df=df, target_col=target_col, model_type="Random Forest",
        test_size=0.2, random_state=42, model_params={"n_estimators": 100}
    )
    assert res_rf["status"] == "success"
    rf_acc = res_rf["metrics"]["accuracy"]
    rf_f1 = res_rf["metrics"]["f1_weighted"]
    assert rf_acc > 0.85

    run_rf_id = save_experiment_run(
        workspace_id=ws_id,
        dataset_name="iris",
        experiment_type="classification",
        config_dict=res_rf["config"],
        metrics_dict=res_rf["metrics"],
        plan_id=plan_id
    )
    assert run_rf_id > 0

    # Step 8: Analyze and Interpret Results
    rf_record = get_experiment_run(run_rf_id)
    analysis = analyze_and_interpret_run(ws, rf_record, plan_dict=plan_schema.model_dump())
    assert "interpretation" in analysis
    assert "recommendations" in analysis
    assert analysis["interpretation"]["hypothesis_status"] in ["Supported", "Not Supported", "Inconclusive"]
    assert len(analysis["recommendations"]) >= 1

    # Step 9: Execute Recommended Next Step (Parameter Sweep on n_estimators)
    sweep_res = run_parameter_sweep(
        df=df, target_col=target_col, model_type="Random Forest",
        param_name="n_estimators", param_values=[10, 25, 50, 100, 200],
        test_size=0.2, random_state=42
    )
    assert sweep_res["status"] == "success"
    assert len(sweep_res["sweep_data"]) == 5

    run_sweep_id = save_experiment_run(
        workspace_id=ws_id,
        dataset_name="iris",
        experiment_type="parameter_sweep",
        config_dict=sweep_res["config"],
        metrics_dict={"sweep_steps": len(sweep_res["sweep_data"])},
        plan_id=plan_id
    )
    assert run_sweep_id > 0

    # Step 10: Verify Experiment History & Cross-Run Comparison
    all_runs = list_experiment_runs(workspace_id=ws_id)
    assert len(all_runs) >= 3
    df_history = format_runs_for_comparison(all_runs)
    assert not df_history.empty
    assert "Run ID" in df_history.columns
    assert "Model" in df_history.columns
