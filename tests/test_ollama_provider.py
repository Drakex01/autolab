import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.schemas import (
    CitationItem,
    ExperimentPlanSchema,
    FollowUpRecommendations,
    InterpretationResult,
    NextExperimentRecommendation,
    RetrievedEvidence,
)
from services import analysis_service, llm_service, planning_service


def _valid_plan_payload(source_file: str = "healthcare-13-00037-v2.pdf") -> dict:
    return {
        "objective": "Compare Random Forest and Logistic Regression for diabetes classification.",
        "testable_hypothesis": "Random Forest will achieve higher weighted F1 than Logistic Regression.",
        "independent_variables": ["Classifier algorithm"],
        "dependent_variables": ["Accuracy", "Weighted F1-score"],
        "controlled_variables": ["80/20 split", "Random seed 42"],
        "dataset_and_required_columns": "diabetes.csv with numeric predictors and binary Outcome target.",
        "methodology": "Impute missing values, standardize features, train LR and RF on 80/20 split.",
        "baseline": "Logistic Regression (C=1.0)",
        "evaluation_metrics": ["accuracy", "f1_weighted"],
        "expected_outputs": "Metrics table and confusion matrix.",
        "risks_and_limitations": "Single train-test split on compact sample size.",
        "next_steps": ["Run Logistic Regression", "Run Random Forest"],
        "evidence_citations": [
            {
                "source_file": source_file,
                "page_or_chunk": "Page 1",
                "factual_note": "Ensemble tree methods often capture non-linear feature interactions.",
            }
        ],
        "assumptions": ["Observations are i.i.d."],
        "is_grounded_in_sources": True,
    }


def _mock_http_response(status_code: int = 200, json_data: dict = None, text: str = ""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text or (json.dumps(json_data) if json_data is not None else "")
    resp.json.return_value = json_data if json_data is not None else {}
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(f"HTTP {status_code}")
    return resp


def test_ollama_connectivity_and_model_availability():
    tags_resp = _mock_http_response(200, {"models": [{"name": "qwen2.5:7b-instruct"}]})
    ps_resp = _mock_http_response(
        200,
        {"models": [{"name": "qwen2.5:7b-instruct", "size": 5000000000, "size_vram": 3200000000, "context_length": 4096}]},
    )

    with patch.object(llm_service.requests, "get", side_effect=[tags_resp, ps_resp]):
        status = llm_service.check_ollama_status(model="qwen2.5:7b-instruct")

    assert status["connected"] is True
    assert status["model_available"] is True
    assert status["running_models"][0]["vram_mb"] > 0
    assert status["running_models"][0]["cpu_ram_mb"] > 0

    # Model not installed locally
    missing_tags = _mock_http_response(200, {"models": [{"name": "qwen2.5:3b-instruct"}]})
    with patch.object(llm_service.requests, "get", return_value=missing_tags):
        missing_status = llm_service.check_ollama_status(model="qwen2.5:7b-instruct")

    assert missing_status["connected"] is True
    assert missing_status["model_available"] is False
    assert "ollama pull qwen2.5:7b-instruct" in missing_status["error"]


def test_ollama_unavailable_fails_fast_and_never_switches_to_gemini():
    with (
        patch.object(llm_service.requests, "post", side_effect=requests.ConnectionError("Connection refused")),
        patch.object(llm_service, "call_gemini_with_retry") as mock_gemini,
    ):
        with pytest.raises(llm_service.OllamaUnavailableError) as exc_info:
            llm_service.generate_plan_llm(
                research_title="T",
                research_question="Q",
                hypothesis="H",
                research_area="DS",
                objective="O",
                constraints="C",
                evidence_context="",
                provider="ollama",
            )
    assert "not reachable" in str(exc_info.value).lower()
    assert mock_gemini.call_count == 0


def test_ollama_model_not_found_raises_explicit_error():
    not_found_resp = _mock_http_response(404, {"error": "model 'qwen2.5:7b-instruct' not found"})
    with patch.object(llm_service.requests, "post", return_value=not_found_resp):
        with pytest.raises(llm_service.OllamaModelNotFoundError) as exc_info:
            llm_service.interpret_results_llm(
                research_question="Q",
                hypothesis="H",
                experiment_type="classification",
                model_name="Random Forest",
                config={"test_size": 0.2},
                metrics={"accuracy": 0.8},
                provider="ollama",
            )
    assert "qwen2.5:7b-instruct" in str(exc_info.value)


def test_ollama_structured_plan_validation_and_invalid_json_retries():
    # Attempt 1: invalid JSON -> Attempt 2: empty required planning arrays -> Attempt 3: valid complete plan
    invalid_resp = _mock_http_response(200, {"message": {"content": "{not valid json"}})
    empty_lists_payload = _valid_plan_payload()
    empty_lists_payload["independent_variables"] = []
    empty_lists_payload["evaluation_metrics"] = []
    empty_lists_resp = _mock_http_response(200, {"message": {"content": json.dumps(empty_lists_payload)}})
    valid_resp = _mock_http_response(200, {"message": {"content": json.dumps(_valid_plan_payload())}})

    with patch.object(llm_service.requests, "post", side_effect=[invalid_resp, empty_lists_resp, valid_resp]) as mock_post:
        plan = llm_service.generate_plan_llm(
            research_title="Diabetes",
            research_question="Can RF beat LR?",
            hypothesis="RF > LR",
            research_area="Data Science",
            objective="Compare classifiers",
            constraints="Seed 42",
            evidence_context="Source: healthcare-13-00037-v2.pdf",
            provider="ollama",
            model="qwen2.5:7b-instruct",
        )

    assert isinstance(plan, ExperimentPlanSchema)
    assert mock_post.call_count == 3
    sent_payload = mock_post.call_args_list[0].kwargs["json"]
    assert sent_payload["model"] == "qwen2.5:7b-instruct"
    assert sent_payload["options"]["num_ctx"] == 4096
    assert sent_payload["options"]["temperature"] == 0.2
    assert isinstance(sent_payload["format"], dict)
    assert "independent_variables" in sent_payload["format"]["required"]
    assert "evidence_citations" in sent_payload["format"]["required"]
    assert sent_payload["format"]["properties"]["independent_variables"]["minItems"] == 1
    assert sent_payload["format"]["properties"]["evaluation_metrics"]["minItems"] == 1

    # Persistent malformed JSON must fail loudly instead of accepting broken plan
    with patch.object(llm_service.requests, "post", return_value=invalid_resp):
        with pytest.raises(ValueError) as val_exc:
            llm_service.generate_plan_llm(
                research_title="T",
                research_question="Q",
                hypothesis="H",
                research_area="DS",
                objective="O",
                constraints="C",
                evidence_context="",
                provider="ollama",
            )
    assert "Failed to obtain valid ExperimentPlanSchema JSON" in str(val_exc.value)

    # Missing required fields must also fail loudly rather than passing partial schema
    incomplete_resp = _mock_http_response(200, {"message": {"content": json.dumps({"objective": "Only one field"})}})
    with patch.object(llm_service.requests, "post", return_value=incomplete_resp):
        with pytest.raises(ValueError) as inc_exc:
            llm_service.generate_plan_llm(
                research_title="T",
                research_question="Q",
                hypothesis="H",
                research_area="DS",
                objective="O",
                constraints="C",
                evidence_context="",
                provider="ollama",
            )
    assert "Failed to obtain valid ExperimentPlanSchema JSON" in str(inc_exc.value)


def test_rag_evidence_citation_sanitization_drops_fabricated_sources():
    chunks = [
        RetrievedEvidence(
            source_filename="healthcare-13-00037-v2.pdf",
            page_number=3,
            chunk_index=4,
            content="Random Forest improved diabetes screening recall in clinical tabular evaluations.",
            similarity_score=0.78,
        )
    ]
    payload = _valid_plan_payload(source_file="healthcare-13-00037-v2.pdf")
    payload["evidence_citations"].append(
        {"source_file": "invented_paper_2099.pdf", "page_or_chunk": "Page 99", "factual_note": "Fake note"}
    )
    mock_plan = ExperimentPlanSchema.model_validate(payload)

    with (
        patch.object(planning_service, "retrieve_relevant_chunks", return_value=chunks),
        patch.object(planning_service, "generate_plan_llm", return_value=mock_plan),
    ):
        ws = {"id": 4, "title": "Diabetes", "research_question": "Does RF beat LR?"}
        final_plan = planning_service.create_experiment_plan(ws, provider="ollama")

    assert final_plan.is_grounded_in_sources is True
    assert len(final_plan.evidence_citations) == 1
    assert final_plan.evidence_citations[0].source_file == "healthcare-13-00037-v2.pdf"


def test_offline_ollama_comparative_analysis_and_unavailable_fallback():
    ws = {
        "id": 4,
        "research_question": "Does Random Forest perform better than Logistic Regression?",
        "hypothesis": "Random Forest beats Logistic Regression on diabetes.csv.",
    }
    rf_run = {
        "id": 5,
        "workspace_id": 4,
        "dataset_name": "diabetes.csv",
        "experiment_type": "classification",
        "config": {"model_type": "Random Forest", "test_size": 0.2, "random_state": 42},
        "metrics": {"accuracy": 0.7597, "precision_weighted": 0.7546, "recall_weighted": 0.7597, "f1_weighted": 0.7555},
    }
    lr_run = {
        "id": 6,
        "workspace_id": 4,
        "dataset_name": "diabetes.csv",
        "experiment_type": "classification",
        "config": {"model_type": "Logistic Regression", "test_size": 0.2, "random_state": 42},
        "metrics": {"accuracy": 0.7143, "precision_weighted": 0.7065, "recall_weighted": 0.7143, "f1_weighted": 0.7084},
    }

    interp_json = json.dumps({
        "summary": "Random Forest (75.97%) outperformed Logistic Regression (71.43%) by 4.54 percentage points.",
        "what_results_suggest": "Non-linear splits improved classification accuracy on this split.",
        "hypothesis_status": "Supported",
        "possible_reasons": ["Tree ensembles capture feature interactions."],
        "limitations": ["Single 80/20 split; class-level metrics not recorded."],
    })
    recs_json = json.dumps({
        "recommendations": [
            {
                "title": "Compare Logistic Regression vs Random Forest",
                "what_to_change": "Switch between Logistic Regression and Random Forest",
                "what_to_keep_constant": "Same split",
                "what_to_measure": "Accuracy",
                "why_useful": "Redundant check",
                "expected_support_or_weakening": "Diff",
                "can_run_predefined": True,
                "recommended_runner_config": {"experiment_type": "Classification", "model_choice": "Logistic Regression"},
            },
            {
                "title": "Logistic Regression C Hyperparameter Sweep",
                "what_to_change": "Sweep C in [0.01, 0.1, 1.0, 10.0]",
                "what_to_keep_constant": "Same split and seed 42",
                "what_to_measure": "Weighted F1",
                "why_useful": "Tune linear baseline regularization",
                "expected_support_or_weakening": "Higher F1 narrows gap",
                "can_run_predefined": True,
                "recommended_runner_config": {"experiment_type": "Parameter Sweep", "model_choice": "Logistic Regression", "sweep_param": "C"},
            },
        ]
    })

    tags_ok = _mock_http_response(200, {"models": [{"name": "qwen2.5:7b-instruct"}]})
    ps_ok = _mock_http_response(200, {"models": []})
    chat_interp = _mock_http_response(200, {"message": {"content": interp_json}})
    chat_recs = _mock_http_response(200, {"message": {"content": recs_json}})

    # Confirm works completely offline with empty GEMINI_API_KEY and zero Gemini calls
    with (
        patch.object(llm_service, "GEMINI_API_KEY", ""),
        patch.object(llm_service, "call_gemini_with_retry") as mock_gemini,
        patch.object(llm_service.requests, "get", side_effect=[tags_ok, ps_ok]),
        patch.object(llm_service.requests, "post", side_effect=[chat_interp, chat_recs]),
        patch.object(analysis_service, "retrieve_relevant_chunks", return_value=[]),
    ):
        out = analysis_service.analyze_and_interpret_run(
            ws, rf_run, workspace_runs=[rf_run, lr_run], provider="ollama", model="qwen2.5:7b-instruct"
        )

    assert mock_gemini.call_count == 0
    assert out["source"] == "ollama"
    assert out["analysis_mode"] == "comparative_ollama"
    assert out["interpretation"]["hypothesis_status"] == "Supported"
    assert "qwen2.5:7b-instruct" in out["interpretation"]["source_label"]
    # Duplicate RF vs LR comparison filtered out
    assert [r["title"] for r in out["recommendations"]] == ["Logistic Regression C Hyperparameter Sweep"]

    # When Ollama is stopped -> clear fallback without calling Gemini
    with (
        patch.object(llm_service, "call_gemini_with_retry") as mock_gemini_2,
        patch.object(llm_service.requests, "get", side_effect=requests.ConnectionError("Stopped")),
    ):
        fb = analysis_service.analyze_and_interpret_run(
            ws, rf_run, workspace_runs=[rf_run, lr_run], provider="ollama"
        )
    assert mock_gemini_2.call_count == 0
    assert fb["ollama_unavailable"] is True
    assert fb["source"] == "rule_based"
    assert fb["analysis_mode"] == "comparative_rule_based"
    assert fb["interpretation"]["hypothesis_status"] == "Supported"


def test_gemini_optional_provider_still_selectable():
    with (
        patch.object(llm_service, "GEMINI_API_KEY", "test-key"),
        patch.object(llm_service, "call_gemini_with_retry", return_value=json.dumps(_valid_plan_payload("study.pdf"))) as mock_gem,
    ):
        plan = llm_service.generate_plan_llm(
            research_title="T",
            research_question="Q",
            hypothesis="H",
            research_area="DS",
            objective="O",
            constraints="C",
            evidence_context="study.pdf",
            provider="gemini",
        )
    assert mock_gem.call_count == 1
    assert isinstance(plan, ExperimentPlanSchema)
