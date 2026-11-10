from typing import Optional, Dict, Any, List
from core.schemas import ExperimentPlanSchema, CitationItem
from services.retrieval_service import retrieve_relevant_chunks
from services.llm_service import is_gemini_configured, generate_plan_llm

def build_evidence_context(workspace_id: int, query: str, top_k: int = 5) -> str:
    """Retrieves and formats evidence chunks into a clean prompt context."""
    chunks = retrieve_relevant_chunks(workspace_id, query=query, top_k=top_k)
    if not chunks:
        return ""

    context_lines = []
    for c in chunks:
        context_lines.append(
            f"--- EVIDENCE ITEM (Source: {c.source_filename}, Page: {c.page_number}, Chunk: {c.chunk_index}, Similarity: {c.similarity_score}) ---\n"
            f"{c.content}\n"
        )
    return "\n".join(context_lines)

def create_experiment_plan(workspace: Dict[str, Any]) -> ExperimentPlanSchema:
    """Creates a grounded experiment plan using Gemini or structured deterministic template."""
    ws_id = workspace["id"]
    query = workspace["research_question"]
    evidence_text = build_evidence_context(ws_id, query)
    has_evidence = bool(evidence_text.strip())

    configured, _ = is_gemini_configured()
    if configured:
        return generate_plan_llm(
            research_title=workspace["title"],
            research_question=workspace["research_question"],
            hypothesis=workspace.get("hypothesis", ""),
            research_area=workspace.get("research_area", "Data Science"),
            objective=workspace.get("objective", ""),
            constraints=workspace.get("constraints", ""),
            evidence_context=evidence_text
        )

    # Deterministic fallback plan if Gemini API key is missing
    citations: List[CitationItem] = []
    if has_evidence:
        # Pull real citations from the retrieved evidence chunks
        raw_chunks = retrieve_relevant_chunks(ws_id, query=query, top_k=2)
        for rc in raw_chunks:
            citations.append(CitationItem(
                source_file=rc.source_filename,
                page_or_chunk=f"Page {rc.page_number}",
                factual_note=rc.content[:140] + "..."
            ))

    return ExperimentPlanSchema(
        objective=workspace.get("objective") or f"Investigate {workspace['research_question']}",
        testable_hypothesis=workspace.get("hypothesis") or f"Model performance will vary significantly based on feature configuration.",
        independent_variables=["Classifier algorithm (Random Forest vs Logistic Regression)", "Feature subsets"],
        dependent_variables=["Test Set Accuracy", "F1-Score (Weighted)", "Confusion Matrix Misclassifications"],
        controlled_variables=["Random seed (42)", "Train/test split ratio (80/20)", "Z-score scaling pipeline"],
        dataset_and_required_columns="Botanical measurements: sepal length, sepal width, petal length, petal width, target species class.",
        methodology="1. Ingest dataset and verify column types.\n2. Apply median imputation and standard scaling.\n3. Train baseline linear model and non-linear ensemble.\n4. Measure classification metrics on held-out test split.",
        baseline="Multinomial Logistic Regression (C=1.0)",
        evaluation_metrics=["Accuracy", "Weighted F1-score", "Precision", "Recall"],
        expected_outputs="Tabular classification report, confusion matrix heatmap, feature importance bar chart.",
        risks_and_limitations="Compact sample size may cause sampling variance across folds. Potential overlap in non-linearly separable classes.",
        next_steps=["Execute classification workflow in 'Run Experiment'.", "Review confusion matrix.", "Perform hyperparameter sweep."],
        evidence_citations=citations,
        assumptions=[
            "Data generated from independent and identically distributed (i.i.d.) observations.",
            "Missing values occur at random and can be imputed."
        ],
        is_grounded_in_sources=has_evidence
    )
