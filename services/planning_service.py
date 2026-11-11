from typing import Any, Dict, List, Optional
from core.schemas import CitationItem, ExperimentPlanSchema
from core.schemas import RetrievedEvidence
from services.llm_service import (
    generate_plan_llm,
    get_active_provider_name,
    is_gemini_configured,
    is_llm_configured,
)
from services.retrieval_service import retrieve_relevant_chunks


def build_evidence_context(
    workspace_id: int,
    query: str,
    top_k: int = 5,
    chunks: Optional[List[RetrievedEvidence]] = None,
) -> str:
    """Retrieves and formats evidence chunks into a clean prompt context."""
    if chunks is None:
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


def _sanitize_citations(plan: ExperimentPlanSchema, retrieved_chunks: List[RetrievedEvidence]) -> ExperimentPlanSchema:
    """Ensures every citation in the plan maps to an actual retrieved document filename (never fabricated)."""
    if not retrieved_chunks:
        plan.evidence_citations = []
        plan.is_grounded_in_sources = False
        return plan

    valid_sources = {c.source_filename for c in retrieved_chunks}
    verified: List[CitationItem] = []
    for cit in plan.evidence_citations:
        src = cit.source_file.strip()
        # Allow exact match or basename match against retrieved sources
        matched = next((vs for vs in valid_sources if vs == src or vs.endswith(src) or src in vs), None)
        if matched:
            verified.append(
                CitationItem(
                    source_file=matched,
                    page_or_chunk=cit.page_or_chunk,
                    factual_note=cit.factual_note,
                )
            )
    plan.evidence_citations = verified
    plan.is_grounded_in_sources = len(verified) > 0
    return plan


def _build_template_plan(
    workspace: Dict[str, Any],
    retrieved_chunks: List[RetrievedEvidence],
) -> ExperimentPlanSchema:
    """Builds a deterministic methodological template when running without an LLM."""
    has_evidence = bool(retrieved_chunks)
    citations: List[CitationItem] = []
    for rc in retrieved_chunks[:2]:
        citations.append(
            CitationItem(
                source_file=rc.source_filename,
                page_or_chunk=f"Page {rc.page_number} (Chunk {rc.chunk_index})",
                factual_note=rc.content[:140] + "...",
            )
        )

    return ExperimentPlanSchema(
        objective=workspace.get("objective") or f"Investigate {workspace['research_question']}",
        testable_hypothesis=workspace.get("hypothesis") or "Model performance will vary significantly based on feature configuration.",
        independent_variables=["Classifier algorithm (Random Forest vs Logistic Regression)", "Feature subsets"],
        dependent_variables=["Test Set Accuracy", "F1-Score (Weighted)", "Confusion Matrix Misclassifications"],
        controlled_variables=["Random seed (42)", "Train/test split ratio (80/20)", "Z-score scaling pipeline"],
        dataset_and_required_columns="Tabular feature columns plus a categorical or numeric target column matching the research question.",
        methodology="1. Ingest dataset and verify column types.\n2. Apply median imputation and standard scaling.\n3. Train baseline linear model and non-linear ensemble.\n4. Measure evaluation metrics on held-out test split.",
        baseline="Multinomial Logistic Regression (C=1.0)",
        evaluation_metrics=["Accuracy", "Weighted F1-score", "Precision", "Recall"],
        expected_outputs="Tabular classification report, confusion matrix heatmap, feature importance bar chart.",
        risks_and_limitations="Compact sample size may cause sampling variance across folds. Single train/test split does not measure cross-validation variance.",
        next_steps=["Execute classification workflow in 'Run Experiment'.", "Review confusion matrix.", "Perform hyperparameter sweep."],
        evidence_citations=citations,
        assumptions=[
            "Data generated from independent and identically distributed (i.i.d.) observations.",
            "Missing values occur at random and can be imputed.",
        ],
        is_grounded_in_sources=has_evidence,
    )


def create_experiment_plan(
    workspace: Dict[str, Any],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    allow_template_fallback: bool = False,
    use_template_only: bool = False,
) -> ExperimentPlanSchema:
    """Creates a grounded experiment plan using the active LLM provider (Ollama default) or template fallback."""
    ws_id = workspace["id"]
    query = workspace["research_question"]
    retrieved_chunks = retrieve_relevant_chunks(ws_id, query=query, top_k=5)
    evidence_text = build_evidence_context(ws_id, query, chunks=retrieved_chunks)

    if use_template_only:
        return _build_template_plan(workspace, retrieved_chunks)

    chosen_provider = get_active_provider_name(provider)

    if chosen_provider == "gemini":
        configured, _ = is_gemini_configured()
        if not configured:
            return _build_template_plan(workspace, retrieved_chunks)
    elif allow_template_fallback:
        ok, _ = is_llm_configured(provider="ollama", model=model)
        if not ok:
            return _build_template_plan(workspace, retrieved_chunks)

    plan = generate_plan_llm(
        research_title=workspace["title"],
        research_question=workspace["research_question"],
        hypothesis=workspace.get("hypothesis", ""),
        research_area=workspace.get("research_area", "Data Science"),
        objective=workspace.get("objective", ""),
        constraints=workspace.get("constraints", ""),
        evidence_context=evidence_text,
        provider=chosen_provider,
        model=model,
    )
    return _sanitize_citations(plan, retrieved_chunks)
