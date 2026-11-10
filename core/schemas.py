from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class WorkspaceCreate(BaseModel):
    title: str = Field(..., description="Research project title")
    research_question: str = Field(..., description="Core research question")
    hypothesis: str = Field("", description="Testable hypothesis")
    research_area: str = Field("Data Science", description="Discipline/field of study")
    objective: str = Field("", description="Specific goals and scope")
    constraints: str = Field("", description="Time, compute, or variable constraints")

class Workspace(WorkspaceCreate):
    id: int
    created_at: str
    updated_at: str

class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: int
    workspace_id: int
    source_filename: str
    page_number: int
    chunk_index: int
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RetrievedEvidence(BaseModel):
    source_filename: str
    page_number: int
    chunk_index: int
    content: str
    similarity_score: float

class CitationItem(BaseModel):
    source_file: str
    page_or_chunk: str
    factual_note: str

class ExperimentPlanSchema(BaseModel):
    objective: str = Field(..., description="Primary experimental objective")
    testable_hypothesis: str = Field(..., description="Formulated hypothesis to be tested")
    independent_variables: List[str] = Field(default_factory=list, description="Variables manipulated")
    dependent_variables: List[str] = Field(default_factory=list, description="Variables measured")
    controlled_variables: List[str] = Field(default_factory=list, description="Variables kept constant or assumptions")
    dataset_and_required_columns: str = Field(..., description="Expected dataset and key columns")
    methodology: str = Field(..., description="Step-by-step experimental procedure")
    baseline: str = Field(..., description="Baseline model or benchmark to compare against")
    evaluation_metrics: List[str] = Field(default_factory=list, description="Metrics used for assessment")
    expected_outputs: str = Field(..., description="Anticipated outputs and visualizations")
    risks_and_limitations: str = Field(..., description="Known risks, limitations, and potential threats to validity")
    next_steps: List[str] = Field(default_factory=list, description="Immediate next steps")
    evidence_citations: List[CitationItem] = Field(default_factory=list, description="Verified citations from uploaded documents")
    assumptions: List[str] = Field(default_factory=list, description="Assumptions made in this plan")
    is_grounded_in_sources: bool = Field(False, description="Whether plan is verified against uploaded literature")

class ExperimentConfig(BaseModel):
    dataset_name: str
    experiment_type: str  # 'classification', 'regression', 'clustering', 'parameter_sweep'
    model_name: str
    target_column: Optional[str] = None
    feature_columns: List[str] = Field(default_factory=list)
    test_size: float = 0.2
    random_state: int = 42
    params: Dict[str, Any] = Field(default_factory=dict)

class ExperimentMetrics(BaseModel):
    metrics: Dict[str, Any]
    train_size: int
    test_size: int
    feature_names: List[str]
    additional_data: Dict[str, Any] = Field(default_factory=dict)

class InterpretationResult(BaseModel):
    summary: str = Field(..., description="Plain-language concise summary of what was observed")
    what_results_suggest: str = Field(..., description="Direct takeaways from the numbers and charts")
    hypothesis_status: str = Field(..., description="Supported, Not Supported, or Inconclusive")
    possible_reasons: List[str] = Field(default_factory=list, description="Potential mechanisms or reasons for the outcome")
    limitations: List[str] = Field(default_factory=list, description="Caveats, sample limitations, risk of leakage, or non-causality notes")

class NextExperimentRecommendation(BaseModel):
    title: str = Field(..., description="Short name of suggested follow-up experiment")
    what_to_change: str = Field(..., description="Exact variable, model, or hyperparameter to alter")
    what_to_keep_constant: str = Field(..., description="Variables or conditions held identical")
    what_to_measure: str = Field(..., description="Evaluation metric or response to monitor")
    why_useful: str = Field(..., description="Scientific or engineering rationale for this next step")
    expected_support_or_weakening: str = Field(..., description="What outcome would support vs challenge the hypothesis")
    can_run_predefined: bool = Field(True, description="Whether this can be executed via the built-in runner")
    recommended_runner_config: Optional[Dict[str, Any]] = Field(default=None, description="Preset configuration for runner")

class FollowUpRecommendations(BaseModel):
    recommendations: List[NextExperimentRecommendation]
