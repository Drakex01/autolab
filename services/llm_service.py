import json
import time
from typing import Tuple, Optional, List, Dict, Any
from core.config import GEMINI_API_KEY, GEMINI_MODEL
from core.schemas import (
    ExperimentPlanSchema,
    InterpretationResult,
    FollowUpRecommendations,
    NextExperimentRecommendation
)

def is_gemini_configured() -> Tuple[bool, str]:
    """Checks whether the Gemini API key is configured."""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        return False, "GEMINI_API_KEY is not set. Add your key to .env or environment variables."
    return True, f"Configured with model: {GEMINI_MODEL}"

def get_gemini_client():
    from google import genai
    return genai.Client(api_key=GEMINI_API_KEY)

def call_gemini_with_retry(prompt: str, system_instruction: str = "", temperature: float = 0.2, max_retries: int = 2) -> str:
    """Calls Gemini API with configurable retries and timeout."""
    configured, msg = is_gemini_configured()
    if not configured:
        raise ValueError(msg)

    from google import genai
    from google.genai import types

    client = get_gemini_client()
    config = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system_instruction if system_instruction else None,
        response_mime_type="application/json"
    )

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=config
            )
            if response and response.text:
                return response.text.strip()
            raise ValueError("Empty response received from Gemini.")
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
            else:
                break

    raise RuntimeError(f"Gemini API call failed after {max_retries + 1} attempts: {str(last_error)}")

def generate_plan_llm(
    research_title: str,
    research_question: str,
    hypothesis: str,
    research_area: str,
    objective: str,
    constraints: str,
    evidence_context: str
) -> ExperimentPlanSchema:
    """Generates a structured, evidence-aware scientific experiment plan."""
    system_instruction = (
        "You are AutoLab, a rigorous scientific experiment planning assistant. "
        "Your task is to generate a structured, testable, empirical experiment plan. "
        "Ground your plan in the provided literature evidence when available. "
        "Strict rule: Never fabricate citations or numbers. "
        "Clearly distinguish verified evidence from general assumptions. "
        "Respond strictly with valid JSON conforming to the requested schema."
    )

    prompt = f"""
Generate a structured experiment plan for the following research project:

PROJECT DETAILS:
- Title: {research_title}
- Research Question: {research_question}
- Hypothesis: {hypothesis or "None stated"}
- Research Area: {research_area}
- Specific Objective: {objective or "None stated"}
- Constraints: {constraints or "None stated"}

RETRIEVED LITERATURE EVIDENCE:
{evidence_context if evidence_context.strip() else "[No uploaded documents found. Ground your plan in general scientific methodology and clearly mark is_grounded_in_sources as false.]"}

JSON SCHEMA REQUIRED:
{{
  "objective": "Clear primary experimental objective",
  "testable_hypothesis": "Testable operationalized hypothesis",
  "independent_variables": ["List of independent variables to manipulate"],
  "dependent_variables": ["List of dependent variables to measure"],
  "controlled_variables": ["List of variables held constant"],
  "dataset_and_required_columns": "Expected dataset format, types, and columns",
  "methodology": "Step-by-step experimental execution protocol",
  "baseline": "Baseline benchmark or reference model",
  "evaluation_metrics": ["List of exact evaluation metrics, e.g. Accuracy, F1-macro, RMSE"],
  "expected_outputs": "Anticipated empirical outputs and charts",
  "risks_and_limitations": "Known experimental limitations, biases, or threats to validity",
  "next_steps": ["Immediate actionable execution steps"],
  "evidence_citations": [
    {{
      "source_file": "Exact filename from retrieved context, e.g. iris_study.pdf",
      "page_or_chunk": "Page X or Chunk Y",
      "factual_note": "Specific finding or metric cited from document"
    }}
  ],
  "assumptions": ["List of methodological assumptions made"],
  "is_grounded_in_sources": true/false
}}
"""
    raw_json = call_gemini_with_retry(prompt=prompt, system_instruction=system_instruction, temperature=0.2)
    # Parse and validate with Pydantic
    try:
        data = json.loads(raw_json)
        return ExperimentPlanSchema.model_validate(data)
    except Exception as e:
        # Fallback if wrapped in markdown block
        cleaned = raw_json.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned)
        return ExperimentPlanSchema.model_validate(data)

def interpret_results_llm(
    research_question: str,
    hypothesis: str,
    experiment_type: str,
    model_name: str,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    plan_context: Optional[str] = None
) -> InterpretationResult:
    """Interprets real experimental metrics and returns plain-language takeaways."""
    system_instruction = (
        "You are AutoLab, a rigorous scientific data analysis assistant. "
        "You are analyzing REAL empirical experiment metrics computed by a reproducible scikit-learn workflow. "
        "Strict rules: Never invent or alter measurements. "
        "Distinguish empirical observations from tentative interpretations. "
        "Never claim an experiment proves a hypothesis; state whether it supports, fails to support, or is inconclusive. "
        "Highlight caveats such as sample size, potential leakage, or non-causality. "
        "Respond strictly with valid JSON."
    )

    prompt = f"""
RESEARCH CONTEXT:
- Research Question: {research_question}
- Hypothesis: {hypothesis or "None stated"}
- Plan Context: {plan_context or "None"}

ACTUAL EXPERIMENTAL MEASUREMENTS:
- Experiment Type: {experiment_type}
- Model Executed: {model_name}
- Configuration: {json.dumps(config, indent=2)}
- Computed Metrics: {json.dumps(metrics, indent=2)}

JSON SCHEMA REQUIRED:
{{
  "summary": "Concise 2-3 sentence factual summary of what was observed",
  "what_results_suggest": "Key scientific and analytical takeaways from the numbers and charts",
  "hypothesis_status": "Supported" OR "Not Supported" OR "Inconclusive",
  "possible_reasons": ["List of 2-4 plausible methodological or structural reasons for this outcome"],
  "limitations": ["List of 2-3 caveats (sample size, data distribution, lack of external validation, etc.)"]
}}
"""
    raw_json = call_gemini_with_retry(prompt=prompt, system_instruction=system_instruction, temperature=0.2)
    try:
        data = json.loads(raw_json)
        return InterpretationResult.model_validate(data)
    except Exception:
        cleaned = raw_json.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned)
        return InterpretationResult.model_validate(data)

def recommend_next_experiments_llm(
    research_question: str,
    hypothesis: str,
    current_model: str,
    current_metrics: Dict[str, Any],
    interpretation_summary: str,
    evidence_context: str
) -> List[NextExperimentRecommendation]:
    """Generates 1 to 3 concrete, non-generic follow-up experiment recommendations."""
    system_instruction = (
        "You are AutoLab, an expert scientific experimentation advisor. "
        "Based on the completed experiment and evidence, suggest 1 to 3 concrete, reproducible follow-up experiments. "
        "Do NOT provide generic advice like 'collect more data'. "
        "Provide exact parameters, variables to change, variables to keep constant, and what to measure. "
        "Indicate whether the experiment can be executed in AutoLab's predefined runner (e.g. parameter sweep, model comparison, split ratio). "
        "Respond strictly with valid JSON conforming to the schema."
    )

    prompt = f"""
CURRENT EXPERIMENT:
- Research Question: {research_question}
- Hypothesis: {hypothesis}
- Model Tested: {current_model}
- Achieved Metrics: {json.dumps(current_metrics, indent=2)}
- Interpretation: {interpretation_summary}

EVIDENCE CONTEXT:
{evidence_context if evidence_context.strip() else "None"}

JSON SCHEMA REQUIRED:
{{
  "recommendations": [
    {{
      "title": "Short descriptive title, e.g. Random Forest Tree Count Sweep",
      "what_to_change": "Specific hyperparameter or baseline model to change",
      "what_to_keep_constant": "Exact variables, test split, and seed to keep identical",
      "what_to_measure": "Evaluation metric to observe, e.g. F1-macro or R2 score",
      "why_useful": "Scientific rationale for this follow-up investigation",
      "expected_support_or_weakening": "What numerical threshold would support vs weaken the hypothesis",
      "can_run_predefined": true/false,
      "recommended_runner_config": {{
        "experiment_type": "Classification or Regression or Parameter Sweep",
        "model_choice": "Random Forest or Logistic Regression or Linear Regression",
        "suggested_param": "e.g. n_estimators or test_size"
      }}
    }}
  ]
}}
"""
    raw_json = call_gemini_with_retry(prompt=prompt, system_instruction=system_instruction, temperature=0.2)
    try:
        data = json.loads(raw_json)
        recs = FollowUpRecommendations.model_validate(data)
        return recs.recommendations
    except Exception:
        cleaned = raw_json.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned)
        recs = FollowUpRecommendations.model_validate(data)
        return recs.recommendations
