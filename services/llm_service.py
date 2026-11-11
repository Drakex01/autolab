import json
import re
import time
from typing import Tuple, Optional, List, Dict, Any, Type, TypeVar
from pydantic import BaseModel, ValidationError
from core.config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_API_VERSION
from core.schemas import (
    ExperimentPlanSchema,
    InterpretationResult,
    FollowUpRecommendations,
    NextExperimentRecommendation
)

T = TypeVar("T", bound=BaseModel)


class GeminiQuotaExhaustedError(RuntimeError):
    """Raised when Gemini API rejects a request with HTTP 429 RESOURCE_EXHAUSTED."""

    def __init__(self, message: str, retry_after: Optional[str] = None, is_daily_quota: bool = True):
        super().__init__(message)
        self.retry_after = retry_after
        self.is_daily_quota = is_daily_quota


def _normalize_retry_duration(raw_duration: str) -> str:
    """Formats long decimal durations like '14.3712485s' into readable strings like '14.4s'."""
    raw = raw_duration.strip()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*(ms|s|m|h)", raw, flags=re.IGNORECASE)
    if not match:
        return raw
    num_str, unit = match.group(1), match.group(2).lower()
    if "." in num_str:
        rounded = f"{float(num_str):.1f}".rstrip("0").rstrip(".")
        return f"{rounded}{unit}"
    return f"{num_str}{unit}"


def _extract_retry_after(error: Exception) -> Optional[str]:
    """Extracts suggested retry time from Gemini error details, headers, or message."""
    details = getattr(error, "details", None)
    if isinstance(details, dict):
        detail_items = details.get("details") or details.get("error", {}).get("details") or []
        if isinstance(detail_items, list):
            for item in detail_items:
                if isinstance(item, dict) and item.get("retryDelay"):
                    return _normalize_retry_duration(str(item["retryDelay"]))

    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        retry_hdr = headers.get("Retry-After") or headers.get("retry-after")
        if retry_hdr:
            hdr_str = str(retry_hdr).strip()
            return f"{hdr_str}s" if hdr_str.isdigit() else hdr_str

    combined_text = f"{getattr(error, 'message', '') or ''} {str(error)}"
    msg_match = re.search(
        r"(?:retry\s+(?:in|after)|retryDelay['\"]?\s*[:=]\s*['\"]?)\s*([0-9]+(?:\.[0-9]+)?\s*(?:ms|s|m|h))",
        combined_text,
        flags=re.IGNORECASE,
    )
    if msg_match:
        return _normalize_retry_duration(msg_match.group(1))
    return None


def _is_daily_or_non_transient_quota(error: Exception) -> bool:
    """Distinguishes non-transient daily/plan quota exhaustion from transient per-minute limits."""
    combined = (
        f"{getattr(error, 'status', '') or ''} "
        f"{getattr(error, 'message', '') or ''} "
        f"{getattr(error, 'details', '') or ''} "
        f"{str(error)}"
    ).lower()

    daily_markers = ("perday", "requestsperday", "free_tier", "daily", "current quota", "limit: 0")
    if any(marker in combined for marker in daily_markers):
        return True

    transient_markers = ("perminute", "requestsperminute", "tokensperminute")
    if any(marker in combined for marker in transient_markers):
        return False

    # Default 429 RESOURCE_EXHAUSTED without explicit per-minute markers as non-transient
    return True


def _build_quota_error(error: Exception, is_daily_quota: bool) -> GeminiQuotaExhaustedError:
    retry_after = _extract_retry_after(error)
    if is_daily_quota:
        msg = "Gemini API daily quota has been exhausted (HTTP 429 RESOURCE_EXHAUSTED)."
    else:
        msg = "Gemini API rate limit exceeded (HTTP 429 RESOURCE_EXHAUSTED)."
    if retry_after:
        msg += f" Please retry in {retry_after}."
    else:
        msg += " Please try again once your quota resets or use the rule-based fallback."
    return GeminiQuotaExhaustedError(msg, retry_after=retry_after, is_daily_quota=is_daily_quota)

def is_gemini_configured() -> Tuple[bool, str]:
    """Checks whether the Gemini API key is configured."""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        return False, "GEMINI_API_KEY is not set. Add your key to .env or environment variables."
    return True, f"Configured with model: {GEMINI_MODEL} (API {GEMINI_API_VERSION})"

def get_gemini_client():
    from google import genai
    from google.genai import types
    return genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(api_version=GEMINI_API_VERSION),
    )

def _parse_json_response(raw_json: str, schema_cls: Type[T]) -> T:
    """Parses raw Gemini text output (stripping markdown fences if needed) and validates with Pydantic."""
    cleaned = raw_json.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "", 1).replace("```", "").strip()
    try:
        data = json.loads(cleaned)
        return schema_cls.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"Gemini returned malformed JSON for {schema_cls.__name__}: {e}") from e

def _maybe_as_quota_error(error: Exception) -> Optional[GeminiQuotaExhaustedError]:
    """Converts HTTP 429 / RESOURCE_EXHAUSTED exceptions into GeminiQuotaExhaustedError."""
    code = getattr(error, "code", None)
    status_upper = str(getattr(error, "status", "") or "").upper()
    err_upper = str(error).upper()
    is_429 = (code == 429) or ("429" in err_upper and "RESOURCE_EXHAUSTED" in err_upper)
    is_resource_exhausted = ("RESOURCE_EXHAUSTED" in status_upper) or ("RESOURCE_EXHAUSTED" in err_upper)
    if is_429 or is_resource_exhausted:
        is_daily = _is_daily_or_non_transient_quota(error)
        return _build_quota_error(error, is_daily_quota=is_daily)
    return None


def call_gemini_with_retry(
    prompt: str,
    system_instruction: str = "",
    temperature: float = 0.2,
    max_retries: int = 2,
    response_schema: Optional[Type[BaseModel]] = None,
) -> str:
    """Calls Gemini API with configurable retries and explicit error handling."""
    configured, msg = is_gemini_configured()
    if not configured:
        raise ValueError(msg)

    from google.genai import types, errors

    client = get_gemini_client()
    config = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system_instruction if system_instruction else None,
        response_mime_type="application/json",
        response_schema=response_schema,
    )

    last_error: Optional[Exception] = None
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
            quota_err = _maybe_as_quota_error(e)
            if quota_err is not None:
                if quota_err.is_daily_quota:
                    # Daily quota exhaustion is not transient — fail immediately without retrying
                    raise quota_err from e
                last_error = quota_err
            elif isinstance(e, errors.ClientError):
                # Non-429 4xx errors are non-retryable client/config errors
                raise RuntimeError(
                    f"Gemini API request rejected ({e.code} {e.status}): {e.message or str(e)}"
                ) from e
            else:
                last_error = e

        if attempt < max_retries:
            time.sleep(1.5 * (attempt + 1))

    if isinstance(last_error, GeminiQuotaExhaustedError):
        raise last_error
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
    raw_json = call_gemini_with_retry(
        prompt=prompt,
        system_instruction=system_instruction,
        temperature=0.2,
        response_schema=ExperimentPlanSchema,
    )
    return _parse_json_response(raw_json, ExperimentPlanSchema)

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
    raw_json = call_gemini_with_retry(
        prompt=prompt,
        system_instruction=system_instruction,
        temperature=0.2,
        response_schema=InterpretationResult,
    )
    return _parse_json_response(raw_json, InterpretationResult)

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
    recs = _parse_json_response(raw_json, FollowUpRecommendations)
    return recs.recommendations
