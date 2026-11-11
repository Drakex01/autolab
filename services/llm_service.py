import copy
import json
import re
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

import requests
from pydantic import BaseModel, ValidationError

from core.config import (
    GEMINI_API_KEY,
    GEMINI_API_VERSION,
    GEMINI_MODEL,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_TEMPERATURE,
    OLLAMA_TIMEOUT_SECONDS,
)
from core.schemas import (
    ExperimentPlanSchema,
    FollowUpRecommendations,
    InterpretationResult,
    NextExperimentRecommendation,
)
from core.state import get_runtime_llm_settings

T = TypeVar("T", bound=BaseModel)

# Serialize local Ollama inference calls to prevent concurrent model loading on 16 GB RAM / 4 GB VRAM
_OLLAMA_REQUEST_LOCK = threading.Lock()

# Array fields that must be non-empty during LLM generation (excludes evidence_citations which may be [] when no docs exist)
_SCHEMA_NONEMPTY_FIELDS: Dict[str, Tuple[str, ...]] = {
    "ExperimentPlanSchema": (
        "independent_variables",
        "dependent_variables",
        "controlled_variables",
        "evaluation_metrics",
        "next_steps",
    ),
    "InterpretationResult": ("possible_reasons", "limitations"),
    "FollowUpRecommendations": ("recommendations",),
}


def _build_ollama_format_schema(response_schema: Type[BaseModel]) -> Dict[str, Any]:
    """Builds a JSON Schema for Ollama's format constraint ensuring all fields and non-empty lists are emitted."""
    schema = copy.deepcopy(response_schema.model_json_schema())
    props = schema.get("properties") or {}
    if props:
        schema["required"] = list(props.keys())
        for field_name in _SCHEMA_NONEMPTY_FIELDS.get(response_schema.__name__, ()):
            field_spec = props.get(field_name)
            if isinstance(field_spec, dict) and field_spec.get("type") == "array":
                field_spec["minItems"] = 1
    return schema


class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama service is stopped, unreachable, or timed out."""


class OllamaModelNotFoundError(RuntimeError):
    """Raised when the requested Ollama model is not downloaded locally."""


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


def _parse_json_response(raw_json: str, schema_cls: Type[T]) -> T:
    """Parses raw LLM text output (stripping markdown fences if needed) and validates with Pydantic."""
    if not raw_json or not raw_json.strip():
        raise ValueError(f"Empty model response received for {schema_cls.__name__}.")

    cleaned = raw_json.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
        return schema_cls.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"Model returned invalid JSON for {schema_cls.__name__}: {e}") from e


def _model_matches_tag(installed_name: str, target_model: str) -> bool:
    if installed_name == target_model:
        return True
    if ":" not in target_model and installed_name == f"{target_model}:latest":
        return True
    return False


def get_active_provider_name(provider: Optional[str] = None) -> str:
    if provider:
        return provider.strip().lower()
    runtime = get_runtime_llm_settings()
    return runtime["provider"]


def get_active_ollama_model(model: Optional[str] = None) -> str:
    if model:
        return model.strip()
    runtime = get_runtime_llm_settings()
    return runtime["ollama_model"]


def check_ollama_status(base_url: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
    """Checks local Ollama connectivity, installed models, and active GPU/CPU memory allocation."""
    url = (base_url or OLLAMA_BASE_URL).rstrip("/")
    target_model = get_active_ollama_model(model)

    result: Dict[str, Any] = {
        "connected": False,
        "base_url": url,
        "configured_model": target_model,
        "model_available": False,
        "installed_models": [],
        "running_models": [],
        "error": None,
    }

    try:
        tags_resp = requests.get(f"{url}/api/tags", timeout=3)
        tags_resp.raise_for_status()
        tags_data = tags_resp.json()
        installed = [m.get("name", "") for m in tags_data.get("models", []) if m.get("name")]
        result["connected"] = True
        result["installed_models"] = installed
        result["model_available"] = any(_model_matches_tag(m, target_model) for m in installed)
        if not result["model_available"]:
            result["error"] = (
                f"Model '{target_model}' is not downloaded in Ollama. "
                f"Run `ollama pull {target_model}` or switch models in LLM Settings."
            )
    except requests.ConnectionError:
        result["error"] = (
            f"Ollama is not running or unreachable at {url}. "
            "Start Ollama (`ollama serve` or launch the Ollama app) to use local AI features."
        )
        return result
    except requests.Timeout:
        result["error"] = f"Timed out connecting to Ollama at {url}."
        return result
    except Exception as e:
        result["error"] = f"Failed to inspect Ollama status: {e}"
        return result

    try:
        ps_resp = requests.get(f"{url}/api/ps", timeout=3)
        if ps_resp.status_code == 200:
            ps_data = ps_resp.json()
            running = []
            for rm in ps_data.get("models", []):
                size_b = int(rm.get("size") or 0)
                vram_b = int(rm.get("size_vram") or 0)
                cpu_b = max(0, size_b - vram_b)
                running.append({
                    "name": rm.get("name", ""),
                    "size_mb": round(size_b / (1024 * 1024), 1),
                    "vram_mb": round(vram_b / (1024 * 1024), 1),
                    "cpu_ram_mb": round(cpu_b / (1024 * 1024), 1),
                    "context_length": rm.get("context_length"),
                })
            result["running_models"] = running
    except Exception:
        pass

    return result


def unload_ollama_model(model_name: str, base_url: Optional[str] = None) -> bool:
    """Explicitly releases an unused Ollama model from RAM/VRAM via keep_alive=0."""
    url = (base_url or OLLAMA_BASE_URL).rstrip("/")
    try:
        resp = requests.post(
            f"{url}/api/generate",
            json={"model": model_name, "keep_alive": 0},
            timeout=5,
        )
        return resp.status_code == 200
    except Exception:
        return False


class BaseLLMProvider(ABC):
    """Abstract interface for AutoLab LLM providers."""

    provider_name: str

    @abstractmethod
    def is_available(self) -> Tuple[bool, str]:
        pass

    @abstractmethod
    def generate_json(
        self,
        prompt: str,
        system_instruction: str = "",
        temperature: float = 0.2,
        max_retries: int = 2,
        response_schema: Optional[Type[BaseModel]] = None,
    ) -> str:
        pass


class OllamaProvider(BaseLLMProvider):
    """Local Ollama provider with schema-constrained output and memory-safe request locking."""

    provider_name = "ollama"

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        num_ctx: Optional[int] = None,
        timeout_seconds: int = OLLAMA_TIMEOUT_SECONDS,
    ):
        runtime = get_runtime_llm_settings()
        self.base_url = (base_url or runtime["ollama_base_url"]).rstrip("/")
        self.model = (model or runtime["ollama_model"]).strip()
        self.num_ctx = int(num_ctx or runtime["ollama_num_ctx"])
        self.timeout_seconds = timeout_seconds

    def is_available(self) -> Tuple[bool, str]:
        status = check_ollama_status(base_url=self.base_url, model=self.model)
        if not status["connected"]:
            return False, status["error"] or f"Ollama unreachable at {self.base_url}."
        if not status["model_available"]:
            return False, status["error"] or f"Ollama model '{self.model}' not installed."
        return True, f"Ollama connected ({self.model}, ctx={self.num_ctx})"

    def generate_json(
        self,
        prompt: str,
        system_instruction: str = "",
        temperature: float = OLLAMA_TEMPERATURE,
        max_retries: int = 2,
        response_schema: Optional[Type[BaseModel]] = None,
    ) -> str:
        messages = []
        if system_instruction.strip():
            messages.append({"role": "system", "content": system_instruction.strip()})
        messages.append({"role": "user", "content": prompt})

        fmt: Any = _build_ollama_format_schema(response_schema) if response_schema is not None else "json"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": fmt,
            "options": {
                "temperature": temperature,
                "num_ctx": self.num_ctx,
            },
            "keep_alive": OLLAMA_KEEP_ALIVE,
        }

        last_error: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                with _OLLAMA_REQUEST_LOCK:
                    resp = requests.post(
                        f"{self.base_url}/api/chat",
                        json=payload,
                        timeout=self.timeout_seconds,
                    )

                if resp.status_code == 404:
                    err_text = resp.text
                    try:
                        err_text = resp.json().get("error", resp.text)
                    except Exception:
                        pass
                    raise OllamaModelNotFoundError(
                        f"Ollama model '{self.model}' is not available locally ({err_text}). "
                        f"Run `ollama pull {self.model}` or switch to `qwen2.5:3b-instruct`."
                    )

                if resp.status_code >= 400:
                    err_msg = resp.text
                    try:
                        err_msg = resp.json().get("error", resp.text)
                    except Exception:
                        pass
                    if "not found" in err_msg.lower() and "model" in err_msg.lower():
                        raise OllamaModelNotFoundError(
                            f"Ollama model '{self.model}' is not installed locally. "
                            f"Run `ollama pull {self.model}` or switch to `qwen2.5:3b-instruct`."
                        )
                    if 400 <= resp.status_code < 500:
                        raise RuntimeError(f"Ollama rejected request ({resp.status_code}): {err_msg}")
                    raise RuntimeError(f"Ollama server error ({resp.status_code}): {err_msg}")

                data = resp.json()
                content = str((data.get("message") or {}).get("content") or "").strip()
                if content:
                    return content
                raise ValueError("Empty response received from Ollama.")

            except (requests.ConnectionError, requests.Timeout) as e:
                # Fail fast when local Ollama server is stopped or timed out
                raise OllamaUnavailableError(
                    f"Ollama is not reachable at {self.base_url}. "
                    "Ensure Ollama is running locally (`ollama serve` or the Ollama desktop app) and retry."
                ) from e
            except OllamaModelNotFoundError:
                raise
            except ValueError as e:
                last_error = e
            except RuntimeError as e:
                if "rejected request" in str(e):
                    raise
                last_error = e

            if attempt < max_retries:
                time.sleep(1.0)

        raise RuntimeError(f"Ollama generation failed after {max_retries + 1} attempts: {last_error}")


def is_gemini_configured() -> Tuple[bool, str]:
    """Checks whether the optional Gemini API key is configured."""
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

    from google.genai import errors, types

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
                config=config,
            )
            if response and response.text:
                return response.text.strip()
            raise ValueError("Empty response received from Gemini.")
        except Exception as e:
            quota_err = _maybe_as_quota_error(e)
            if quota_err is not None:
                if quota_err.is_daily_quota:
                    raise quota_err from e
                last_error = quota_err
            elif isinstance(e, errors.ClientError):
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


class GeminiProvider(BaseLLMProvider):
    """Optional cloud Gemini provider (used only when explicitly selected)."""

    provider_name = "gemini"

    def is_available(self) -> Tuple[bool, str]:
        return is_gemini_configured()

    def generate_json(
        self,
        prompt: str,
        system_instruction: str = "",
        temperature: float = 0.2,
        max_retries: int = 2,
        response_schema: Optional[Type[BaseModel]] = None,
    ) -> str:
        return call_gemini_with_retry(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            max_retries=max_retries,
            response_schema=response_schema,
        )


def get_llm_provider(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    num_ctx: Optional[int] = None,
) -> BaseLLMProvider:
    """Factory returning the configured LLM provider instance (Ollama by default)."""
    chosen = get_active_provider_name(provider)
    if chosen == "gemini":
        return GeminiProvider()
    if chosen == "ollama":
        return OllamaProvider(model=model, num_ctx=num_ctx)
    raise ValueError(f"Unsupported LLM provider: '{chosen}'. Choose 'ollama' or 'gemini'.")


def is_llm_configured(provider: Optional[str] = None, model: Optional[str] = None) -> Tuple[bool, str]:
    """Checks availability of the active provider (Ollama by default)."""
    chosen = get_active_provider_name(provider)
    if chosen == "gemini":
        return is_gemini_configured()
    return get_llm_provider(provider="ollama", model=model).is_available()


def test_ollama_prompt(
    prompt: str = "Respond with JSON containing {\"status\": \"ok\", \"message\": \"Local Ollama inference is working.\"}.",
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Executes a lightweight test prompt against the local Ollama instance."""
    runtime = get_runtime_llm_settings()
    chosen_model = (model or runtime["ollama_model"]).strip()
    url = runtime["ollama_base_url"]
    num_ctx = runtime["ollama_num_ctx"]

    with _OLLAMA_REQUEST_LOCK:
        try:
            resp = requests.post(
                f"{url}/api/chat",
                json={
                    "model": chosen_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": OLLAMA_TEMPERATURE, "num_ctx": num_ctx},
                    "keep_alive": OLLAMA_KEEP_ALIVE,
                },
                timeout=60,
            )
        except (requests.ConnectionError, requests.Timeout) as e:
            raise OllamaUnavailableError(
                f"Ollama is not reachable at {url}. Start Ollama and try again."
            ) from e

    if resp.status_code == 404 or ("not found" in resp.text.lower() and "model" in resp.text.lower()):
        raise OllamaModelNotFoundError(
            f"Ollama model '{chosen_model}' is not installed locally. Run `ollama pull {chosen_model}`."
        )
    resp.raise_for_status()
    data = resp.json()
    content = str((data.get("message") or {}).get("content") or "").strip()
    eval_ns = data.get("eval_duration") or 0
    return {
        "ok": True,
        "model": chosen_model,
        "response": content,
        "eval_duration_s": round(eval_ns / 1e9, 2) if eval_ns else None,
    }


def _generate_validated_schema(
    schema_cls: Type[T],
    prompt: str,
    system_instruction: str,
    temperature: float = OLLAMA_TEMPERATURE,
    max_retries: int = 2,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    pass_schema_to_provider: bool = True,
    required_nonempty_fields: Optional[tuple] = None,
) -> T:
    """
    Generates JSON from the active provider and strictly validates against Pydantic.
    Retries only on malformed/empty/incomplete model outputs (never on connection/quota errors).
    """
    llm = get_llm_provider(provider=provider, model=model)
    schema_arg = schema_cls if pass_schema_to_provider else None
    nonempty_fields = required_nonempty_fields or _SCHEMA_NONEMPTY_FIELDS.get(schema_cls.__name__, ())
    current_prompt = prompt
    last_val_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        raw_json = llm.generate_json(
            prompt=current_prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            max_retries=1,
            response_schema=schema_arg,
        )
        try:
            parsed = _parse_json_response(raw_json, schema_cls)
            if nonempty_fields:
                empty_fields = [
                    field_name
                    for field_name in nonempty_fields
                    if not getattr(parsed, field_name, None)
                ]
                if empty_fields:
                    raise ValueError(
                        f"These required list/text fields cannot be empty: {', '.join(empty_fields)}"
                    )
            return parsed
        except ValueError as e:
            last_val_error = e
            if attempt < max_retries:
                current_prompt = (
                    f"{prompt}\n\n"
                    f"[VALIDATION ERROR ON PREVIOUS ATTEMPT: {e}. "
                    f"Return a complete, valid JSON object matching every required field and exact type in the schema, "
                    f"with non-empty entries for all experiment planning arrays.]"
                )

    raise ValueError(
        f"Failed to obtain valid {schema_cls.__name__} JSON after {max_retries + 1} attempts: {last_val_error}"
    )


def generate_plan_llm(
    research_title: str,
    research_question: str,
    hypothesis: str,
    research_area: str,
    objective: str,
    constraints: str,
    evidence_context: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> ExperimentPlanSchema:
    """Generates a structured, evidence-aware scientific experiment plan."""
    system_instruction = (
        "You are AutoLab, a rigorous scientific experiment planning assistant. "
        "Your task is to generate a structured, testable, empirical experiment plan. "
        "Ground your plan in the provided literature evidence when available. "
        "Strict rule: Never fabricate citations, document names, page numbers, or numbers. "
        "Every item in evidence_citations MUST come directly from the RETRIEVED LITERATURE EVIDENCE section. "
        "If no retrieved evidence is provided, return evidence_citations as [] and set is_grounded_in_sources to false. "
        "Clearly distinguish verified evidence from general assumptions. "
        "CRITICAL: You MUST populate independent_variables, dependent_variables, controlled_variables, "
        "evaluation_metrics, next_steps, and assumptions with at least 1 to 4 concrete, non-empty strings each — NEVER leave them as []. "
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
{evidence_context if evidence_context.strip() else "[No uploaded documents found. Ground your plan in general scientific methodology, return evidence_citations as [], and mark is_grounded_in_sources as false.]"}

JSON SCHEMA REQUIRED (preserve exact data types — strings vs non-empty string arrays):
{{
  "objective": "Clear primary experimental objective (non-empty string)",
  "testable_hypothesis": "Testable operationalized hypothesis (non-empty string)",
  "independent_variables": ["Model choice (e.g., Random Forest vs Logistic Regression)", "Primary predictor features or hyperparameters"],
  "dependent_variables": ["Target outcome prediction", "Primary classification/regression performance scores"],
  "controlled_variables": ["Train/test split ratio (e.g., 80/20)", "Fixed random seed (e.g., 42)", "Consistent preprocessing pipeline"],
  "dataset_and_required_columns": "Expected dataset format, types, and columns (non-empty string)",
  "methodology": "Step-by-step experimental execution protocol (single non-empty string)",
  "baseline": "Baseline benchmark or reference model (non-empty string)",
  "evaluation_metrics": ["Accuracy", "Precision", "Recall", "Weighted F1-score"],
  "expected_outputs": "Anticipated empirical outputs and charts (non-empty string)",
  "risks_and_limitations": "Known experimental limitations, biases, or threats to validity (single non-empty string)",
  "next_steps": ["Train baseline model", "Train comparison model under identical split and seed", "Compare empirical metrics"],
  "evidence_citations": [
    {{
      "source_file": "Exact filename from retrieved context",
      "page_or_chunk": "Exact Page X / Chunk Y from retrieved context",
      "factual_note": "Specific finding or metric cited from the retrieved excerpt"
    }}
  ],
  "assumptions": ["List of methodological assumptions made"],
  "is_grounded_in_sources": true/false
}}
"""
    return _generate_validated_schema(
        ExperimentPlanSchema,
        prompt=prompt,
        system_instruction=system_instruction,
        temperature=OLLAMA_TEMPERATURE,
        provider=provider,
        model=model,
        pass_schema_to_provider=True,
        required_nonempty_fields=(
            "independent_variables",
            "dependent_variables",
            "controlled_variables",
            "evaluation_metrics",
            "next_steps",
        ),
    )


def interpret_results_llm(
    research_question: str,
    hypothesis: str,
    experiment_type: str,
    model_name: str,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    plan_context: Optional[str] = None,
    comparison_context: str = "",
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> InterpretationResult:
    """Interprets real experimental metrics and returns plain-language takeaways."""
    system_instruction = (
        "You are AutoLab, a rigorous scientific data analysis assistant. "
        "You are analyzing REAL empirical experiment metrics computed by a reproducible scikit-learn workflow. "
        "Strict rules: Never invent or alter measurements, statistical significance, p-values, or citations. "
        "Never infer class-level performance from weighted averages when class-specific metrics are not recorded. "
        "Distinguish empirical observations from tentative interpretations. "
        "Never claim an experiment proves a hypothesis; state whether it supports, fails to support, or is inconclusive. "
        "Highlight caveats such as sample size, potential leakage, or non-causality. "
        "Respond strictly with valid JSON."
    )

    comp_block = f"\nCOMPARABLE WORKSPACE RUNS:\n{comparison_context}\n" if comparison_context.strip() else ""

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
{comp_block}
JSON SCHEMA REQUIRED:
{{
  "summary": "Concise 2-3 sentence factual summary of what was observed",
  "what_results_suggest": "Key scientific and analytical takeaways from the numbers and charts",
  "hypothesis_status": "Supported" OR "Not Supported" OR "Inconclusive",
  "possible_reasons": ["List of 2-4 plausible methodological or structural reasons for this outcome"],
  "limitations": ["List of 2-3 caveats (sample size, data distribution, lack of external validation, etc.)"]
}}
"""
    return _generate_validated_schema(
        InterpretationResult,
        prompt=prompt,
        system_instruction=system_instruction,
        temperature=OLLAMA_TEMPERATURE,
        provider=provider,
        model=model,
        pass_schema_to_provider=True,
    )


def recommend_next_experiments_llm(
    research_question: str,
    hypothesis: str,
    current_model: str,
    current_metrics: Dict[str, Any],
    interpretation_summary: str,
    evidence_context: str,
    comparison_context: str = "",
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> List[NextExperimentRecommendation]:
    """Generates 1 to 3 concrete, non-generic follow-up experiment recommendations."""
    system_instruction = (
        "You are AutoLab, an expert scientific experimentation advisor. "
        "Based on the completed experiment and evidence, suggest 1 to 3 concrete, reproducible follow-up experiments. "
        "Do NOT provide generic advice like 'collect more data'. "
        "Do NOT recommend repeating already-completed comparisons or hyperparameter sweeps listed in the context. "
        "Prefer meaningful next steps such as hyperparameter sweeps (e.g., Logistic Regression C, Random Forest n_estimators/max_depth), "
        "stratified cross-validation, recall/false-negative evaluation, or multi-seed stability comparison. "
        "Provide exact parameters, variables to change, variables to keep constant, and what to measure. "
        "Indicate whether the experiment can be executed in AutoLab's predefined runner. "
        "Respond strictly with valid JSON conforming to the schema."
    )

    comp_block = f"\nCOMPLETED COMPARISONS & SWEEPS (DO NOT REPEAT):\n{comparison_context}\n" if comparison_context.strip() else ""

    prompt = f"""
CURRENT EXPERIMENT:
- Research Question: {research_question}
- Hypothesis: {hypothesis}
- Model Tested: {current_model}
- Achieved Metrics: {json.dumps(current_metrics, indent=2)}
- Interpretation: {interpretation_summary}
{comp_block}
EVIDENCE CONTEXT:
{evidence_context if evidence_context.strip() else "None"}

JSON SCHEMA REQUIRED:
{{
  "recommendations": [
    {{
      "title": "Short descriptive title, e.g. Logistic Regression C Hyperparameter Sweep",
      "what_to_change": "Specific hyperparameter or test setting to change",
      "what_to_keep_constant": "Exact variables, test split, and seed to keep identical",
      "what_to_measure": "Evaluation metric to observe, e.g. F1-weighted or recall",
      "why_useful": "Scientific rationale for this follow-up investigation",
      "expected_support_or_weakening": "What numerical threshold would support vs weaken the hypothesis",
      "can_run_predefined": true/false,
      "recommended_runner_config": {{
        "experiment_type": "Classification or Regression or Parameter Sweep",
        "model_choice": "Random Forest or Logistic Regression or Linear Regression",
        "suggested_param": "e.g. C or max_depth or random_state"
      }}
    }}
  ]
}}
"""
    chosen = get_active_provider_name(provider)
    # Gemini Developer API rejects Dict[str, Any] (additionalProperties) in response_schema, while Ollama supports full schema
    use_schema = (chosen != "gemini")
    recs = _generate_validated_schema(
        FollowUpRecommendations,
        prompt=prompt,
        system_instruction=system_instruction,
        temperature=OLLAMA_TEMPERATURE,
        provider=provider,
        model=model,
        pass_schema_to_provider=use_schema,
    )
    return recs.recommendations
