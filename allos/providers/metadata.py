# allos/providers/metadata.py

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# --- Static Pricing Database (expandable) ---
_STATIC_PRICING = {
    "openai": {
        "gpt-4o": {"input": 5.00, "output": 15.00},  # Prices per million tokens
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    }
}

# --- Pydantic Models for the Schema ---


class ModelConfiguration(BaseModel):
    # Not working as of now
    temperature: Optional[float] = None
    max_tokens: Optional[int] = Field(None, alias="max_output_tokens")
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop_sequences: Optional[List[str]] = None


class ModelInfo(BaseModel):
    provider: str
    model_id: str
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    api_endpoint: Optional[str] = None
    configuration: ModelConfiguration


# --- Models for Multi-Modal Details (Future) ---
class ImageDetail(BaseModel):
    format: Optional[str] = None
    resolution: Optional[str] = None
    size_bytes: Optional[int] = None
    tokens_consumed: Optional[int] = None


class MultiModalDetails(BaseModel):
    images: List[ImageDetail] = []
    audio_clips: List[Any] = []  # Placeholder for future models
    video_clips: List[Any] = []  # Placeholder for future models


class CacheUsage(BaseModel):
    # Not working as of now
    cache_read_tokens: int = 0
    cache_hit_rate: float = 0.0
    cache_creation_tokens: int = 0


class EstimatedCost(BaseModel):
    total_usd: float = 0.0
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    pricing_source: str = "static_config"
    pricing_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    # Future
    cache_write_cost_usd: float = 0.0
    cache_read_cost_usd: float = 0.0


class Usage(BaseModel):
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_usage: Optional[CacheUsage] = None
    estimated_cost: Optional[EstimatedCost] = None
    # Future
    tokens_by_modality: Optional[Dict[str, int]] = None
    multi_modal_details: Optional[MultiModalDetails] = None


class Latency(BaseModel):
    total_duration_ms: int
    time_to_first_token_ms: Optional[int] = None
    # Future
    generation_time_ms: Optional[int] = None
    network_time_ms: Optional[int] = None
    queue_time_ms: Optional[int] = None
    tool_execution_time_ms: Optional[int] = None
    tokens_per_second: Optional[float] = None


class ToolCallDetail(BaseModel):
    tool_call_id: str
    tool_name: str
    arguments: Dict[str, Any]
    # Fields to be populated by the agent after execution
    execution_time_ms: Optional[int] = None
    status: Optional[str] = None


class ToolInfo(BaseModel):
    tools_available: List[str]
    total_tool_calls: int = 0
    tool_calls: List[ToolCallDetail] = []
    # Future
    unique_tools_used: List[str] = []


class TurnTokensUsed(BaseModel):
    input_tokens: int
    output_tokens: int


class TurnLog(BaseModel):
    turn_number: int
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    model_used: str
    content_type: str
    tokens_used: TurnTokensUsed
    duration_ms: int
    tools_called: List[str] = []
    stop_reason: Optional[str] = None


class TurnsInfo(BaseModel):
    total_turns: int = 0
    max_turns_reached: bool = False
    turn_history: List[TurnLog] = []


# Models for Errors, Warnings, and Cache (Future)
class ErrorDetail(BaseModel):
    error_code: str
    error_message: str
    source: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    recoverable: bool = False


class WarningDetail(BaseModel):
    warning_code: str
    warning_message: str
    severity: str


class RetryInfo(BaseModel):
    retry_count: int
    retry_reasons: List[str]
    backoff_strategy: str


class ErrorsAndWarnings(BaseModel):
    errors: List[ErrorDetail] = []
    warnings: List[WarningDetail] = []
    retries: Optional[RetryInfo] = None


class CacheInfo(BaseModel):
    cache_enabled: bool = False
    cache_hit: bool = False
    cache_type: Optional[str] = None
    cache_key: Optional[str] = None
    cache_ttl_seconds: Optional[int] = None


class QualitySignals(BaseModel):
    finish_reason: Optional[str] = None
    refusal_detected: bool = False
    response_truncated: bool = False
    # Future
    refusal_reason: Optional[str] = None
    confidence_score: Optional[float] = None
    content_filter_results: Optional[Dict[str, str]] = None
    context_window_usage: Optional[float] = None


class ProviderSpecificOpenAI(BaseModel):
    # Not working as of now
    system_fingerprint: Optional[str] = None
    # Future
    logprobs: Optional[Any] = None


class ProviderSpecific(BaseModel):
    openai: Optional[ProviderSpecificOpenAI] = None
    # Future
    anthropic: Optional[Any] = None
    google: Optional[Any] = None
    ollama: Optional[Any] = None
    chat_completions: Optional[Any] = None


class SdkInfo(BaseModel):
    sdk_version: str
    # Future
    sdk_language: str = "python"
    user_agent: Optional[str] = None
    custom_metadata: Dict[str, Any] = {}


class Metadata(BaseModel):
    request_id: str = Field(default_factory=lambda: f"req_{uuid.uuid4().hex[:16]}")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    status: str
    model: ModelInfo
    usage: Usage
    latency: Latency
    tools: ToolInfo
    turns: TurnsInfo = Field(default_factory=TurnsInfo)
    quality_signals: QualitySignals
    provider_specific: ProviderSpecific
    sdk: SdkInfo
    # Future
    cache: CacheInfo = Field(default_factory=CacheInfo)
    errors_and_warnings: ErrorsAndWarnings = Field(default_factory=ErrorsAndWarnings)
    metadata_version: str = "1.0.0"


class MetadataBuilder:
    """Builds the comprehensive metadata object from a provider response."""

    def __init__(
        self, provider_name: str, request_kwargs: Dict[str, Any], start_time: float
    ):
        self._provider_name = provider_name
        self._request_kwargs = request_kwargs
        self._start_time = start_time
        self._end_time = time.time()
        self._response_obj: Optional[Any] = None

    def with_response_obj(self, response_obj: Any) -> "MetadataBuilder":
        self._response_obj = response_obj
        return self

    def build(self) -> Metadata:
        if not self._response_obj:
            raise ValueError("A response object must be provided to build metadata.")

        # --- Direct mappings from response ---
        request_id_raw = getattr(
            self._response_obj, "id", f"req_{uuid.uuid4().hex[:16]}"
        )
        request_id = (
            str(request_id_raw) if request_id_raw else f"req_{uuid.uuid4().hex[:16]}"
        )

        model_id = getattr(self._response_obj, "model", "unknown")

        status_str_raw = getattr(self._response_obj, "status", "unknown")
        status_str = str(status_str_raw) if status_str_raw else "unknown"

        usage_obj = getattr(self._response_obj, "usage", None)

        # --- Build sub-objects ---
        model_info = self._build_model_info(model_id)
        usage_info = self._build_usage_info(model_id, usage_obj)
        latency_info = self._build_latency_info()
        tool_info = self._build_tool_info()
        quality_signals = self._build_quality_signals(status_str)
        provider_specific = self._build_provider_specific()
        sdk_info = self._build_sdk_info()

        return Metadata(
            request_id=request_id,
            status="success" if status_str == "completed" else status_str,
            model=model_info,
            usage=usage_info,
            latency=latency_info,
            tools=tool_info,
            quality_signals=quality_signals,
            provider_specific=provider_specific,
            sdk=sdk_info,
        )

    def _build_model_info(self, model_id: str) -> ModelInfo:
        model_version = None
        model_id_str = str(model_id) if model_id else "unknown"
        try:  # Best-effort parsing of version from ID like 'gpt-4o-2024-08-06'
            parts = model_id_str.split("-")
            if len(parts) >= 4 and all(p.isdigit() for p in parts[-3:]):
                model_version = "-".join(parts[-3:])
        except (AttributeError, TypeError, ValueError):
            # model_id wasn't a string or split/isdigit processing failed
            model_version = None

        return ModelInfo(
            provider=self._provider_name,
            model_id=model_id_str,
            model_version=model_version,
            configuration=ModelConfiguration.model_validate(
                self._request_kwargs, from_attributes=True
            ),
        )

    def _build_usage_info(self, model_id: str, usage_obj: Any) -> Usage:
        # Handle both Responses API (input_tokens/output_tokens) and
        # Chat Completions API (prompt_tokens/completion_tokens)
        input_tokens_raw = getattr(usage_obj, "input_tokens", None)
        if input_tokens_raw is None:
            input_tokens_raw = getattr(usage_obj, "prompt_tokens", None)
        input_tokens = int(input_tokens_raw) if input_tokens_raw is not None else 0

        output_tokens_raw = getattr(usage_obj, "output_tokens", None)
        if output_tokens_raw is None:
            output_tokens_raw = getattr(usage_obj, "completion_tokens", None)
        output_tokens = int(output_tokens_raw) if output_tokens_raw is not None else 0

        # Handle cached tokens from both API formats
        cached_tokens = 0
        if (
            hasattr(usage_obj, "input_tokens_details")
            and usage_obj.input_tokens_details
        ):
            cached_tokens_raw = getattr(
                usage_obj.input_tokens_details, "cached_tokens", None
            )
            cached_tokens = (
                int(cached_tokens_raw) if cached_tokens_raw is not None else 0
            )
        elif (
            hasattr(usage_obj, "prompt_tokens_details")
            and usage_obj.prompt_tokens_details
        ):
            cached_tokens_raw = getattr(
                usage_obj.prompt_tokens_details, "cached_tokens", None
            )
            cached_tokens = (
                int(cached_tokens_raw) if cached_tokens_raw is not None else 0
            )

        cost = self._calculate_cost(model_id, input_tokens, output_tokens)

        return Usage(
            total_tokens=input_tokens + output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_usage=CacheUsage(
                cache_read_tokens=cached_tokens,
                cache_hit_rate=(
                    (cached_tokens / input_tokens) if input_tokens > 0 else 0.0
                ),
            ),
            estimated_cost=cost,
        )

    def _calculate_cost(
        self, model_id: str, input_tokens: int, output_tokens: int
    ) -> Optional[EstimatedCost]:
        provider_prices = _STATIC_PRICING.get(self._provider_name, {})
        model_key = next((key for key in provider_prices if key in model_id), None)

        if not model_key:
            return None

        prices = provider_prices[model_key]
        input_cost = (input_tokens / 1_000_000) * prices["input"]
        output_cost = (output_tokens / 1_000_000) * prices["output"]

        return EstimatedCost(
            input_cost_usd=input_cost,
            output_cost_usd=output_cost,
            total_usd=input_cost + output_cost,
        )

    def _build_latency_info(self) -> Latency:
        return Latency(
            total_duration_ms=int((self._end_time - self._start_time) * 1000)
        )

    def _build_tool_info(self) -> ToolInfo:
        tools_available = [t.name for t in self._request_kwargs.get("tools", [])]
        tool_calls = []
        if hasattr(self._response_obj, "output") and self._response_obj.output:  # type: ignore
            for item in self._response_obj.output:  # type: ignore
                if item.type == "function_call":
                    # Skip if call_id is missing
                    call_id = getattr(item, "call_id", None)
                    if not call_id:
                        continue

                    try:
                        # Handle None or missing arguments
                        args_str = getattr(item, "arguments", None)
                        args = json.loads(args_str) if args_str else {}

                        tool_calls.append(
                            ToolCallDetail(
                                tool_call_id=call_id,
                                tool_name=item.name,
                                arguments=args,
                            )
                        )
                    except (json.JSONDecodeError, AttributeError):
                        continue

        return ToolInfo(
            tools_available=tools_available,
            total_tool_calls=len(tool_calls),
            tool_calls=tool_calls,
        )

    def _build_quality_signals(self, status: str) -> QualitySignals:
        refusal = False
        status_str = str(status) if status else "unknown"
        if hasattr(self._response_obj, "output") and self._response_obj.output:  # type: ignore
            for item in self._response_obj.output:  # type: ignore
                if item.type == "message" and hasattr(item, "content") and item.content:
                    if any(part.type == "refusal" for part in item.content):
                        refusal = True
                        break

        finish_reason = status_str
        if status_str == "incomplete" and hasattr(
            self._response_obj, "incomplete_details"
        ):
            finish_reason = getattr(
                self._response_obj.incomplete_details,  # type: ignore
                "reason",
                "incomplete",
            )

        return QualitySignals(
            finish_reason=finish_reason,
            refusal_detected=refusal,
            response_truncated=(finish_reason == "max_output_tokens"),
        )

    def _build_provider_specific(self) -> ProviderSpecific:
        # In a real scenario, this would be extracted from HTTP headers.
        # For now, we are leaving it as a placeholder.
        system_fingerprint_raw = getattr(self._response_obj, "system_fingerprint", None)
        # Convert to string only if it's actually a string, otherwise None
        system_fingerprint = (
            system_fingerprint_raw if isinstance(system_fingerprint_raw, str) else None
        )
        return ProviderSpecific(
            openai=ProviderSpecificOpenAI(system_fingerprint=system_fingerprint)
        )

    def _build_sdk_info(self) -> SdkInfo:
        from allos import __version__

        return SdkInfo(sdk_version=__version__)
