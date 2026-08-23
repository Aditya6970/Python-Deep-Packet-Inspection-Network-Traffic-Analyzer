"""OpenAI client wrapper: structured outputs, retries, timeouts, failure modes.

The engine must keep working when this does not, so every failure here is a
returned value rather than a raised exception. :class:`LLMResult` carries
either a parsed model or a reason it is absent.

Retry policy is deliberately narrow: transient failures (429, 5xx, timeouts,
connection errors) are retried with exponential backoff; authentication and
bad-request failures are **never** retried, because an invalid key or a
malformed schema will not fix itself and retrying only wastes time and quota.

API keys are never logged, never included in an exception message, and never
reach a prompt.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from .config import AIConfig
from .providers import StructuredMode, to_strict_json_schema

__all__ = [
    "FailureReason",
    "LLMResult",
    "LLMClient",
    "ProviderClient",
    "OpenAIClient",
    "FakeLLMClient",
    "create_client",
]

T = TypeVar("T", bound=BaseModel)

#: Backoff base, in seconds.  Attempt n waits roughly base * 2**n plus jitter.
_BACKOFF_BASE_SECONDS = 0.5
_BACKOFF_MAX_SECONDS = 8.0


class FailureReason(str, Enum):
    """Why an analysis did not produce a result."""

    NO_API_KEY = "no_api_key"
    SDK_MISSING = "sdk_missing"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    INVALID_PROVIDER = "invalid_provider"
    AUTH_FAILED = "auth_failed"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    API_ERROR = "api_error"
    INVALID_RESPONSE = "invalid_response"
    REFUSED = "refused"


@dataclass(slots=True)
class LLMResult:
    """Outcome of one structured completion.

    Exactly one of ``parsed`` / ``failure`` is set.
    """

    parsed: BaseModel | None = None
    failure: FailureReason | None = None
    detail: str = ""
    attempts: int = 0
    model: str = ""
    provider: str = ""
    elapsed_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.parsed is not None


class LLMClient(Protocol):
    """The seam that lets tests run without a network.

    :class:`OpenAIClient` and :class:`FakeLLMClient` both satisfy it.
    """

    def is_available(self) -> bool:
        """Whether a call could be attempted.  Makes no network request."""
        ...

    def complete_structured(
        self, messages: list[dict[str, str]], schema: type[T]
    ) -> LLMResult:
        """Request a completion constrained to ``schema``."""
        ...


class ProviderClient:
    """One client for every provider, driven by :class:`~ai.providers.ProviderSpec`.

    Groq and Ollama both expose OpenAI-compatible ``/chat/completions``
    endpoints, so all three providers run through the ``openai`` SDK with a
    different ``base_url``. No provider-specific SDK is needed.

    What genuinely differs is how schema conformance is obtained. That is
    handled by :meth:`_call_once`, which dispatches on
    :attr:`~ai.providers.ProviderSpec.structured_mode`. All three paths end in
    the same Pydantic validation, so there is one canonical schema regardless
    of provider.
    """

    __slots__ = ("_config", "_client")

    def __init__(self, config: AIConfig) -> None:
        self._config = config
        self._client: Any = None

    # ------------------------------------------------------------------
    @property
    def provider_name(self) -> str:
        return self._config.provider.value

    def is_available(self) -> bool:
        """Whether a call could be attempted.  Makes no network request.

        For Ollama this reports only that the SDK is importable — whether the
        local server is actually running is discovered on the first call and
        surfaces as :attr:`FailureReason.PROVIDER_UNAVAILABLE`.
        """
        if self._config.invalid_provider is not None:
            return False
        if not self._config.has_api_key():
            return False
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True

    def _ensure_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            kwargs: dict[str, Any] = {
                "api_key": self._config.api_key or "not-required",
                "timeout": self._config.timeout_seconds,
                # Our own retry loop handles backoff; disable the SDK's so the
                # two do not compound into a much longer wall-clock time.
                "max_retries": 0,
            }
            if self._config.base_url:
                kwargs["base_url"] = self._config.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    # ------------------------------------------------------------------
    @staticmethod
    def _classify(exc: Exception) -> tuple[FailureReason, bool]:
        """Map an SDK exception to a reason and whether it is worth retrying."""
        import openai

        if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
            return FailureReason.AUTH_FAILED, False
        if isinstance(exc, openai.RateLimitError):
            return FailureReason.RATE_LIMITED, True
        if isinstance(exc, openai.APITimeoutError):
            return FailureReason.TIMEOUT, True
        if isinstance(exc, openai.APIConnectionError):
            # A refused connection usually means a local server is not running,
            # which is a different problem from a flaky network.
            return FailureReason.PROVIDER_UNAVAILABLE, True
        if isinstance(exc, openai.NotFoundError):
            # Typically a retired or misspelled model name.
            return FailureReason.API_ERROR, False
        if isinstance(exc, openai.BadRequestError):
            return FailureReason.API_ERROR, False
        if isinstance(exc, openai.APIStatusError):
            retryable = getattr(exc, "status_code", 0) >= 500
            return FailureReason.API_ERROR, retryable
        return FailureReason.API_ERROR, False

    # ------------------------------------------------------------------
    def _call_once(self, messages: list[dict[str, str]], schema: type[T]) -> T | None:
        """One attempt, dispatched on the provider's structured-output tier.

        Returns a validated model, or raises for the retry loop to classify.
        ``None`` means the provider returned nothing usable.
        """
        client = self._ensure_client()
        mode = self._config.spec.structured_mode
        common: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_output_tokens,
        }

        # --- tier 1: the SDK constrains and parses for us ------------------
        if mode is StructuredMode.NATIVE_PARSE:
            completion = client.beta.chat.completions.parse(
                response_format=schema, **common
            )
            message = completion.choices[0].message
            if getattr(message, "refusal", None):
                raise _Refusal(str(message.refusal))
            return message.parsed

        # --- tier 2: provider-enforced JSON Schema ------------------------
        if mode is StructuredMode.JSON_SCHEMA:
            json_schema = to_strict_json_schema(schema.model_json_schema())
            completion = client.chat.completions.create(
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "strict": self._config.spec.strict_schema,
                        "schema": json_schema,
                    },
                },
                **common,
            )
            content = completion.choices[0].message.content
            return schema.model_validate_json(content) if content else None

        # --- tier 3: valid JSON only; we enforce the schema ---------------
        completion = client.chat.completions.create(
            response_format={"type": "json_object"}, **common
        )
        content = completion.choices[0].message.content
        if not content:
            return None
        return schema.model_validate_json(_strip_code_fence(content))

    def complete_structured(
        self, messages: list[dict[str, str]], schema: type[T]
    ) -> LLMResult:
        """Call the provider, constrained to ``schema``, with bounded retries."""
        started = time.monotonic()
        cfg = self._config

        def fail(reason: FailureReason, detail: str, attempts: int = 0) -> LLMResult:
            return LLMResult(
                failure=reason,
                detail=detail,
                attempts=attempts,
                model=cfg.model,
                provider=cfg.provider.value,
                elapsed_seconds=time.monotonic() - started,
            )

        if cfg.invalid_provider is not None:
            return fail(
                FailureReason.INVALID_PROVIDER,
                f"Unknown provider {cfg.invalid_provider!r}.",
            )

        if not cfg.has_api_key():
            return fail(FailureReason.NO_API_KEY, f"{cfg.spec.api_key_env} is not set.")

        try:
            import openai  # noqa: F401
        except ImportError as exc:
            return fail(FailureReason.SDK_MISSING, f"openai package not installed: {exc}")

        last_reason = FailureReason.API_ERROR
        last_detail = ""
        attempts = 0

        for attempt in range(cfg.max_retries + 1):
            attempts = attempt + 1
            try:
                parsed = self._call_once(messages, schema)

                if parsed is None:
                    return fail(
                        FailureReason.INVALID_RESPONSE,
                        "Provider returned no usable content.",
                        attempts,
                    )

                return LLMResult(
                    parsed=parsed,
                    attempts=attempts,
                    model=cfg.model,
                    provider=cfg.provider.value,
                    elapsed_seconds=time.monotonic() - started,
                )

            except _Refusal as exc:
                return fail(FailureReason.REFUSED, str(exc), attempts)

            except ValidationError as exc:
                # Expected occasionally in JSON_OBJECT mode, where the provider
                # guarantees valid JSON but not our schema.  Worth one retry:
                # a different sample may conform.
                last_reason = FailureReason.INVALID_RESPONSE
                last_detail = f"Response failed validation: {exc.error_count()} error(s)."
                if attempt >= cfg.max_retries:
                    break
                time.sleep(_backoff(attempt))
                continue

            except json.JSONDecodeError:
                last_reason = FailureReason.INVALID_RESPONSE
                last_detail = "Provider returned content that is not valid JSON."
                if attempt >= cfg.max_retries:
                    break
                time.sleep(_backoff(attempt))
                continue

            except Exception as exc:  # noqa: BLE001 - deliberately broad
                last_reason, retryable = self._classify(exc)
                # Keep the type only: never echo a provider message verbatim,
                # in case it reflects request material back.
                last_detail = type(exc).__name__
                if not retryable or attempt >= cfg.max_retries:
                    break
                time.sleep(_backoff(attempt))

        return fail(last_reason, last_detail, attempts)


class _Refusal(Exception):
    """Raised internally when a model declines to answer."""


def _backoff(attempt: int) -> float:
    """Exponential backoff with jitter."""
    delay = min(_BACKOFF_BASE_SECONDS * (2**attempt), _BACKOFF_MAX_SECONDS)
    return delay + random.uniform(0, delay * 0.1)


def _strip_code_fence(content: str) -> str:
    """Remove a markdown fence a weaker model may have wrapped the JSON in."""
    text = content.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def create_client(config: AIConfig) -> "LLMClient":
    """Return the client for the configured provider.

    A factory rather than a conditional at the call site, so
    :mod:`ai.analyzer` never names a provider.
    """
    return ProviderClient(config)


#: Backwards-compatible alias.  The original OpenAI-only client is now just
#: the provider client with an OpenAI config.
OpenAIClient = ProviderClient


class FakeLLMClient:
    """Offline stand-in for :class:`OpenAIClient`.

    Lets the whole pipeline — extraction, redaction, prompt building,
    validation, rendering — be tested without a key or a network. Configure it
    to return a canned result, or to fail in a specific way.
    """

    __slots__ = (
        "_response", "_failure", "_detail", "available",
        "provider_name", "calls", "last_messages",
    )

    def __init__(
        self,
        response: BaseModel | None = None,
        failure: FailureReason | None = None,
        detail: str = "",
        available: bool = True,
        provider_name: str = "fake",
    ) -> None:
        self._response = response
        self._failure = failure
        self._detail = detail
        self.available = available
        self.provider_name = provider_name
        self.calls = 0
        self.last_messages: list[dict[str, str]] = []

    def is_available(self) -> bool:
        return self.available

    def complete_structured(
        self, messages: list[dict[str, str]], schema: type[T]
    ) -> LLMResult:
        self.calls += 1
        self.last_messages = messages

        if self._failure is not None:
            return LLMResult(
                failure=self._failure, detail=self._detail, attempts=1,
                model="fake", provider=self.provider_name,
            )
        if self._response is None:
            return LLMResult(
                failure=FailureReason.INVALID_RESPONSE,
                detail="FakeLLMClient has no configured response.",
                attempts=1, model="fake", provider=self.provider_name,
            )
        return LLMResult(
            parsed=self._response, attempts=1, model="fake",
            provider=self.provider_name,
        )
