"""Configuration for the AI analysis layer.

Everything the layer needs is resolved here, from the environment, so that no
other module reads ``os.environ`` directly and no module holds a secret it did
not ask for.

Secrets
-------
The API key is read from the environment (or an optional ``.env`` file) and is
never written to a log, a repr, an exception message, or a prompt.
:class:`AIConfig` deliberately implements ``__repr__`` to mask it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Final

from .providers import Provider, ProviderSpec, get_provider_spec, parse_provider

__all__ = [
    "IPRedactionMode",
    "AIConfig",
    "ENV_API_KEY",
    "ENV_PROVIDER",
    "load_dotenv",
]

#: Environment variable selecting the provider: groq | ollama | openai.
ENV_PROVIDER: Final[str] = "DPI_LLM_PROVIDER"

#: Legacy/OpenAI key variable.  Each provider also has its own -- see
#: :data:`ai.providers.PROVIDERS`.
ENV_API_KEY: Final[str] = "OPENAI_API_KEY"

_ENV_MODEL: Final[str] = "DPI_AI_MODEL"
_ENV_TIMEOUT: Final[str] = "DPI_AI_TIMEOUT"
_ENV_MAX_RETRIES: Final[str] = "DPI_AI_MAX_RETRIES"
_ENV_MAX_FLOWS: Final[str] = "DPI_AI_MAX_FLOWS"
_ENV_IP_MODE: Final[str] = "DPI_AI_IP_MODE"
_ENV_BASE_URL: Final[str] = "DPI_AI_BASE_URL"

#: Provider used when nothing is configured.  Groq is the development default:
#: it has a free tier and supports strict structured outputs.
DEFAULT_PROVIDER: Final[str] = "groq"

DEFAULT_TIMEOUT_SECONDS: Final[float] = 30.0
DEFAULT_MAX_RETRIES: Final[int] = 3
#: Cap on how many flow records reach a single request.  A capture with
#: thousands of flows must hit this wall rather than a context limit or a bill.
DEFAULT_MAX_FLOWS: Final[int] = 40
DEFAULT_MAX_OUTPUT_TOKENS: Final[int] = 2000


class IPRedactionMode(str, Enum):
    """How IP addresses are treated before leaving the machine.

    ``REDACT_PRIVATE`` is the default: private (RFC 1918) addresses identify
    hosts on the user's own network and are replaced with stable pseudonyms,
    while public destination addresses carry the analytical signal and are
    kept.
    """

    FULL = "full"  # send addresses as-is (lab use, opt-in)
    REDACT_PRIVATE = "redact_private"  # default
    NONE = "none"  # send no addresses at all


def load_dotenv(path: str | Path = ".env") -> dict[str, str]:
    """Load ``KEY=value`` pairs from a ``.env`` file into ``os.environ``.

    Existing environment variables always win, so a real environment cannot be
    silently overridden by a stale file.  Missing file is not an error.

    Intentionally minimal — no third-party dependency, no interpolation, no
    export syntax.  Returns the keys that were actually set.
    """
    env_path = Path(path)
    applied: dict[str, str] = {}

    if not env_path.is_file():
        return applied

    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return applied

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            applied[key] = value

    return applied


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass(slots=True)
class AIConfig:
    """Resolved settings for one analysis run.

    Provider-specific details (endpoint, key variable, default model,
    structured-output mode) come from :mod:`ai.providers`; this class holds
    only the resolved values.
    """

    provider: Provider = Provider.GROQ
    api_key: str | None = None
    model: str = ""
    base_url: str | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    max_flows: int = DEFAULT_MAX_FLOWS
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    ip_mode: IPRedactionMode = IPRedactionMode.REDACT_PRIVATE
    #: Deterministic generation, so evaluation later is meaningful.
    temperature: float = 0.0
    #: Set when DPI_LLM_PROVIDER named something unrecognised.  Carried rather
    #: than raised so the caller can report it and still print DPI results.
    invalid_provider: str | None = None

    def __post_init__(self) -> None:
        # Fill the model from the provider default only if nothing was given,
        # so an explicit model always wins.
        if not self.model:
            self.model = get_provider_spec(self.provider).default_model

    # ------------------------------------------------------------------
    @property
    def spec(self) -> ProviderSpec:
        """Static description of the selected provider."""
        return get_provider_spec(self.provider)

    @classmethod
    def from_env(
        cls,
        api_key: str | None = None,
        dotenv_path: str | Path | None = ".env",
        provider: str | Provider | None = None,
    ) -> "AIConfig":
        """Build a config from the environment.

        Provider resolution: explicit argument, then ``DPI_LLM_PROVIDER``,
        then :data:`DEFAULT_PROVIDER`.  An unrecognised name is **not**
        silently replaced — it is recorded in :attr:`invalid_provider` so the
        caller can refuse to run rather than quietly using a different
        provider than the one asked for.

        Key resolution, per provider: explicit argument, then that provider's
        own variable (``GROQ_API_KEY``, ``OPENAI_API_KEY``, ...).
        """
        if dotenv_path is not None:
            load_dotenv(dotenv_path)

        invalid: str | None = None
        if isinstance(provider, Provider):
            resolved = provider
        else:
            requested = provider or os.environ.get(ENV_PROVIDER) or DEFAULT_PROVIDER
            parsed = parse_provider(requested)
            if parsed is None:
                invalid = str(requested)
                resolved = parse_provider(DEFAULT_PROVIDER) or Provider.GROQ
            else:
                resolved = parsed

        spec = get_provider_spec(resolved)

        # --- key -----------------------------------------------------------
        key = api_key if api_key is not None else os.environ.get(spec.api_key_env)
        if key is not None:
            key = key.strip() or None
        # Ollama's OpenAI-compatible layer requires the header to exist but
        # ignores its value, so supply a placeholder rather than demanding one.
        if key is None and not spec.requires_api_key:
            key = "ollama-local"

        # --- model ---------------------------------------------------------
        # Provider-specific variable wins, then the generic override.
        model = (
            os.environ.get(spec.model_env)
            or os.environ.get(_ENV_MODEL)
            or spec.default_model
        ).strip() or spec.default_model

        # --- endpoint ------------------------------------------------------
        base_url = (
            os.environ.get(spec.base_url_env)
            or os.environ.get(_ENV_BASE_URL)
            or spec.default_base_url
        )

        raw_mode = os.environ.get(_ENV_IP_MODE, IPRedactionMode.REDACT_PRIVATE.value)
        try:
            ip_mode = IPRedactionMode(raw_mode.strip().lower())
        except ValueError:
            ip_mode = IPRedactionMode.REDACT_PRIVATE

        return cls(
            provider=resolved,
            api_key=key,
            model=model,
            base_url=base_url or None,
            timeout_seconds=_env_float(_ENV_TIMEOUT, DEFAULT_TIMEOUT_SECONDS),
            max_retries=_env_int(_ENV_MAX_RETRIES, DEFAULT_MAX_RETRIES),
            max_flows=_env_int(_ENV_MAX_FLOWS, DEFAULT_MAX_FLOWS),
            ip_mode=ip_mode,
            invalid_provider=invalid,
        )

    def has_api_key(self) -> bool:
        """Whether a usable key is present.  Makes no network call.

        Always ``True`` for providers that do not need one (Ollama).
        """
        if not self.spec.requires_api_key:
            return True
        return bool(self.api_key)

    def __repr__(self) -> str:
        """Mask the key.  A config object must be safe to log or print."""
        key = "set" if self.api_key else "missing"
        return (
            f"AIConfig(provider={self.provider.value}, api_key=<{key}>, "
            f"model={self.model!r}, base_url={self.base_url!r}, "
            f"timeout={self.timeout_seconds}s, max_retries={self.max_retries}, "
            f"max_flows={self.max_flows}, ip_mode={self.ip_mode.value})"
        )
