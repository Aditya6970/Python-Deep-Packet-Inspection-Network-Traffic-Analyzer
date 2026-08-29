"""LLM provider registry and structured-output capability negotiation.

The analyzer must not know which provider it is talking to. This module holds
everything provider-specific: endpoints, environment variable names, default
models, and — the part that actually differs between them — *how* each one can
be made to return schema-conforming JSON.

Why one SDK for three providers
-------------------------------
Groq and Ollama both expose OpenAI-compatible ``/chat/completions`` endpoints,
so all three providers are driven through the ``openai`` package already in
``requirements.txt``. No ``groq`` SDK, no ``ollama`` SDK, no extra dependency.

Structured outputs are not uniform
----------------------------------
This is the one real difference, and pretending otherwise would produce silent
failures. Providers sit in one of three tiers:

``NATIVE_PARSE``
    ``client.beta.chat.completions.parse(response_format=PydanticModel)``.
    The SDK constrains generation and hands back a parsed object. OpenAI.

``JSON_SCHEMA``
    ``response_format={"type": "json_schema", ...}`` with a JSON Schema and a
    ``strict`` flag. Groq supports this on its ``openai/gpt-oss-*`` models.
    Content comes back as a JSON string that we parse and validate ourselves.

``JSON_OBJECT``
    ``response_format={"type": "json_object"}`` — valid JSON is guaranteed,
    conformance to *our* schema is not. The schema is described in the prompt
    instead, and validation catches anything that drifts. This is the Ollama
    path and the universal fallback.

Every tier ends at the same place: the JSON is validated against
:class:`~ai.schemas.AnalysisResult`. There is one canonical schema, not one per
provider.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Sequence

__all__ = [
    "Provider",
    "StructuredMode",
    "ProviderSpec",
    "PROVIDERS",
    "get_provider_spec",
    "parse_provider",
    "to_strict_json_schema",
    "PROVIDER_OMITTED_FIELDS",
    "unenforced_constants",
]


class Provider(str, Enum):
    """Supported LLM providers."""

    GROQ = "groq"
    OLLAMA = "ollama"
    OPENAI = "openai"


class StructuredMode(str, Enum):
    """How a provider can be made to return schema-conforming JSON."""

    NATIVE_PARSE = "native_parse"
    JSON_SCHEMA = "json_schema"
    JSON_OBJECT = "json_object"


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """Static description of one provider."""

    provider: Provider
    label: str
    #: Endpoint. ``None`` means the SDK default (OpenAI's own).
    default_base_url: str | None
    #: Environment variable holding the API key.
    api_key_env: str
    #: Environment variable overriding the model.
    model_env: str
    #: Environment variable overriding the endpoint.
    base_url_env: str
    default_model: str
    structured_mode: StructuredMode
    #: Whether a real key is needed. Ollama ignores it entirely.
    requires_api_key: bool
    #: Whether the provider guarantees schema conformance in its top tier.
    strict_schema: bool
    setup_hint: str
    #: Largest request, in estimated input tokens, worth sending to this
    #: provider.  ``None`` means "no locally-known limit"; the provider decides.
    #:
    #: This is not the model's context window.  It is the point past which a
    #: request is *rejected by policy* rather than answered -- a per-minute
    #: token allowance, most often -- which is a different and much lower
    #: number.  Where that number is known it is worth checking before spending
    #: a round trip on a request that cannot succeed.
    max_input_tokens: int | None = None
    #: What to tell someone whose request exceeded :attr:`max_input_tokens`.
    oversize_hint: str = ""


PROVIDERS: Final[dict[Provider, ProviderSpec]] = {
    Provider.GROQ: ProviderSpec(
        provider=Provider.GROQ,
        label="Groq",
        default_base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        model_env="GROQ_MODEL",
        base_url_env="GROQ_BASE_URL",
        # Groq's production line-up moves; `openai/gpt-oss-20b` is a current
        # production model AND one of the two that support strict structured
        # outputs, which is exactly what this task needs.  Override with
        # GROQ_MODEL rather than editing this file.
        default_model="openai/gpt-oss-20b",
        structured_mode=StructuredMode.JSON_SCHEMA,
        requires_api_key=True,
        strict_schema=True,
        setup_hint=(
            "Get a free key at https://console.groq.com/keys, then set "
            "GROQ_API_KEY in your environment or .env file. Groq retires "
            "models periodically — if you see a model_not_found error, check "
            "https://console.groq.com/docs/models and set GROQ_MODEL."
        ),
        # Deliberately None, despite Groq being the provider where oversized
        # requests actually bite. Groq meters tokens per minute and rejects a
        # single request larger than the whole allowance with HTTP 413
        # "Request too large" — not throttled, rejected. But that allowance is
        # a property of the *account tier*, which this code cannot observe: a
        # free key and a paid key differ by more than an order of magnitude.
        #
        # A guessed ceiling would refuse requests that would have succeeded,
        # which is a worse failure than the one it prevents — so the guess is
        # not made. What happens instead: an actual 413 is classified as
        # REQUEST_TOO_LARGE rather than a bare API_ERROR, and the hint below is
        # shown either way. Someone who knows their own limit sets
        # DPI_MAX_INPUT_TOKENS and gets the refusal before the round trip.
        max_input_tokens=None,
        oversize_hint=(
            "Reduce the request with --rag-max-items/--rag-max-chars, or "
            "--max-flows to send fewer flows; raise the ceiling with "
            "DPI_MAX_INPUT_TOKENS if your account's rate limit allows it."
        ),
    ),
    Provider.OLLAMA: ProviderSpec(
        provider=Provider.OLLAMA,
        label="Ollama (local)",
        default_base_url="http://localhost:11434/v1",
        api_key_env="OLLAMA_API_KEY",
        model_env="OLLAMA_MODEL",
        base_url_env="OLLAMA_BASE_URL",
        default_model="llama3.1",
        # Ollama's OpenAI-compatible layer documents JSON mode, not json_schema,
        # so we take the universal path and validate ourselves.
        structured_mode=StructuredMode.JSON_OBJECT,
        requires_api_key=False,
        strict_schema=False,
        setup_hint=(
            "Install from https://ollama.com, then run `ollama serve` and "
            "`ollama pull llama3.1`. No API key is needed. Set OLLAMA_MODEL to "
            "use a different local model."
        ),
    ),
    Provider.OPENAI: ProviderSpec(
        provider=Provider.OPENAI,
        label="OpenAI",
        default_base_url=None,
        api_key_env="OPENAI_API_KEY",
        model_env="OPENAI_MODEL",
        base_url_env="OPENAI_BASE_URL",
        default_model="gpt-4o-mini",
        structured_mode=StructuredMode.NATIVE_PARSE,
        requires_api_key=True,
        strict_schema=True,
        setup_hint=(
            "Set OPENAI_API_KEY. Note that OpenAI's API requires paid credits; "
            "a key on an account with no quota returns a rate-limit error."
        ),
    ),
}


def parse_provider(name: str | None) -> Provider | None:
    """Parse a provider name, returning ``None`` if it is not recognised.

    Returning ``None`` rather than raising lets the caller report an
    actionable error instead of crashing, and — importantly — lets it refuse
    to silently fall back to a different provider.
    """
    if not name:
        return None
    try:
        return Provider(name.strip().lower())
    except ValueError:
        return None


def get_provider_spec(provider: Provider) -> ProviderSpec:
    """Return the static spec for a provider."""
    return PROVIDERS[provider]


# ---------------------------------------------------------------------------
# JSON Schema conversion
# ---------------------------------------------------------------------------
#: Root properties withheld from a provider-enforced schema.
#:
#: ``schema_version`` is ours, not the model's.  Pydantic renders it as
#: ``{"const": "1.1"}``, and ``const`` is the one keyword in this schema that a
#: constrained decoder may not enforce while the provider's own validator still
#: checks it.  When those two disagree the model emits some other string, the
#: completion fails validation server-side, and the call comes back HTTP 400
#: ``code=json_validate_failed`` -- which is exactly what one live case did.
#:
#: It is also the only field the model has no way to get right: nothing in the
#: prompt tells it our schema version, so it is guessing a literal.  Asking a
#: language model to reproduce a constant we already know is a request that can
#: only fail, so it is not asked.  :class:`~ai.schemas.AnalysisResult` keeps
#: the field, keeps the ``Literal`` and keeps validating it; the default fills
#: it in when the response arrives.
PROVIDER_OMITTED_FIELDS: Final[tuple[str, ...]] = ("schema_version",)


def unenforced_constants(schema: dict[str, Any]) -> list[str]:
    """Paths in a strict schema that use ``const``, which decoders may ignore.

    A standing check rather than a one-off fix.  ``const`` is safe when the
    provider constrains generation to it and unsafe when the provider only
    validates afterwards, and which of those happens is not something this
    project can see.  So the rule is simply that a provider-enforced schema
    carries none: a future ``Literal`` field either gets omitted like
    ``schema_version`` or gets a deliberate decision, and the test that calls
    this refuses to let it pass unnoticed.
    """
    found: list[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            if "const" in node:
                found.append(path or "<root>")
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else key)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(schema, "")
    return sorted(found)


def to_strict_json_schema(schema: dict[str, Any],
                          omit: Sequence[str] = PROVIDER_OMITTED_FIELDS
                          ) -> dict[str, Any]:
    """Adapt a Pydantic JSON Schema for strict structured-output mode.

    Providers that guarantee conformance (OpenAI, Groq on gpt-oss) impose
    rules a stock Pydantic schema does not satisfy:

    * every object must set ``additionalProperties: false``
    * every property must appear in ``required`` — optional fields are not
      allowed, even when they have defaults
    * unsupported validation keywords must be removed

    Making a field "required" here does not change our own model: Pydantic
    still applies its defaults and constraints when validating the response.
    It only tells the provider to always emit the key.

    Operates on a deep copy; the input is not modified.
    """
    result = copy.deepcopy(schema)

    # Withheld before strictification, so the omitted names never reach
    # ``required`` either.  Dropping a property but leaving it required would
    # produce a schema nothing can satisfy.
    properties = result.get("properties")
    if isinstance(properties, dict):
        for name in omit:
            properties.pop(name, None)

    _strictify(result)

    # $defs are referenced by the root, so they need the same treatment.
    for definition in result.get("$defs", {}).values():
        _strictify(definition)

    return result


#: Keywords a strict schema endpoint may reject.  Pydantic emits several of
#: these from Field(min_length=..., ge=..., etc.); our own validation still
#: enforces them after the response comes back.
_UNSUPPORTED_KEYWORDS: Final[tuple[str, ...]] = (
    "minLength",
    "maxLength",
    "pattern",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minItems",
    "maxItems",
    "uniqueItems",
    "format",
    "default",
)


def _strictify(node: Any) -> None:
    """Recursively apply strict-mode rules in place."""
    if isinstance(node, dict):
        for keyword in _UNSUPPORTED_KEYWORDS:
            node.pop(keyword, None)

        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"].keys())

        for value in node.values():
            _strictify(value)

    elif isinstance(node, list):
        for item in node:
            _strictify(item)
