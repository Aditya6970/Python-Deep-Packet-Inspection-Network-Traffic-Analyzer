"""Signal/knowledge compatibility: is this document *about* what was observed?

The problem this solves
-----------------------
Cosine similarity answers "does this text talk about the same things as the
query?".  It does not answer "is this document appropriate for the traffic that
was actually seen?".  Those come apart in a predictable way.

A capture of ordinary browsing fires ``baseline_web_browsing`` and
``quic_present``.  Its capture-wide query mentions protocols, ports and flow
metadata, and the phrase *deep packet inspection*.  A DNS-tunneling document is
written in exactly that vocabulary -- ports, protocols, flow counts, DPI -- so
it scores well.  It is topically related and situationally wrong, and no
threshold on cosine separates the two, because the score is not measuring the
thing that is wrong.

The signal that separates them is already in the corpus
-------------------------------------------------------
Every knowledge document carries ``applies_to``: the signal names its author
said it speaks to.  That list was written in step 1, from the DPI schema, long
before any retrieval was measured -- so using it here is reading a declaration,
not fitting a parameter to the evaluation set.

Comparing ``applies_to`` against the signals that actually fired gives three
states, and only three:

``declared``
    The document declares at least one signal this capture produced.  The
    corpus author expected this connection.
``unscoped``
    The document declares no signals at all.  It claims general applicability,
    so there is nothing to contradict; it is judged on similarity alone.
``undeclared``
    The document declares signals, and none of them fired.  It is scoped, and
    this capture is outside its scope.

Similarity is preserved, never replaced
---------------------------------------
Nothing here recomputes, rescales or hides a cosine score.  Compatibility
produces a **tier**, and ranking becomes ``(tier, -similarity, chunk_id)``:
within a tier the original cosine order is untouched, and the original score
travels with every result.  A chunk is never removed -- an ``undeclared`` chunk
still appears once the compatible ones are exhausted -- and every adjusted
result carries a sentence saying why.

That ordering is deliberate rather than additive.  An additive bonus needs a
magnitude, and the right magnitude depends on how this particular model happens
to score this particular corpus; a number picked without that measurement is a
guess wearing a decimal point.  A tier needs no magnitude.

What this is not
----------------
Not a relevance label, not a filter, and not a second similarity.  It does not
know which documents the evaluation set considers correct, contains no case
ids, and would behave identically on a corpus this project has never seen.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final, Iterable, Mapping, Sequence

__all__ = [
    "AffinityMode",
    "Compatibility",
    "SignalAffinity",
    "TIER",
    "assess",
    "assess_many",
]

#: Most signal names listed in a note before it is abbreviated.  A note is read
#: by a person in a report; the full list of six is noise.
_MAX_NAMED: Final[int] = 3


class Compatibility(str, Enum):
    """How a document's declared scope compares with the observed signals."""

    #: ``applies_to`` names at least one signal this capture produced.
    DECLARED = "declared"
    #: ``applies_to`` is empty: the document claims no particular scope.
    UNSCOPED = "unscoped"
    #: ``applies_to`` is non-empty and none of it fired.
    UNDECLARED = "undeclared"


#: Rank order.  Lower sorts first.  ``unscoped`` sits between the two because a
#: document that declares nothing has made no claim to contradict -- demoting it
#: to the bottom would punish generality, and promoting it to the top would
#: reward silence.
TIER: Final[dict[Compatibility, int]] = {
    Compatibility.DECLARED: 0,
    Compatibility.UNSCOPED: 1,
    Compatibility.UNDECLARED: 2,
}


class AffinityMode(str, Enum):
    """Whether compatibility takes part in ranking."""

    #: Rank on cosine similarity alone -- the behaviour before this existed.
    OFF = "off"
    #: Rank by ``(tier, -similarity, chunk_id)``.
    RANK = "rank"


@dataclass(frozen=True, slots=True)
class SignalAffinity:
    """The verdict for one chunk, and the sentence explaining it."""

    compatibility: Compatibility
    tier: int
    #: Observed signals this document declares, sorted.  Empty unless ``declared``.
    declared_matches: tuple[str, ...]
    #: Signals the document declares, sorted.  Empty when it declares none.
    declared_scope: tuple[str, ...]
    #: Why the result was ranked where it was, in one sentence.
    note: str

    @property
    def adjusted(self) -> bool:
        """Whether ranking treated this chunk as anything other than neutral."""
        return self.tier != TIER[Compatibility.UNSCOPED]

    def as_dict(self) -> dict[str, object]:
        return {
            "compatibility": self.compatibility.value,
            "tier": self.tier,
            "declared_matches": list(self.declared_matches),
            "note": self.note,
        }


def _abbreviate(names: Sequence[str]) -> str:
    """``a, b, c and 2 more`` -- deterministic, and bounded in length."""
    if len(names) <= _MAX_NAMED:
        return ", ".join(names)
    shown = ", ".join(names[:_MAX_NAMED])
    return f"{shown} and {len(names) - _MAX_NAMED} more"


def assess(applies_to: Iterable[str], fired: Iterable[str]) -> SignalAffinity:
    """Classify one document's scope against the signals a capture produced.

    ``applies_to`` is the document's own declaration; ``fired`` is the set of
    :class:`~ai.rag.signals.SignalType` values present in the signal report.
    Both are read as plain strings, so this function needs neither the corpus
    nor the signal module and can be checked on its own.

    When nothing fired there is no observation to be compatible *with*, so
    every document is reported ``unscoped``: a capture that produced no signals
    must not have its whole corpus demoted on the strength of an empty set.
    """
    scope = tuple(sorted({name for name in applies_to if name}))
    observed = frozenset(name for name in fired if name)

    if not observed:
        return SignalAffinity(
            compatibility=Compatibility.UNSCOPED,
            tier=TIER[Compatibility.UNSCOPED],
            declared_matches=(),
            declared_scope=scope,
            note="No signals fired, so nothing constrains which notes apply; "
                 "ranked on similarity alone.",
        )

    if not scope:
        return SignalAffinity(
            compatibility=Compatibility.UNSCOPED,
            tier=TIER[Compatibility.UNSCOPED],
            declared_matches=(),
            declared_scope=(),
            note="This note declares no signal scope, so it is ranked on "
                 "similarity alone.",
        )

    matches = tuple(name for name in scope if name in observed)
    if matches:
        return SignalAffinity(
            compatibility=Compatibility.DECLARED,
            tier=TIER[Compatibility.DECLARED],
            declared_matches=matches,
            declared_scope=scope,
            note=f"Ranked first: this note declares {_abbreviate(matches)}, "
                 f"which this capture produced.",
        )

    return SignalAffinity(
        compatibility=Compatibility.UNDECLARED,
        tier=TIER[Compatibility.UNDECLARED],
        declared_matches=(),
        declared_scope=scope,
        note=f"Ranked after compatible notes: this note applies to "
             f"{_abbreviate(scope)}, none of which this capture produced.",
    )


def assess_many(
    scopes: Mapping[str, Sequence[str]], fired: Iterable[str]
) -> dict[str, SignalAffinity]:
    """:func:`assess` over a mapping of id to ``applies_to``.

    Convenience for the retrieval merge, which holds chunks by id.  Iteration
    order of the input is irrelevant: the result is a mapping and the caller
    sorts.
    """
    observed = tuple(sorted({name for name in fired if name}))
    return {key: assess(value, observed) for key, value in scopes.items()}
