"""Ranked-retrieval metrics: Hit@K, Recall@K, Precision@K, MRR.

Deliberately the simple four.  A ranked list of ids and a set of relevant ids
is all the information this project has -- there are no graded relevance
judgements, so nDCG would be nDCG over binary labels, which is a more
complicated way to say the same thing.  When a metric here stops being
informative, that is a reason to collect better labels, not to compute a
fancier average of the ones we have.

Undefined is ``None``, never zero
---------------------------------
A case with no relevant documents has no recall: the denominator is zero.
Reporting that as ``0.0`` would drag an aggregate down as though the system had
failed, when in fact the question was never asked.  Every metric returns
``None`` when it is undefined, and :func:`aggregate` skips those rather than
counting them -- so an average always names how many cases it actually covers.

Duplicates
----------
A ranked list may legitimately contain the same document twice: two chunks of
one document are two results but one document.  :func:`deduplicate` collapses
them to first occurrence, and every document-level metric applies it, so rank 1
means "the best-ranked result from this document" rather than "some copy of
it".  Chunk-level metrics take the list as given.
"""

from __future__ import annotations

from typing import Final, Hashable, Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "DEFAULT_K_VALUES",
    "MetricSummary",
    "RetrievalMetrics",
    "aggregate",
    "deduplicate",
    "hit_at_k",
    "mean_reciprocal_rank",
    "precision_at_k",
    "recall_at_k",
    "score_ranking",
]

#: Cut-offs reported by default.  Small, because the corpus is small: at 37
#: chunks and a final_top_k of 8, K above 10 measures nothing.
DEFAULT_K_VALUES: Final[tuple[int, ...]] = (1, 3, 5, 8)


def deduplicate(ranked: Sequence[Hashable]) -> list[Hashable]:
    """First occurrence of each id, order preserved."""
    seen: set[Hashable] = set()
    unique: list[Hashable] = []
    for item in ranked:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _prepared(ranked: Sequence[Hashable], k: int, dedupe: bool) -> list[Hashable]:
    if k < 0:
        raise ValueError(f"k must be zero or positive, got {k}")
    items = deduplicate(ranked) if dedupe else list(ranked)
    return items[:k]


def hit_at_k(
    ranked: Sequence[Hashable],
    relevant: Iterable[Hashable],
    k: int,
    dedupe: bool = True,
) -> float | None:
    """1.0 if any relevant id appears in the top ``k``, else 0.0.

    ``None`` when nothing is relevant -- there is no hit to find.
    """
    targets = set(relevant)
    if not targets:
        return None
    return 1.0 if targets & set(_prepared(ranked, k, dedupe)) else 0.0


def recall_at_k(
    ranked: Sequence[Hashable],
    relevant: Iterable[Hashable],
    k: int,
    dedupe: bool = True,
) -> float | None:
    """Share of the relevant ids that appear in the top ``k``.

    ``None`` when nothing is relevant.  Note the ceiling: with five relevant
    documents and ``k=3``, recall cannot exceed 0.6, which is a property of the
    question rather than a failure of the system.
    """
    targets = set(relevant)
    if not targets:
        return None
    found = targets & set(_prepared(ranked, k, dedupe))
    return len(found) / len(targets)


def precision_at_k(
    ranked: Sequence[Hashable],
    relevant: Iterable[Hashable],
    k: int,
    dedupe: bool = True,
) -> float | None:
    """Share of the top ``k`` results that are relevant.

    Divided by the number of results actually returned, not by ``k``: when
    only two results exist, ``precision@5`` asks about those two.  Dividing by
    ``k`` would report a system that returned two perfect answers as 0.4.

    ``None`` when nothing was returned, or when nothing is relevant.
    """
    targets = set(relevant)
    if not targets:
        return None
    top = _prepared(ranked, k, dedupe)
    if not top:
        return None
    return sum(1 for item in top if item in targets) / len(top)


def mean_reciprocal_rank(
    ranked: Sequence[Hashable],
    relevant: Iterable[Hashable],
    dedupe: bool = True,
) -> float | None:
    """``1 / rank`` of the first relevant id, or 0.0 if none appears.

    Single-query reciprocal rank; the mean across cases is taken by
    :func:`aggregate`.  ``None`` when nothing is relevant.
    """
    targets = set(relevant)
    if not targets:
        return None
    items = deduplicate(ranked) if dedupe else list(ranked)
    for position, item in enumerate(items, start=1):
        if item in targets:
            return 1.0 / position
    return 0.0


class RetrievalMetrics(BaseModel):
    """Every metric for one ranking, at one cut-off."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    k: int = Field(ge=0)
    level: str = Field(description='"document" or "chunk".')
    returned: int = Field(ge=0, description="Results the ranking contained.")
    considered: int = Field(ge=0, description="Results inside the cut-off.")
    relevant_total: int = Field(ge=0, description="Labelled relevant ids for this case.")
    relevant_found: int = Field(ge=0)
    irrelevant_found: int = Field(
        ge=0, description="Results labelled explicitly irrelevant, inside the cut-off."
    )
    hit: float | None = None
    recall: float | None = None
    precision: float | None = None
    reciprocal_rank: float | None = None

    @model_validator(mode="after")
    def _consistent(self) -> RetrievalMetrics:
        if self.relevant_found > self.relevant_total:
            raise ValueError("more relevant results found than exist")
        if self.considered > self.returned:
            raise ValueError("considered more results than were returned")
        return self

    def row(self) -> str:
        """One fixed-width line, for the console report."""
        def show(value: float | None) -> str:
            return " n/a " if value is None else f"{value:.2f}"

        return (f"K={self.k:<2} hit={show(self.hit)} recall={show(self.recall)} "
                f"prec={show(self.precision)} rr={show(self.reciprocal_rank)} "
                f"({self.relevant_found}/{self.relevant_total} relevant, "
                f"{self.irrelevant_found} irrelevant)")


def score_ranking(
    ranked: Sequence[Hashable],
    relevant: Iterable[Hashable],
    k: int,
    level: str = "document",
    irrelevant: Iterable[Hashable] = (),
    dedupe: bool = True,
) -> RetrievalMetrics:
    """Score one ranking at one cut-off.

    ``irrelevant`` is the explicitly-labelled negative set -- ids a correct
    system should not surface for this case.  It is counted, never subtracted:
    an id that is neither relevant nor labelled irrelevant is simply unjudged,
    which is the honest state for most of a corpus.
    """
    targets = set(relevant)
    negatives = set(irrelevant)
    items = deduplicate(ranked) if dedupe else list(ranked)
    top = items[:k] if k >= 0 else []

    return RetrievalMetrics(
        k=k,
        level=level,
        returned=len(items),
        considered=len(top),
        relevant_total=len(targets),
        relevant_found=sum(1 for item in top if item in targets),
        irrelevant_found=sum(1 for item in top if item in negatives),
        hit=hit_at_k(ranked, targets, k, dedupe),
        recall=recall_at_k(ranked, targets, k, dedupe),
        precision=precision_at_k(ranked, targets, k, dedupe),
        reciprocal_rank=mean_reciprocal_rank(ranked, targets, dedupe),
    )


class MetricSummary(BaseModel):
    """Averages across cases, with the count each average is over."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    k: int = Field(ge=0)
    level: str
    cases: int = Field(ge=0, description="Cases contributing at least one defined metric.")
    hit: float | None = None
    recall: float | None = None
    precision: float | None = None
    mrr: float | None = None
    irrelevant_found: int = Field(default=0, ge=0)

    def row(self) -> str:
        def show(value: float | None) -> str:
            return " n/a " if value is None else f"{value:.3f}"

        return (f"K={self.k:<2} hit@K={show(self.hit)} recall@K={show(self.recall)} "
                f"prec@K={show(self.precision)} MRR={show(self.mrr)} "
                f"[{self.cases} case(s), {self.irrelevant_found} irrelevant hit(s)]")


def _mean(values: Sequence[float | None]) -> float | None:
    """Mean of the defined values, or ``None`` if there are none."""
    defined = [value for value in values if value is not None]
    return sum(defined) / len(defined) if defined else None


def aggregate(results: Sequence[RetrievalMetrics], k: int, level: str) -> MetricSummary:
    """Average one cut-off across cases, ignoring undefined values.

    ``cases`` reports how many contributed, so a headline number can never
    quietly be an average of two.
    """
    rows = [row for row in results if row.k == k and row.level == level]
    contributing = [row for row in rows
                    if any(value is not None for value in
                           (row.hit, row.recall, row.precision, row.reciprocal_rank))]
    return MetricSummary(
        k=k,
        level=level,
        cases=len(contributing),
        hit=_mean([row.hit for row in rows]),
        recall=_mean([row.recall for row in rows]),
        precision=_mean([row.precision for row in rows]),
        mrr=_mean([row.reciprocal_rank for row in rows]),
        irrelevant_found=sum(row.irrelevant_found for row in rows),
    )
