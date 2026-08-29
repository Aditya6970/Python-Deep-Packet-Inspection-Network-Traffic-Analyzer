"""Named retrieval/budget configurations, and the arithmetic that compares them.

Why this is a module and not a loop in the runner
-------------------------------------------------
Step 9 shipped a change on the strength of an argument and had it rejected by
the first real measurement.  The lesson taken here is not "argue less" -- it is
that the *accounting* which decides such a question has to be inspectable and
testable on its own, separately from the model run that produces the inputs.

So this module holds two things and nothing else:

* :data:`CANDIDATES` -- the configurations under consideration, as data.
* :class:`CaseAccount` / :class:`CandidateAccount` / :func:`assess` -- pure
  arithmetic over numbers someone else measured.

It loads no model, builds no index and reaches no network.  Everything here can
be checked against worked examples, and is, in ``run_rag_eval_tests.py``.
``run_rag_evaluation.py`` supplies the real measurements.

What the comparison is actually for
-----------------------------------
Not maximum recall.  Recall is trivially maximised by supplying the whole
corpus, and the live run says what that costs: five of six live cases failed
with HTTP 413 or 429, and the largest request measured about 7,240 estimated
tokens.  A configuration that retrieves perfectly and cannot be sent has a
recall of zero in the only sense that matters.

The question is **evidence per token**: which configuration puts the most
relevant knowledge in front of the model, without supplying more irrelevant
knowledge than the baseline, and without growing a request that is already
failing.

Where the request actually goes
-------------------------------
The context budget bounds the knowledge block.  It does not bound the request,
and the knowledge block is the smaller half of it by a wide margin -- the
capture JSON is most of what is sent.  :class:`CaseAccount` therefore records
the capture-only size alongside the knowledge size, because a change to the
knowledge budget that is framed as fixing a request-size problem should have to
show what fraction of the request it is touching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Mapping, Sequence

__all__ = [
    "CANDIDATES",
    "OBSERVED_FAILING_PROMPT_TOKENS",
    "Candidate",
    "CaseAccount",
    "CandidateAccount",
    "Verdict",
    "assess",
    "by_name",
    "rank",
]


#: An estimated total prompt size that has been **observed to fail**.
#:
#: Provenance, so this number is never mistaken for a limit someone looked up:
#: it is the largest estimated prompt in this project's own live Groq run, in
#: which one of six cases was analysed and five failed with HTTP 413
#: ``request_too_large`` or HTTP 429 ``rate_limit_exceeded``.
#:
#: What it establishes is one-sided and that is the whole point.  A request of
#: this size has failed; it does not follow that anything smaller succeeds, and
#: the exact ceiling was not measured and is a property of the account tier
#: rather than of this code.  So it is used **only to flag a candidate as
#: known-at-risk in a report**.  It guards nothing, blocks nothing, and is not
#: wired into any runtime path.
OBSERVED_FAILING_PROMPT_TOKENS: Final[int] = 7240


@dataclass(frozen=True, slots=True)
class Candidate:
    """One named configuration: how to retrieve, and how much may be supplied."""

    name: str
    description: str
    #: Keyword arguments for :class:`~ai.rag.retrieval.RetrievalConfig`.
    retrieval: Mapping[str, Any]
    #: Keyword arguments for :class:`~ai.rag.context.KnowledgeContextConfig`.
    budget: Mapping[str, Any]
    #: Measured and reported, never selected.  See :data:`CANDIDATES`.
    reference_only: bool = False

    def summary(self) -> str:
        """One line naming every parameter that differs from a bare default."""
        retrieval = ", ".join(f"{key}={value}" for key, value in self.retrieval.items())
        budget = ", ".join(f"{key}={value}" for key, value in self.budget.items())
        return f"retrieval({retrieval})  budget({budget})"


#: The configurations compared in step 10.
#:
#: ``baseline`` is the shipped configuration, restated explicitly rather than
#: left implicit, so the table does not depend on what the defaults happen to be
#: on the day it is run.
#:
#: ``A`` through ``D`` share one retrieval shape -- the shape the step 8 sweep
#: pointed at: more candidates per query, one chunk per document, and the
#: highest similarity floor that cost no recall (0.75; 0.80 dropped recall to
#: 0.73 and 0.85 collapsed it to 0.23).  They differ only in what the budget
#: then allows through, which isolates the budget question from the retrieval
#: question.
#:
#: ``A2`` is not one of the four that were asked for, and it is here because
#: the shape sweep points at it and none of the four try it.  ``A`` through
#: ``D`` all cap a document at one chunk, and the sweep measured that cap as the
#: *worst* setting for noise: ``max_per_document=1`` gave precision 0.49 with
#: four irrelevant documents retrieved, while releasing the cap entirely gave
#: precision 0.55 with one.  Six documents and eight slots is the reason -- a
#: cap of one fills the result with the whole corpus, and the notes that do not
#: apply come along with the ones that do.  ``A2`` is ``A`` with that one
#: parameter changed, so the comparison isolates it.
#:
#: ``unbounded`` exists to measure the ceiling: the best any budget could do,
#: and what it would cost to send.  It is marked ``reference_only`` and is
#: excluded from selection by :func:`rank`, because "supply everything" is the
#: configuration whose live failures started this.
CANDIDATES: Final[tuple[Candidate, ...]] = (
    Candidate(
        name="baseline",
        description="The shipped configuration.",
        retrieval={"per_query_top_k": 4, "final_top_k": 8,
                   "max_per_document": 2, "min_similarity": None},
        budget={"max_items": 4, "max_chars": 3000, "max_total_tokens": 900},
    ),
    Candidate(
        name="A",
        description="Better retrieval shape, unchanged budget.",
        retrieval={"per_query_top_k": 6, "final_top_k": 8,
                   "max_per_document": 1, "min_similarity": 0.75},
        budget={"max_items": 4, "max_chars": 3000, "max_total_tokens": 900},
    ),
    Candidate(
        name="B",
        description="Shape A, generous budget: 6 / 6000 / 1200.",
        retrieval={"per_query_top_k": 6, "final_top_k": 8,
                   "max_per_document": 1, "min_similarity": 0.75},
        budget={"max_items": 6, "max_chars": 6000, "max_total_tokens": 1200},
    ),
    Candidate(
        name="C",
        description="Shape A, four longer excerpts: 4 / 6000 / 1200.",
        retrieval={"per_query_top_k": 6, "final_top_k": 8,
                   "max_per_document": 1, "min_similarity": 0.75},
        budget={"max_items": 4, "max_chars": 6000, "max_total_tokens": 1200},
    ),
    Candidate(
        name="D",
        description="Shape A, six shorter excerpts: 6 / 4000 / 1200.",
        retrieval={"per_query_top_k": 6, "final_top_k": 8,
                   "max_per_document": 1, "min_similarity": 0.75},
        budget={"max_items": 6, "max_chars": 4000, "max_total_tokens": 1200},
    ),
    Candidate(
        name="D-partial",
        description="D's budget and query breadth, without D's retrieval shape.",
        # Not one of the originally-proposed candidates. It exists because the
        # step 11 adoption request named three changes -- per_query_top_k 4->6,
        # max_chars 3000->4000, max_total_tokens 900->1200 -- and described them
        # as "candidate D". D is six changes: it also sets max_per_document 2->1,
        # min_similarity None->0.75 and max_items 4->6, and the same request
        # explicitly forbids touching the first two.
        #
        # So the three-change set is a *different configuration*, and D's
        # measured numbers do not describe it: three of the six parameters that
        # produced them would be missing. Rather than assume the difference is
        # immaterial or refuse the question, the configuration is named here and
        # measured alongside the others, so the next evaluation run answers it.
        retrieval={"per_query_top_k": 6, "final_top_k": 8,
                   "max_per_document": 2, "min_similarity": None},
        budget={"max_items": 4, "max_chars": 4000, "max_total_tokens": 1200},
    ),
    # -- attribution probes ---------------------------------------------
    # If D-partial does not reproduce D, the next question is *which*
    # parameter is responsible, and guessing is how step 9 went wrong. These
    # three change exactly one thing each away from D, so one run attributes
    # the difference instead of narrowing it over three runs.
    Candidate(
        name="D-items4",
        description="D with max_items 6->4. Tests whether the item count ever binds.",
        # The step 10 run supplied at most four excerpts in any case under D,
        # against an allowance of six -- so the char and token ceilings bound
        # first and the item count never did. If that holds, this row is
        # identical to D and the max_items difference between D and D-partial
        # is inert. A row that differs would falsify it, which is the point.
        retrieval={"per_query_top_k": 6, "final_top_k": 8,
                   "max_per_document": 1, "min_similarity": 0.75},
        budget={"max_items": 4, "max_chars": 4000, "max_total_tokens": 1200},
    ),
    Candidate(
        name="D-mpd2",
        description="D with max_per_document 1->2. Isolates the per-document cap.",
        retrieval={"per_query_top_k": 6, "final_top_k": 8,
                   "max_per_document": 2, "min_similarity": 0.75},
        budget={"max_items": 6, "max_chars": 4000, "max_total_tokens": 1200},
    ),
    Candidate(
        name="D-nofloor",
        description="D with min_similarity 0.75->None. Isolates the similarity floor.",
        # The step 10 threshold sweep found every floor from None to 0.75
        # produced identical rows; only 0.80 changed anything. If that carries
        # over to D's retrieval shape this row equals D, and the floor is
        # decoration rather than a working part.
        retrieval={"per_query_top_k": 6, "final_top_k": 8,
                   "max_per_document": 1, "min_similarity": None},
        budget={"max_items": 6, "max_chars": 4000, "max_total_tokens": 1200},
    ),
    Candidate(
        name="A2",
        description="A, but with the per-document cap released.",
        retrieval={"per_query_top_k": 6, "final_top_k": 8,
                   "max_per_document": None, "min_similarity": 0.75},
        budget={"max_items": 4, "max_chars": 3000, "max_total_tokens": 900},
    ),
    Candidate(
        name="unbounded",
        description="Reference only: no budget at all. Never a recommendation.",
        retrieval={"per_query_top_k": 6, "final_top_k": 8,
                   "max_per_document": 1, "min_similarity": 0.75},
        budget={"max_items": 99, "max_chars": 10 ** 6, "max_total_tokens": None},
        reference_only=True,
    ),
)


def by_name(name: str) -> Candidate:
    """Look one up, raising rather than returning ``None`` on a typo."""
    for candidate in CANDIDATES:
        if candidate.name == name:
            return candidate
    raise KeyError(f"no candidate named {name!r}; have {[c.name for c in CANDIDATES]}")


# ===========================================================================
# Per-case accounting
# ===========================================================================
@dataclass(frozen=True, slots=True)
class CaseAccount:
    """What one configuration did to one case.

    Everything here is a measurement or a set operation on measurements.  No
    metric is estimated and none is filled in when it could not be taken; a case
    whose capture is unavailable simply has no account.
    """

    case_id: str
    #: Documents the ranking returned, in order, deduplicated.
    retrieved_documents: tuple[str, ...]
    #: Documents that survived the budget and reached the prompt, in order.
    supplied_documents: tuple[str, ...]
    relevant: frozenset[str]
    irrelevant: frozenset[str]

    #: Size of the rendered knowledge block.
    knowledge_chars: int
    knowledge_tokens: int
    #: Size of the whole user+system prompt, with the knowledge block in it.
    prompt_chars: int
    prompt_tokens: int
    #: The same prompt with no knowledge at all -- the capture JSON and the
    #: instructions.  The difference between this and ``prompt_tokens`` is what
    #: the knowledge budget actually controls.
    capture_only_tokens: int
    #: Excerpts retrieval found and the budget could not afford.
    excluded_by_budget: int
    #: Tokens the provider meters *alongside* the prompt and no budget counts.
    #:
    #: For a ``JSON_SCHEMA`` provider that is the response schema, which for
    #: this project is around a thousand tokens -- larger than the entire
    #: knowledge allowance. Leaving it out made every request look smaller than
    #: the provider sees it, which is precisely the wrong direction for a
    #: figure being compared against a size that has been observed to fail.
    alongside_tokens: int = 0

    def __post_init__(self) -> None:
        if self.prompt_tokens < self.capture_only_tokens:
            raise ValueError(
                f"{self.case_id}: the prompt with knowledge cannot be smaller "
                "than the prompt without it"
            )
        for name, value in (("knowledge_chars", self.knowledge_chars),
                            ("knowledge_tokens", self.knowledge_tokens),
                            ("prompt_chars", self.prompt_chars),
                            ("excluded_by_budget", self.excluded_by_budget)):
            if value < 0:
                raise ValueError(f"{self.case_id}: {name} cannot be negative")

    # -- what went wrong ----------------------------------------------------
    @property
    def request_tokens(self) -> int:
        """What the provider meters: the prompt plus whatever travels with it."""
        return self.prompt_tokens + self.alongside_tokens

    @property
    def never_retrieved(self) -> tuple[str, ...]:
        """Relevant documents the ranking never found.  A retrieval failure."""
        return tuple(sorted(self.relevant - set(self.retrieved_documents)))

    @property
    def lost_before_prompt(self) -> tuple[str, ...]:
        """Relevant documents that were retrieved and then budgeted away.

        Kept apart from :attr:`never_retrieved` deliberately: they are the same
        loss to the model and completely different faults to fix.  One says the
        retriever missed it, the other says the retriever found it and the
        budget threw it out -- and only the second is fixed by spending tokens.
        """
        found = set(self.retrieved_documents) & self.relevant
        return tuple(sorted(found - set(self.supplied_documents)))

    @property
    def irrelevant_supplied(self) -> tuple[str, ...]:
        """Labelled-irrelevant documents that reached the prompt."""
        return tuple(sorted(self.irrelevant & set(self.supplied_documents)))

    @property
    def knowledge_share(self) -> float:
        """Share of the request the knowledge block accounts for.

        The number that decides whether a budget change can plausibly fix a
        request-size failure at all.
        """
        if self.prompt_tokens <= 0:
            return 0.0
        return (self.prompt_tokens - self.capture_only_tokens) / self.prompt_tokens

    @property
    def at_risk(self) -> bool:
        """Whether this request is at least as large as one already seen to fail."""
        return self.request_tokens >= OBSERVED_FAILING_PROMPT_TOKENS

    def problems(self) -> list[str]:
        """Everything worth naming about this case, in a fixed order."""
        found: list[str] = []
        for document in self.never_retrieved:
            found.append(f"{document} was never retrieved")
        for document in self.lost_before_prompt:
            found.append(f"{document} was retrieved, then dropped by the budget")
        for document in self.irrelevant_supplied:
            found.append(f"{document} is labelled irrelevant and was supplied")
        if self.at_risk:
            found.append(
                f"the request is ~{self.request_tokens} tokens "
                f"({self.prompt_tokens} prompt + {self.alongside_tokens} alongside), "
                "at or above a size already observed to fail"
            )
        return found


# ===========================================================================
# Per-candidate accounting
# ===========================================================================
def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass(frozen=True, slots=True)
class CandidateAccount:
    """One configuration's totals across every case that could be scored.

    The retrieval-level metrics are passed in already aggregated by
    :mod:`evaluation.metrics`; everything about the *supplied* set is computed
    here from the per-case accounts, so the two levels cannot silently be
    computed over different case sets.
    """

    candidate: Candidate
    cases: tuple[CaseAccount, ...]
    #: ``aggregate(...).model_dump()`` at the retrieval cut-off.
    retrieval_metrics: Mapping[str, Any]
    #: ``aggregate(...).model_dump()`` over the supplied sets.
    supplied_metrics: Mapping[str, Any]

    # -- retrieval level ----------------------------------------------------
    @property
    def retrieval_recall(self) -> float | None:
        return self.retrieval_metrics.get("recall")

    @property
    def retrieval_precision(self) -> float | None:
        return self.retrieval_metrics.get("precision")

    @property
    def mrr(self) -> float | None:
        return self.retrieval_metrics.get("mrr")

    # -- supplied level -----------------------------------------------------
    @property
    def supplied_recall(self) -> float | None:
        return self.supplied_metrics.get("recall")

    @property
    def supplied_precision(self) -> float | None:
        return self.supplied_metrics.get("precision")

    @property
    def lost_before_prompt(self) -> int:
        return sum(len(case.lost_before_prompt) for case in self.cases)

    @property
    def never_retrieved(self) -> int:
        return sum(len(case.never_retrieved) for case in self.cases)

    @property
    def irrelevant_supplied(self) -> int:
        return sum(len(case.irrelevant_supplied) for case in self.cases)

    # -- cost ---------------------------------------------------------------
    @property
    def mean_knowledge_tokens(self) -> float:
        return _mean([case.knowledge_tokens for case in self.cases])

    @property
    def max_knowledge_tokens(self) -> int:
        return max((case.knowledge_tokens for case in self.cases), default=0)

    @property
    def mean_prompt_chars(self) -> float:
        return _mean([case.prompt_chars for case in self.cases])

    @property
    def max_prompt_chars(self) -> int:
        return max((case.prompt_chars for case in self.cases), default=0)

    @property
    def max_prompt_tokens(self) -> int:
        """Largest prompt, excluding anything sent alongside it."""
        return max((case.prompt_tokens for case in self.cases), default=0)

    @property
    def max_request_tokens(self) -> int:
        """Largest request as the provider meters it.  The live failures are about this."""
        return max((case.request_tokens for case in self.cases), default=0)

    @property
    def mean_knowledge_share(self) -> float:
        return _mean([case.knowledge_share for case in self.cases])

    @property
    def cases_at_risk(self) -> int:
        return sum(1 for case in self.cases if case.at_risk)

    @property
    def evidence_per_1k_tokens(self) -> float | None:
        """Supplied recall per thousand tokens of the largest request.

        The largest rather than the mean, because the failures are per-request:
        a configuration is only usable if its worst case is sendable, and an
        average hides exactly the case that fails.
        """
        recall = self.supplied_recall
        if recall is None or self.max_request_tokens <= 0:
            return None
        return recall / (self.max_request_tokens / 1000.0)

    def failures(self) -> list[str]:
        """Per-case problems, prefixed with the case id."""
        return [f"{case.case_id}: {problem}"
                for case in self.cases for problem in case.problems()]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.candidate.name,
            "description": self.candidate.description,
            "configuration": self.candidate.summary(),
            "reference_only": self.candidate.reference_only,
            "cases_scored": len(self.cases),
            "retrieval_recall": self.retrieval_recall,
            "retrieval_precision": self.retrieval_precision,
            "mrr": self.mrr,
            "supplied_recall": self.supplied_recall,
            "supplied_precision": self.supplied_precision,
            "never_retrieved": self.never_retrieved,
            "lost_before_prompt": self.lost_before_prompt,
            "irrelevant_supplied": self.irrelevant_supplied,
            "mean_knowledge_tokens": round(self.mean_knowledge_tokens, 1),
            "max_knowledge_tokens": self.max_knowledge_tokens,
            "mean_prompt_chars": round(self.mean_prompt_chars, 1),
            "max_prompt_chars": self.max_prompt_chars,
            "max_prompt_tokens": self.max_prompt_tokens,
            "max_request_tokens": self.max_request_tokens,
            "mean_knowledge_share": round(self.mean_knowledge_share, 4),
            "cases_at_risk": self.cases_at_risk,
            "evidence_per_1k_tokens": (None if self.evidence_per_1k_tokens is None
                                       else round(self.evidence_per_1k_tokens, 4)),
            "failures": self.failures(),
            "per_case": [{
                "case": case.case_id,
                "retrieved": list(case.retrieved_documents),
                "supplied": list(case.supplied_documents),
                "never_retrieved": list(case.never_retrieved),
                "lost_before_prompt": list(case.lost_before_prompt),
                "irrelevant_supplied": list(case.irrelevant_supplied),
                "excluded_by_budget": case.excluded_by_budget,
                "knowledge_tokens": case.knowledge_tokens,
                "prompt_tokens": case.prompt_tokens,
                "request_tokens": case.request_tokens,
                "alongside_tokens": case.alongside_tokens,
                "capture_only_tokens": case.capture_only_tokens,
                "at_risk": case.at_risk,
            } for case in self.cases],
        }


# ===========================================================================
# The rule
# ===========================================================================
@dataclass(frozen=True, slots=True)
class Verdict:
    """Whether a candidate may be recommended, and why."""

    name: str
    admissible: bool
    reasons: tuple[str, ...]
    score: float | None

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "admissible": self.admissible,
                "reasons": list(self.reasons),
                "score": None if self.score is None else round(self.score, 4)}


def assess(account: CandidateAccount, baseline: CandidateAccount) -> Verdict:
    """Decide whether one candidate may be recommended over the baseline.

    Three conditions, all of which must hold.  They are stated before any
    numbers are seen, and they are conditions rather than a weighted score
    because a weighted score lets a large win on one axis pay for a
    disqualifying loss on another:

    1. **Recall is preserved.** Supplied recall is at least the baseline's.
       Not "close to": a configuration that supplies less relevant knowledge
       than what already ships is not an improvement whatever it saves.
    2. **Irrelevant knowledge does not grow.** At most as many labelled
       irrelevant documents reach the prompt as under the baseline. Supplying
       a wrong note is worse than supplying nothing, because the model cannot
       tell it is wrong.
    3. **The request does not grow into known failure.** The largest prompt is
       either below :data:`OBSERVED_FAILING_PROMPT_TOKENS`, or no larger than
       the baseline's largest. A configuration already at risk may not get
       bigger to buy recall.

    A ``reference_only`` candidate is never admissible, whatever it scores.

    Among admissible candidates the ranking key is
    :attr:`~CandidateAccount.evidence_per_1k_tokens` -- supplied recall against
    the size of the largest request it produces.
    """
    reasons: list[str] = []
    admissible = True

    if account.candidate.reference_only:
        return Verdict(account.candidate.name, False,
                       ("measured for reference only; never a recommendation",),
                       account.evidence_per_1k_tokens)

    recall, base_recall = account.supplied_recall, baseline.supplied_recall
    if recall is None or base_recall is None:
        return Verdict(account.candidate.name, False,
                       ("supplied recall was undefined; nothing to compare",), None)

    if recall + 1e-9 < base_recall:
        admissible = False
        reasons.append(f"supplied recall {recall:.3f} is below the baseline's "
                       f"{base_recall:.3f}")
    else:
        reasons.append(f"supplied recall {recall:.3f} >= baseline {base_recall:.3f}")

    if account.irrelevant_supplied > baseline.irrelevant_supplied:
        admissible = False
        reasons.append(f"supplies {account.irrelevant_supplied} irrelevant "
                       f"document(s) against the baseline's "
                       f"{baseline.irrelevant_supplied}")
    else:
        reasons.append(f"irrelevant supplied {account.irrelevant_supplied} <= "
                       f"baseline {baseline.irrelevant_supplied}")

    # Rule 3, in two halves, because the live data made the single-clause
    # version dishonest.  Measured against the real index, *every* candidate --
    # the shipped baseline included -- produces a largest request above the
    # size already observed to fail, once the response schema the provider is
    # also sent is counted.  A rule that rejects on absolute size would reject
    # everything including what is running today, and a rule that ignores size
    # would wave through a change that makes a failing request larger.  So:
    baseline_at_risk = baseline.max_request_tokens >= OBSERVED_FAILING_PROMPT_TOKENS
    candidate_at_risk = account.max_request_tokens >= OBSERVED_FAILING_PROMPT_TOKENS

    if candidate_at_risk and not baseline_at_risk:
        # The change is what pushes the request into known failure.  That is
        # the change's fault and it is disqualifying.
        admissible = False
        reasons.append(
            f"largest request ~{account.max_request_tokens} tokens crosses the "
            f"~{OBSERVED_FAILING_PROMPT_TOKENS} size observed to fail, which the "
            f"baseline (~{baseline.max_request_tokens}) does not")
    elif candidate_at_risk and baseline_at_risk:
        # Both are already in the failing regime, so size cannot separate them
        # and is not used to.  It is still said out loud, because the honest
        # conclusion is that no configuration here is safe to send until the
        # request shrinks -- and the knowledge budget is not the term that
        # would shrink it.
        reasons.append(
            f"largest request ~{account.max_request_tokens} tokens; the baseline "
            f"(~{baseline.max_request_tokens}) is already at or above the "
            f"~{OBSERVED_FAILING_PROMPT_TOKENS} size observed to fail, so request "
            "size does not separate these candidates -- see --max-flows")
    else:
        reasons.append(f"largest request ~{account.max_request_tokens} tokens, below "
                       f"the ~{OBSERVED_FAILING_PROMPT_TOKENS} observed to fail "
                       f"(baseline ~{baseline.max_request_tokens})")

    return Verdict(account.candidate.name, admissible, tuple(reasons),
                   account.evidence_per_1k_tokens)


def rank(accounts: Sequence[CandidateAccount],
         baseline_name: str = "baseline") -> dict[str, Any]:
    """Assess every candidate and name the best admissible one, if any.

    Returns the verdicts in the order the candidates were supplied, plus a
    ``recommended`` name or ``None``.  ``None`` is a real answer: "keep what
    ships" is the correct outcome when nothing clears the conditions, and it is
    the outcome step 9 should have reached before changing a default.
    """
    by_key = {account.candidate.name: account for account in accounts}
    if baseline_name not in by_key:
        raise KeyError(f"no account named {baseline_name!r} to compare against")
    baseline = by_key[baseline_name]

    verdicts = [assess(account, baseline) for account in accounts
                if account.candidate.name != baseline_name]
    eligible = [verdict for verdict in verdicts
                if verdict.admissible and verdict.score is not None]

    best: str | None = None
    if eligible:
        # Ties on evidence-per-token are common and were previously broken by
        # name, which quietly made "D" beat an equally-scoring "B" for no
        # reason anyone could defend.  The order now says what it prefers:
        # more evidence per token, then fewer unjudged documents crowding the
        # prompt, then the smaller request, then the name so the result is
        # reproducible.
        by_name_lookup = {account.candidate.name: account for account in accounts}

        def key(verdict: Verdict) -> tuple[float, float, int, str]:
            entry = by_name_lookup[verdict.name]
            precision = entry.supplied_precision
            return (verdict.score or 0.0,
                    precision if precision is not None else -1.0,
                    -entry.max_request_tokens,
                    verdict.name)

        best = max(eligible, key=key).name
        # A candidate that merely ties the baseline on every axis is not an
        # improvement; require it to beat the baseline's evidence-per-token.
        baseline_score = baseline.evidence_per_1k_tokens
        chosen = by_key[best]
        if (baseline_score is not None
                and chosen.evidence_per_1k_tokens is not None
                and chosen.evidence_per_1k_tokens <= baseline_score):
            best = None

    return {
        "baseline": baseline_name,
        "baseline_score": (None if baseline.evidence_per_1k_tokens is None
                           else round(baseline.evidence_per_1k_tokens, 4)),
        "baseline_max_request_tokens": baseline.max_request_tokens,
        # False when the baseline is already at or above a size observed to
        # fail: the report must then say that request size is not a
        # discriminator here rather than implying the winner is safe to send.
        "size_axis_informative":
            baseline.max_request_tokens < OBSERVED_FAILING_PROMPT_TOKENS,
        "verdicts": [verdict.as_dict() for verdict in verdicts],
        "recommended": best,
        "note": ("Recommended for adoption only after the numbers above are read; "
                 "this function selects, it does not change a default."),
    }
