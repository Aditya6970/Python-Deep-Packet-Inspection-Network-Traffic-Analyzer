"""Evaluation harness for the RAG + AI analysis pipeline.

What this package is for
------------------------
Measuring the pipeline, not running it.  Nothing here is imported by
production code: ``dpi/``, ``ai/`` and ``ai/rag/`` do not know this package
exists, and deleting it changes nothing about how the analyzer behaves.  That
separation is the point -- an evaluation harness that the thing under test
depends on cannot be trusted to judge it.

Three modules:

* :mod:`evaluation.metrics` -- the arithmetic.  Recall@K, Precision@K, Hit@K
  and MRR over ranked id lists.  Pure functions with no notion of retrieval,
  embeddings or documents, so they can be checked against worked examples.
* :mod:`evaluation.cases` -- the dataset.  Capture reports paired with what a
  correct system should do with them, written by hand and never derived from
  model output.
* :mod:`evaluation.candidates` -- the configurations under consideration and
  the accounting that compares them, including what each one costs to send.
  Pure arithmetic over measurements someone else took.

The runner lives at the repository root as ``run_rag_evaluation.py``, beside
the other ``run_*`` entry points.

Labels are fixed before results are seen
----------------------------------------
Every expectation in :mod:`evaluation.cases` -- which signals should fire,
which documents are relevant, which are not -- was written from the corpus and
the DPI schema, not from a retrieval run.  A label adjusted to match an
observed result measures nothing.
"""

from __future__ import annotations

__all__ = ["candidates", "cases", "metrics"]

__version__ = "1.1"
