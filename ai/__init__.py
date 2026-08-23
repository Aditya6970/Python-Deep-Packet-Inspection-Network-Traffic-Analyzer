"""Optional LLM analysis layer for the DPI engine.

This package is a **consumer** of the DPI engine, never a participant in it.
The dependency runs one way only::

    dpi/  ──────────►  ai/          (dpi never imports ai)

Nothing here runs on a packet-processing thread, and no DPI code path calls
into this package.  Deleting ``ai/`` entirely leaves the engine and all of its
self-tests working unchanged.

Importing this package has **no side effects**: it does not read the
environment, construct a client, or open a network connection.  Everything is
resolved explicitly when :mod:`ai.analyzer` is invoked.
"""

from __future__ import annotations

__version__ = "0.1.0"
