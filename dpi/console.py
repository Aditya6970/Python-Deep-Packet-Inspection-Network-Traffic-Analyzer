"""Console output encoding.

The reports this project prints are framed in Unicode box-drawing characters
(``╔ ═ ╗ ║ ╚ ╝``).  Whether those survive depends entirely on the encoding
Python picked for ``sys.stdout``, and on Windows that is often not UTF-8.

Why it breaks
-------------
Python chooses ``sys.stdout``'s encoding differently depending on where the
output is going:

* **A real Windows console.**  Since Python 3.6 (PEP 528) console I/O goes
  through ``_WindowsConsoleIO``, which is Unicode-native.  Box characters work.
* **A pipe or a file** — ``python main_dpi.py > out.txt``, ``| more``, a CI
  job, or a ``subprocess`` with ``capture_output=True``.  Here Python falls
  back to the *locale* encoding, which on a typical Windows install is
  ``cp1252``.  ``cp1252`` has no code point for ``╔``, so the very first
  ``print`` of a report raises::

      UnicodeEncodeError: 'charmap' codec can't encode characters
      in position 0-63: character maps to <undefined>

That is why the failure looks intermittent: running a tool directly in
PowerShell works, but running it through ``run_selftests.py`` — which captures
output via a pipe — crashes.  Linux and macOS are unaffected only because
their locale is normally UTF-8 already.

The fix
-------
:func:`enable_utf8_console` re-points ``sys.stdout`` and ``sys.stderr`` at
UTF-8 using :meth:`io.TextIOWrapper.reconfigure` (Python 3.7+).  It is called
once from :mod:`dpi`'s package ``__init__``, so *every* entry point and every
``python -m dpi.<module>`` self-test is covered by a single call, with no
per-file boilerplate.

Deliberately conservative:

* Streams already on UTF-8 are left untouched, so Linux and macOS behaviour is
  byte-for-byte unchanged.
* Streams that cannot be reconfigured — a :class:`io.StringIO` installed by
  :func:`contextlib.redirect_stdout`, a closed stream, ``pythonw.exe`` where
  ``sys.stdout`` is ``None`` — are skipped rather than raising.
* ``errors="replace"`` is set as a last-resort guard so a stray unencodable
  character degrades to ``?`` instead of taking the process down mid-report.
  It should never trigger: UTF-8 can encode every code point.
* No characters are replaced, no ASCII fallback is used, and no report text
  changes.  The output is identical — it is only the transport that is fixed.

Opting out
----------
Set ``DPI_NO_CONSOLE_UTF8=1`` in the environment to skip the reconfiguration
entirely, for a caller that wants full control of its own streams.
"""

from __future__ import annotations

import os
import sys
from typing import Final

__all__ = ["OPT_OUT_ENV_VAR", "enable_utf8_console", "console_encodings"]

#: Set this environment variable to any non-empty value to disable the fix.
OPT_OUT_ENV_VAR: Final[str] = "DPI_NO_CONSOLE_UTF8"

#: Encoding names that already handle the full Unicode range, normalised.
_UTF8_ALIASES: Final[frozenset[str]] = frozenset({"utf8", "utf8mb4", "cp65001"})

#: Set once so repeated imports do not re-wrap the streams.
_applied = False


def _normalise(encoding: str | None) -> str:
    """Fold an encoding name for comparison (``UTF-8`` -> ``utf8``)."""
    if not encoding:
        return ""
    return encoding.strip().lower().replace("-", "").replace("_", "")


def enable_utf8_console(force: bool = False) -> list[str]:
    """Ensure ``sys.stdout``/``sys.stderr`` can carry non-ASCII output.

    Returns the names of the streams that were actually reconfigured, so the
    caller can report what happened.  An empty list means nothing needed
    changing — the normal case on Linux and macOS, and on a real Windows
    console.

    Safe to call more than once; only the first call does any work unless
    ``force`` is set.
    """
    global _applied

    if _applied and not force:
        return []
    _applied = True

    if os.environ.get(OPT_OUT_ENV_VAR):
        return []

    changed: list[str] = []

    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)

        # pythonw.exe and some embedded hosts give None here.
        if stream is None:
            continue

        # Already Unicode-capable: leave it exactly as it is.
        if _normalise(getattr(stream, "encoding", None)) in _UTF8_ALIASES:
            continue

        # StringIO (contextlib.redirect_stdout), custom writers, and closed
        # streams have no reconfigure(); those need no fixing anyway.
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue

        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            # Detached, closed, or otherwise not reconfigurable.  Printing a
            # warning here could itself fail, so stay silent and carry on.
            continue

        changed.append(name)

    return changed


def console_encodings() -> dict[str, str]:
    """Report the current stdout/stderr encodings, for diagnostics."""
    return {
        name: getattr(getattr(sys, name, None), "encoding", None) or "<none>"
        for name in ("stdout", "stderr")
    }


if __name__ == "__main__":  # pragma: no cover - manual check
    print(f"Python {sys.version.split()[0]} on {sys.platform}")
    print(f"before: {console_encodings()}")
    print(f"reconfigured: {enable_utf8_console(force=True) or 'nothing (already UTF-8)'}")
    print(f"after:  {console_encodings()}")
    print("box-drawing test: ╔══════════╗")
    print("                  ║  it works ║")
    print("                  ╚══════════╝")
