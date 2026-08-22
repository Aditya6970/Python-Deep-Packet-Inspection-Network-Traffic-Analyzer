"""Deep Packet Inspection engine — Python port of the C++ Packet Analyzer.

Package layout note
-------------------
The C++ project split declarations (``include/*.h``) from definitions
(``src/*.cpp``).  Python has no such split, so each header/source pair collapses
into a single module here under its original file name.

Two of those names — ``platform`` and ``types`` — collide with standard library
modules.  Python 3's absolute imports make this harmless for *imports*
(``from dpi import types`` and ``import types`` resolve to different modules),
but running such a module as a **script** puts ``dpi/`` at the front of
``sys.path`` and does shadow the stdlib.  Run their self-tests with ``-m``::

    python -m dpi.types        # correct
    python dpi/types.py        # breaks: shadows stdlib `types`


Console encoding
----------------
Importing this package calls :func:`dpi.console.enable_utf8_console` once, so
the Unicode box-drawing characters used in the reports survive being piped or
redirected on Windows (where the default pipe encoding is ``cp1252``, which
cannot represent them).  Streams that are already UTF-8 are left untouched, so
this is a no-op on Linux and macOS.  Set ``DPI_NO_CONSOLE_UTF8=1`` to skip it.
"""

from __future__ import annotations

from .console import enable_utf8_console

# Run once, at package import, so every entry point and every
# `python -m dpi.<module>` self-test is covered by this single call.
enable_utf8_console()

__version__ = "1.0"
