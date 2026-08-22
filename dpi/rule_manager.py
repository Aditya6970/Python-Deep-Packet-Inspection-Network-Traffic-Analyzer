"""Blocking / filtering rule storage and evaluation.

Python port of ``include/rule_manager.h`` + ``src/rule_manager.cpp``
(C++ ``namespace DPI``).

Rules come in four independent flavours, each with its own lock:

1. **IP-based**   — block specific source IPs
2. **App-based**  — block applications detected via SNI
3. **Domain**     — exact names plus ``*.example.com`` wildcard patterns
4. **Port**       — block specific destination ports

One :class:`RuleManager` is shared by every FP thread, which reads from it on
the packet path while the control path may be mutating it.

C++ concepts replaced
---------------------
``std::shared_mutex`` + ``shared_lock``/``unique_lock``
    Becomes a plain :class:`threading.Lock` per rule group.  A reader-writer
    lock buys read *concurrency*, not different semantics, and every critical
    section here is a single set lookup that the GIL already serialises — so
    the lock type is an implementation detail, and mutual exclusion against
    writers (the part that matters for correctness) is preserved exactly.  On
    a free-threaded build a true RWLock could be dropped in unchanged.

``std::unordered_set`` / ``std::vector``
    Become insertion-ordered ``dict``-backed sets and a ``list``.  See
    "Iteration order" below.

Overloaded ``blockIP(uint32_t)`` / ``blockIP(const std::string&)``
    Python has no overloading, so :meth:`RuleManager.block_ip` accepts either
    and dispatches on type — the same two entry points behind one name.

``std::optional<BlockReason>``
    Becomes ``BlockReason | None``.

Iteration order
---------------
``std::unordered_set`` iteration order is unspecified in C++, so the order of
lines written by ``saveRules`` was never defined and differed between runs,
compilers and load factors.  This port stores rules in insertion order, which
is *within* what the original guaranteed (nothing) while making saved files
stable and diffable, and save/load round-trips reproducible.  Rule membership —
the only thing any decision depends on — is identical either way.
"""

from __future__ import annotations

import fnmatch
import sys
import threading
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Final, Iterable

from .types import AppType, app_type_to_string

__all__ = ["BlockReason", "RuleStats", "RuleManager"]

_UINT32_MASK: Final[int] = 0xFFFFFFFF
_UINT16_MASK: Final[int] = 0xFFFF
_INT32_MIN: Final[int] = -0x80000000
_INT32_MAX: Final[int] = 0x7FFFFFFF


def _to_int32(value: int) -> int:
    """Truncate to a 32-bit signed integer, as a C++ ``int`` does on overflow."""
    value &= _UINT32_MASK
    return value - 0x100000000 if value >= 0x80000000 else value


def _stoi(text: str) -> int:
    """Reproduce ``std::stoi`` closely enough for rule files.

    ``std::stoi`` skips leading whitespace, accepts an optional sign, consumes
    the leading run of digits and **ignores trailing garbage** — so ``"80abc"``
    yields 80, where Python's ``int()`` would raise.  It raises
    ``std::invalid_argument`` when no digits are found and ``std::out_of_range``
    beyond ``int`` range; both surface here as :class:`ValueError`, matching the
    original's uncaught-exception behaviour (see :meth:`RuleManager.load_rules`).
    """
    i = 0
    n = len(text)
    while i < n and text[i].isspace():
        i += 1

    start = i
    if i < n and text[i] in "+-":
        i += 1

    digits_start = i
    while i < n and text[i].isdigit():
        i += 1

    if i == digits_start:
        raise ValueError(f"stoi: no conversion for {text!r}")

    value = int(text[start:i])
    if value < _INT32_MIN or value > _INT32_MAX:
        raise ValueError(f"stoi: {text!r} out of int range")
    return value


# ============================================================================
# Block reason / statistics records
# ============================================================================
@dataclass(frozen=True, slots=True)
class BlockReason:
    """Why a packet was blocked.  Mirrors ``struct RuleManager::BlockReason``."""

    class Type(IntEnum):
        """Which rule category matched.  Mirrors the nested ``enum Type``."""

        IP = 0
        APP = 1
        DOMAIN = 2
        PORT = 3

    type: "BlockReason.Type"
    detail: str

    def __str__(self) -> str:
        return f"{self.type.name}: {self.detail}"


@dataclass(frozen=True, slots=True)
class RuleStats:
    """Rule counts.  Mirrors ``struct RuleManager::RuleStats``."""

    blocked_ips: int = 0
    blocked_apps: int = 0
    blocked_domains: int = 0
    blocked_ports: int = 0


# ============================================================================
# Rule Manager
# ============================================================================
class RuleManager:
    """Thread-safe store of blocking rules.  Mirrors ``class RuleManager``."""

    __slots__ = (
        "_ip_lock",
        "_blocked_ips",
        "_app_lock",
        "_blocked_apps",
        "_domain_lock",
        "_blocked_domains",
        "_domain_patterns",
        "_port_lock",
        "_blocked_ports",
    )

    def __init__(self) -> None:
        # dict-as-ordered-set: keys are the members, values are ignored.
        self._ip_lock = threading.Lock()
        self._blocked_ips: dict[int, None] = {}

        self._app_lock = threading.Lock()
        self._blocked_apps: dict[AppType, None] = {}

        self._domain_lock = threading.Lock()
        self._blocked_domains: dict[str, None] = {}
        self._domain_patterns: list[str] = []  # For wildcard matching

        self._port_lock = threading.Lock()
        self._blocked_ports: dict[int, None] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def parse_ip(ip: str) -> int:
        """Convert a dotted-quad string to the project's wire-order integer.

        Mirrors ``RuleManager::parseIP`` **including all of its sloppiness**,
        because rule files and CLI arguments are parsed with it and the results
        must match:

        * Any character that is neither a digit nor ``'.'`` is silently
          ignored, so ``"-1.2.3.4"`` and ``"1.2.3.x4"`` parse as ``1.2.3.4``.
        * Octets are never range-checked, so ``"999.0.0.0"`` yields 999 and
          bleeds into the next octet's bits; ``"256.1.1.1"`` collides with
          ``"0.1.1.1"``'s second octet.
        * ``octet`` is a C++ ``int`` and overflows silently on very long runs
          of digits — reproduced here via :func:`_to_int32`.
        * With more than four octets, ``shift`` reaches 32 and
          ``octet << shift`` is undefined behaviour in C++.  On x86 the shift
          count is taken modulo 32, so the fifth octet wraps back onto the
          first; this port reproduces that observed behaviour (``shift & 31``)
          and it is verified against the compiled binary.

        The result places the first octet in the **low** byte, matching
        :func:`ip_to_string` and :class:`~dpi.types.FiveTuple`.
        """
        result = 0
        octet = 0
        shift = 0

        for c in ip:
            if c == ".":
                result |= _to_int32(octet << (shift & 31)) & _UINT32_MASK
                shift += 8
                octet = 0
            elif "0" <= c <= "9":
                octet = _to_int32(octet * 10 + (ord(c) - 48))

        result |= _to_int32(octet << (shift & 31)) & _UINT32_MASK

        return result & _UINT32_MASK

    @staticmethod
    def ip_to_string(ip: int) -> str:
        """Format a wire-order integer as a dotted quad.

        Mirrors ``RuleManager::ipToString`` — identical to
        :meth:`dpi.types.FiveTuple.format_ip`, shifting by 0/8/16/24.
        """
        return (
            f"{(ip >> 0) & 0xFF}."
            f"{(ip >> 8) & 0xFF}."
            f"{(ip >> 16) & 0xFF}."
            f"{(ip >> 24) & 0xFF}"
        )

    # ------------------------------------------------------------------
    # IP Blocking
    # ------------------------------------------------------------------
    def block_ip(self, ip: int | str) -> None:
        """Block a source IP, given as an int or a dotted-quad string.

        Mirrors both ``blockIP`` overloads.
        """
        if isinstance(ip, str):
            ip = self.parse_ip(ip)
        ip &= _UINT32_MASK

        with self._ip_lock:
            self._blocked_ips[ip] = None
        print(f"[RuleManager] Blocked IP: {self.ip_to_string(ip)}")

    def unblock_ip(self, ip: int | str) -> None:
        """Unblock a source IP.  Mirrors both ``unblockIP`` overloads.

        Note the message prints whether or not the IP was actually present —
        ``std::unordered_set::erase`` on a missing key is a no-op, and the
        original logs unconditionally.
        """
        if isinstance(ip, str):
            ip = self.parse_ip(ip)
        ip &= _UINT32_MASK

        with self._ip_lock:
            self._blocked_ips.pop(ip, None)
        print(f"[RuleManager] Unblocked IP: {self.ip_to_string(ip)}")

    def is_ip_blocked(self, ip: int) -> bool:
        """Return whether an IP is blocked.  Mirrors ``isIPBlocked``."""
        with self._ip_lock:
            return (ip & _UINT32_MASK) in self._blocked_ips

    def get_blocked_ips(self) -> list[str]:
        """Return blocked IPs as dotted quads.  Mirrors ``getBlockedIPs``."""
        with self._ip_lock:
            return [self.ip_to_string(ip) for ip in self._blocked_ips]

    # ------------------------------------------------------------------
    # Application Blocking
    # ------------------------------------------------------------------
    def block_app(self, app: AppType) -> None:
        """Block an application type.  Mirrors ``blockApp``."""
        with self._app_lock:
            self._blocked_apps[app] = None
        print(f"[RuleManager] Blocked app: {app_type_to_string(app)}")

    def unblock_app(self, app: AppType) -> None:
        """Unblock an application type.  Mirrors ``unblockApp``."""
        with self._app_lock:
            self._blocked_apps.pop(app, None)
        print(f"[RuleManager] Unblocked app: {app_type_to_string(app)}")

    def is_app_blocked(self, app: AppType) -> bool:
        """Return whether an application is blocked.  Mirrors ``isAppBlocked``."""
        with self._app_lock:
            return app in self._blocked_apps

    def get_blocked_apps(self) -> list[AppType]:
        """Return the blocked application types.  Mirrors ``getBlockedApps``."""
        with self._app_lock:
            return list(self._blocked_apps)

    # ------------------------------------------------------------------
    # Domain Blocking
    # ------------------------------------------------------------------
    def block_domain(self, domain: str) -> None:
        """Block a domain, or a ``*``-containing pattern.  Mirrors ``blockDomain``.

        A pattern is anything containing ``*`` or ``?``; everything else is an
        exact name, stored case-folded.  Unlike the C++, every glob shape
        actually matches — see :meth:`domain_matches_pattern`.
        """
        with self._domain_lock:
            if "*" in domain or "?" in domain:
                self._domain_patterns.append(domain)
            else:
                # Stored folded, so lookups match regardless of case.
                self._blocked_domains[domain.lower()] = None
        print(f"[RuleManager] Blocked domain: {domain}")

    def unblock_domain(self, domain: str) -> None:
        """Unblock a domain or pattern.  Mirrors ``unblockDomain``.

        Removes only the **first** matching pattern, as ``std::find`` +
        ``erase`` does, so a duplicate pattern needs unblocking twice.
        """
        with self._domain_lock:
            if "*" in domain or "?" in domain:
                try:
                    self._domain_patterns.remove(domain)
                except ValueError:
                    pass  # C++: std::find == end() -> no erase
            else:
                self._blocked_domains.pop(domain.lower(), None)
        print(f"[RuleManager] Unblocked domain: {domain}")

    @staticmethod
    def domain_matches_pattern(domain: str, pattern: str) -> bool:
        """Match a domain against a wildcard pattern.

        FIXED (was UPSTREAM BUG): ``domainMatchesPattern`` only understood a
        leading ``"*."``.  Any other pattern — ``"face*book.com"``,
        ``"*.cdn.*"``, ``"ads*"`` — was still accepted by ``blockDomain``,
        counted in ``getStats`` and written to the rules file, but could never
        match anything.  A rule you asked for silently did nothing.

        The ``*.example.com`` case keeps its exact original semantics,
        including matching the bare ``example.com``.  Anything else now falls
        through to :func:`fnmatch.fnmatchcase` glob matching, so ``*`` and
        ``?`` work wherever they appear.
        """
        # Handle *.example.com pattern (unchanged semantics)
        if len(pattern) >= 2 and pattern[0] == "*" and pattern[1] == ".":
            suffix = pattern[1:]  # .example.com

            # Check if domain ends with the pattern
            if len(domain) >= len(suffix) and domain.endswith(suffix):
                return True

            # Also match the bare domain (example.com matches *.example.com)
            if domain == pattern[2:]:
                return True

            return False

        # Any other glob shape now actually works.
        return fnmatch.fnmatchcase(domain, pattern)

    def is_domain_blocked(self, domain: str) -> bool:
        """Return whether a domain matches any block rule.

        FIXED (was UPSTREAM BUG — asymmetric case handling).  The C++ checked
        the exact-match set with the domain verbatim (case-SENSITIVE) while
        lowercasing both sides for pattern matching (case-INSENSITIVE), so
        blocking ``"Example.com"`` failed to stop ``"example.com"`` but
        ``"*.Example.com"`` succeeded.  DNS names are case-insensitive, so both
        paths now fold case — a blocked domain stays blocked however the client
        capitalises it.
        """
        lower_domain = domain.lower()

        with self._domain_lock:
            # Check exact match (case-insensitive, as DNS requires)
            if lower_domain in self._blocked_domains:
                return True

            # Check patterns
            for pattern in self._domain_patterns:
                if self.domain_matches_pattern(lower_domain, pattern.lower()):
                    return True

            return False

    def get_blocked_domains(self) -> list[str]:
        """Return exact domains followed by patterns.  Mirrors ``getBlockedDomains``."""
        with self._domain_lock:
            return list(self._blocked_domains) + list(self._domain_patterns)

    # ------------------------------------------------------------------
    # Port Blocking
    # ------------------------------------------------------------------
    def block_port(self, port: int) -> None:
        """Block a destination port.  Mirrors ``blockPort``."""
        port &= _UINT16_MASK
        with self._port_lock:
            self._blocked_ports[port] = None
        print(f"[RuleManager] Blocked port: {port}")

    def unblock_port(self, port: int) -> None:
        """Unblock a destination port.  Mirrors ``unblockPort``.

        Note: unlike every other unblock method, the original prints **no**
        message here.  Preserved.
        """
        port &= _UINT16_MASK
        with self._port_lock:
            self._blocked_ports.pop(port, None)

    def is_port_blocked(self, port: int) -> bool:
        """Return whether a port is blocked.  Mirrors ``isPortBlocked``."""
        with self._port_lock:
            return (port & _UINT16_MASK) in self._blocked_ports

    def get_blocked_ports(self) -> list[int]:
        """Return the blocked ports.  No C++ counterpart; used by save_rules."""
        with self._port_lock:
            return list(self._blocked_ports)

    # ------------------------------------------------------------------
    # Combined Check
    # ------------------------------------------------------------------
    def should_block(
        self,
        src_ip: int,
        dst_port: int,
        app: AppType,
        domain: str,
    ) -> BlockReason | None:
        """Evaluate every rule category; return the first match or ``None``.

        Mirrors ``shouldBlock``.  Evaluation order is fixed — **IP, PORT, APP,
        DOMAIN** — and determines which reason is reported when several rules
        would match.  (The C++ comment says "Check IP first (most specific)";
        the ordering is what it is, and is preserved.)

        Each check takes its own lock independently, so this is *not* an atomic
        snapshot of the rule set — a concurrent edit can land between two
        checks.  That is the original's behaviour too.
        """
        # Check IP first (most specific)
        if self.is_ip_blocked(src_ip):
            return BlockReason(BlockReason.Type.IP, self.ip_to_string(src_ip))

        # Check port
        if self.is_port_blocked(dst_port):
            return BlockReason(BlockReason.Type.PORT, str(dst_port))

        # Check app
        if self.is_app_blocked(app):
            return BlockReason(BlockReason.Type.APP, app_type_to_string(app))

        # Check domain
        if domain and self.is_domain_blocked(domain):
            return BlockReason(BlockReason.Type.DOMAIN, domain)

        return None

    # ------------------------------------------------------------------
    # Rule Persistence
    # ------------------------------------------------------------------
    def save_rules(self, filename: str | Path) -> bool:
        """Write all rules to a file.  Mirrors ``saveRules``.

        Format (note the first section has no leading blank line, the rest do)::

            [BLOCKED_IPS]
            192.168.1.50

            [BLOCKED_APPS]
            YouTube

            [BLOCKED_DOMAINS]
            *.tiktok.com

            [BLOCKED_PORTS]
            8080
        """
        try:
            handle = Path(filename).open("w", newline="\n")
        except OSError:
            return False

        with handle as file:
            # Save blocked IPs
            file.write("[BLOCKED_IPS]\n")
            for ip in self.get_blocked_ips():
                file.write(f"{ip}\n")

            # Save blocked apps
            file.write("\n[BLOCKED_APPS]\n")
            for app in self.get_blocked_apps():
                file.write(f"{app_type_to_string(app)}\n")

            # Save blocked domains
            file.write("\n[BLOCKED_DOMAINS]\n")
            for domain in self.get_blocked_domains():
                file.write(f"{domain}\n")

            # Save blocked ports
            file.write("\n[BLOCKED_PORTS]\n")
            for port in self.get_blocked_ports():
                file.write(f"{port}\n")

        print(f"[RuleManager] Rules saved to: {filename}")
        return True

    def load_rules(self, filename: str | Path) -> bool:
        """Load rules from a file.  Mirrors ``loadRules``.

        Rules are **added to** whatever is already loaded; this is not a
        replace.  Returns ``False`` only when the file cannot be opened.

        Faithfully preserved sharp edges:

        * A section header is any line starting with ``'['``, stored whole and
          compared literally against ``"[BLOCKED_IPS]"`` and friends.  A
          straggling ``'\\r'`` therefore stops a section from matching and the
          file **silently loads nothing**.  In C++ whether that happens is
          *platform-dependent*: MSVC's text-mode ``ifstream`` translates CRLF
          to LF (so a Windows-written file loads), while on Linux text mode is
          binary and the ``'\\r'`` survives (so the same file loads nothing).
          This port opens with Python's default universal-newline translation,
          which matches the Windows behaviour — the platform this project
          targets, per ``WINDOWS_SETUP.md`` — on every OS.  That is the one
          place the port is deliberately *more* portable than the original,
          and it can only turn a silent no-op into a successful load.
        * There is no comment syntax; a ``#`` line is parsed as data.
        * FIXED (was UPSTREAM BUG): an unparseable port used to terminate the
          process — ``std::stoi`` throws ``std::invalid_argument`` and the
          original never caught it, so a single typo in a rules file killed
          the engine on startup.  The bad line is now reported on stderr and
          skipped, and the rest of the file still loads.
        * FIXED: an unrecognised app name was silently ignored; it is now
          reported on stderr too, so a typo is visible rather than a rule that
          quietly does nothing.
        """
        try:
            handle = Path(filename).open("r")
        except OSError:
            return False

        with handle as file:
            current_section = ""

            for raw_line in file:
                # std::getline strips only the '\n'; a '\r' would survive.
                line = raw_line[:-1] if raw_line.endswith("\n") else raw_line

                # Skip empty lines
                if not line:
                    continue

                # Check for section headers
                if line[0] == "[":
                    current_section = line
                    continue

                # Process based on section
                if current_section == "[BLOCKED_IPS]":
                    self.block_ip(line)
                elif current_section == "[BLOCKED_APPS]":
                    # Convert string back to AppType
                    for i in range(int(AppType.APP_COUNT)):
                        if app_type_to_string(AppType(i)) == line:
                            self.block_app(AppType(i))
                            break
                    else:
                        print(
                            f"[RuleManager] Unknown app in {filename}: {line!r} (skipped)",
                            file=sys.stderr,
                        )
                elif current_section == "[BLOCKED_DOMAINS]":
                    self.block_domain(line)
                elif current_section == "[BLOCKED_PORTS]":
                    try:
                        port = _stoi(line)
                    except ValueError:
                        print(
                            f"[RuleManager] Invalid port in {filename}: {line!r} (skipped)",
                            file=sys.stderr,
                        )
                        continue
                    self.block_port(port & _UINT16_MASK)

        print(f"[RuleManager] Rules loaded from: {filename}")
        return True

    def clear_all(self) -> None:
        """Remove every rule.  Mirrors ``clearAll``."""
        with self._ip_lock:
            self._blocked_ips.clear()
        with self._app_lock:
            self._blocked_apps.clear()
        with self._domain_lock:
            self._blocked_domains.clear()
            self._domain_patterns.clear()
        with self._port_lock:
            self._blocked_ports.clear()
        print("[RuleManager] All rules cleared")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def get_stats(self) -> RuleStats:
        """Return rule counts.  Mirrors ``getStats``.

        ``blocked_domains`` is the sum of exact names and wildcard patterns.
        """
        with self._ip_lock:
            blocked_ips = len(self._blocked_ips)
        with self._app_lock:
            blocked_apps = len(self._blocked_apps)
        with self._domain_lock:
            blocked_domains = len(self._blocked_domains) + len(self._domain_patterns)
        with self._port_lock:
            blocked_ports = len(self._blocked_ports)

        return RuleStats(
            blocked_ips=blocked_ips,
            blocked_apps=blocked_apps,
            blocked_domains=blocked_domains,
            blocked_ports=blocked_ports,
        )

    def __repr__(self) -> str:
        s = self.get_stats()
        return (
            f"RuleManager(ips={s.blocked_ips}, apps={s.blocked_apps}, "
            f"domains={s.blocked_domains}, ports={s.blocked_ports})"
        )

    # ------------------------------------------------------------------
    # C++-style aliases
    # ------------------------------------------------------------------
    parseIP = staticmethod(parse_ip)
    ipToString = staticmethod(ip_to_string)
    blockIP = block_ip
    unblockIP = unblock_ip
    isIPBlocked = is_ip_blocked
    getBlockedIPs = get_blocked_ips
    blockApp = block_app
    unblockApp = unblock_app
    isAppBlocked = is_app_blocked
    getBlockedApps = get_blocked_apps
    blockDomain = block_domain
    unblockDomain = unblock_domain
    domainMatchesPattern = staticmethod(domain_matches_pattern)
    isDomainBlocked = is_domain_blocked
    getBlockedDomains = get_blocked_domains
    blockPort = block_port
    unblockPort = unblock_port
    isPortBlocked = is_port_blocked
    shouldBlock = should_block
    saveRules = save_rules
    loadRules = load_rules
    clearAll = clear_all
    getStats = get_stats


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import io
    import tempfile
    from contextlib import redirect_stdout

    quiet = io.StringIO()

    with redirect_stdout(quiet):
        rm = RuleManager()

        # --- parseIP: wire order, and every documented sloppiness ------
        assert RuleManager.parse_ip("192.168.1.10") == 0x0A01A8C0
        assert RuleManager.ip_to_string(RuleManager.parse_ip("8.8.8.8")) == "8.8.8.8"
        assert RuleManager.parse_ip("-1.2.3.4") == RuleManager.parse_ip("1.2.3.4")
        assert RuleManager.parse_ip("1.2.3.x4") == RuleManager.parse_ip("1.2.3.4")
        assert RuleManager.parse_ip("999.0.0.0") == 999
        assert RuleManager.parse_ip("1.2.3.4.5") == 67305989  # 5th octet wraps onto 1st

        # --- IP rules ---------------------------------------------------
        rm.block_ip("192.168.1.50")
        assert rm.is_ip_blocked(RuleManager.parse_ip("192.168.1.50"))
        assert not rm.is_ip_blocked(RuleManager.parse_ip("192.168.1.51"))
        assert rm.get_blocked_ips() == ["192.168.1.50"]
        rm.unblock_ip("192.168.1.50")
        assert not rm.is_ip_blocked(RuleManager.parse_ip("192.168.1.50"))

        # --- App rules --------------------------------------------------
        rm.block_app(AppType.YOUTUBE)
        assert rm.is_app_blocked(AppType.YOUTUBE)
        assert not rm.is_app_blocked(AppType.GOOGLE)

        # --- Domain rules, exact and wildcard ---------------------------
        rm.block_domain("ads.example.com")
        rm.block_domain("*.tiktok.com")
        assert rm.is_domain_blocked("ads.example.com")
        assert rm.is_domain_blocked("cdn.tiktok.com")
        assert rm.is_domain_blocked("tiktok.com")  # bare domain matches too
        assert rm.is_domain_blocked("CDN.TIKTOK.COM")  # patterns are case-insensitive
        assert not rm.is_domain_blocked("nottiktok.com")
        assert not rm.is_domain_blocked("tiktok.com.evil.net")
        # FIXED: exact matches are now case-insensitive, like DNS
        assert rm.is_domain_blocked("ADS.EXAMPLE.COM")
        # FIXED: every glob shape now actually matches
        rm.block_domain("face*book.com")
        assert rm.is_domain_blocked("facebook.com")
        assert rm.is_domain_blocked("faceXbook.com")
        assert not rm.is_domain_blocked("notabook.com")

        # --- Port rules -------------------------------------------------
        rm.block_port(8080)
        assert rm.is_port_blocked(8080) and not rm.is_port_blocked(80)

        # --- should_block precedence: IP > PORT > APP > DOMAIN ----------
        rm.block_ip("10.0.0.9")
        ip9 = RuleManager.parse_ip("10.0.0.9")
        r = rm.should_block(ip9, 8080, AppType.YOUTUBE, "cdn.tiktok.com")
        assert r is not None and r.type is BlockReason.Type.IP
        r = rm.should_block(0, 8080, AppType.YOUTUBE, "cdn.tiktok.com")
        assert r is not None and r.type is BlockReason.Type.PORT
        r = rm.should_block(0, 443, AppType.YOUTUBE, "cdn.tiktok.com")
        assert r is not None and r.type is BlockReason.Type.APP
        r = rm.should_block(0, 443, AppType.GOOGLE, "cdn.tiktok.com")
        assert r is not None and r.type is BlockReason.Type.DOMAIN
        assert rm.should_block(0, 443, AppType.GOOGLE, "example.org") is None
        # Empty domain short-circuits the domain check
        assert rm.should_block(0, 443, AppType.GOOGLE, "") is None

        # --- stats ------------------------------------------------------
        st = rm.get_stats()
        assert (st.blocked_ips, st.blocked_apps, st.blocked_domains, st.blocked_ports) == (
            1,
            1,
            3,
            1,
        ), st

        # --- save/load round-trip --------------------------------------
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rules.txt"
            assert rm.save_rules(path)
            saved = path.read_text()

            rm2 = RuleManager()
            assert rm2.load_rules(path)
            assert rm2.get_stats() == rm.get_stats()
            assert rm2.get_blocked_ips() == rm.get_blocked_ips()
            assert rm2.get_blocked_apps() == rm.get_blocked_apps()
            assert rm2.get_blocked_domains() == rm.get_blocked_domains()
            assert rm2.get_blocked_ports() == rm.get_blocked_ports()

            assert not RuleManager().load_rules(Path(tmp) / "missing.txt")

            # std::stoi tolerance: trailing garbage is ignored
            (Path(tmp) / "p.txt").write_text("[BLOCKED_PORTS]\n80abc\n")
            rm3 = RuleManager()
            rm3.load_rules(Path(tmp) / "p.txt")
            assert rm3.is_port_blocked(80)

            # FIXED: a malformed port no longer kills the process; the rest
            # of the file still loads.
            (Path(tmp) / "bad.txt").write_text(
                "[BLOCKED_PORTS]\noops\n8080\n[BLOCKED_APPS]\nNotAnApp\nYouTube\n"
            )
            rm5 = RuleManager()
            assert rm5.load_rules(Path(tmp) / "bad.txt")
            assert rm5.is_port_blocked(8080), "good lines must still load"
            assert rm5.is_app_blocked(AppType.YOUTUBE)

            # CRLF rules file: loads, matching the MSVC text-mode build.
            # (A Linux C++ build would silently load nothing here.)
            (Path(tmp) / "crlf.txt").write_bytes(b"[BLOCKED_PORTS]\r\n8080\r\n")
            rm4 = RuleManager()
            rm4.load_rules(Path(tmp) / "crlf.txt")
            assert rm4.is_port_blocked(8080)

        rm.clear_all()
        assert rm.get_stats() == RuleStats(0, 0, 0, 0)

    print("Saved rules file:")
    print("---")
    print(saved, end="")
    print("---")
    print("rule_manager.py self-test OK")
