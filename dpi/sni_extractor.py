"""Payload inspectors: TLS SNI, QUIC, HTTP Host and DNS query extraction.

Python port of ``include/sni_extractor.h`` + ``src/sni_extractor.cpp``
(C++ ``namespace DPI``).

This is the deep-inspection layer.  Given a transport payload it tries to
recover the server name the client is asking for, which is what the engine
classifies and filters on.

TLS Client Hello structure (the shape :class:`SNIExtractor` walks)::

    Record Layer:
      Content Type (1)      0x16 = Handshake
      Version (2)           0x0301 = TLS 1.0, 0x0303 = TLS 1.2
      Length (2)
    Handshake Layer:
      Handshake Type (1)    0x01 = Client Hello
      Length (3)
      Client Version (2)
      Random (32)
      Session ID Length (1) + Session ID (variable)
      Cipher Suites Length (2) + Cipher Suites (variable)
      Compression Methods Length (1) + Compression Methods (variable)
      Extensions Length (2) + Extensions (variable)
    SNI Extension (type 0x0000):
      Extension Type (2)    0x0000
      Extension Length (2)
      SNI List Length (2)
      SNI Type (1)          0x00 = hostname
      SNI Length (2)
      SNI Value (variable)  <- the hostname

C++ concepts replaced
---------------------
``const uint8_t* payload, size_t length`` (pointer + length pairs)
    Become a single ``bytes``/``bytearray``/``memoryview`` argument.  Slicing a
    :class:`memoryview` is zero-copy, so passing sub-windows around costs no
    more than the C++ pointer arithmetic did.

``std::optional<std::string>``
    Becomes ``str | None``.

``std::string(reinterpret_cast<const char*>(p), n)``
    Raw bytes with no encoding.  Decoded here as ``latin-1``, which is
    byte-preserving and cannot fail, so any byte sequence round-trips and
    ASCII substring matching behaves exactly as ``std::string::find`` did.

``static`` member functions
    Become ``@staticmethod``s, keeping the ``SNIExtractor.extract(...)``
    call syntax.

Fixed upstream bugs
-------------------
Three defects in the C++ original are corrected here, each marked ``FIXED`` at
its site: a malformed or non-hostname SNI entry no longer abandons the whole
extension walk; ``extract_extensions`` is implemented rather than a stub; and
``HTTPHostExtractor`` anchors ``Host:`` to the start of a header line (so a
client-supplied ``X-Forwarded-Host`` cannot win) and no longer skips a header
in the payload's last six bytes.

:meth:`QUICSNIExtractor.extract` remains the one place this port cannot be
bug-compatible: the C++ reads *before* the start of the buffer — see that
method's docstring.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "CONTENT_TYPE_HANDSHAKE",
    "HANDSHAKE_CLIENT_HELLO",
    "EXTENSION_SNI",
    "SNI_TYPE_HOSTNAME",
    "read_uint16_be",
    "read_uint24_be",
    "SNIExtractor",
    "QUICSNIExtractor",
    "HTTPHostExtractor",
    "DNSExtractor",
]

# Accepted payload types; all support integer indexing and zero-copy slicing.
_Payload = "bytes | bytearray | memoryview"

# ---------------------------------------------------------------------------
# TLS Constants (C++ private static constexpr members of SNIExtractor)
# ---------------------------------------------------------------------------
CONTENT_TYPE_HANDSHAKE: Final[int] = 0x16
HANDSHAKE_CLIENT_HELLO: Final[int] = 0x01
EXTENSION_SNI: Final[int] = 0x0000
SNI_TYPE_HOSTNAME: Final[int] = 0x00

#: Encoding used to turn raw name bytes into ``str``.  latin-1 maps bytes 1:1
#: onto codepoints, so it never raises and never loses information.
_NAME_ENCODING: Final[str] = "latin-1"


# ---------------------------------------------------------------------------
# Helpers to read big-endian values (C++ readUint16BE / readUint24BE)
# ---------------------------------------------------------------------------
def read_uint16_be(data, offset: int = 0) -> int:
    """Read a big-endian ``uint16`` at ``offset``.  Mirrors ``readUint16BE``."""
    return (data[offset] << 8) | data[offset + 1]


def read_uint24_be(data, offset: int = 0) -> int:
    """Read a big-endian ``uint24`` at ``offset``.  Mirrors ``readUint24BE``."""
    return (data[offset] << 16) | (data[offset + 1] << 8) | data[offset + 2]


# ============================================================================
# SNI Extractor - Parses TLS Client Hello to extract Server Name Indication
# ============================================================================
class SNIExtractor:
    """Extracts the SNI hostname from a TLS Client Hello.

    Mirrors ``class SNIExtractor``.  All members are static.
    """

    __slots__ = ()

    # Constants re-exposed as class attributes, as in the C++ original.
    CONTENT_TYPE_HANDSHAKE: Final[int] = CONTENT_TYPE_HANDSHAKE
    HANDSHAKE_CLIENT_HELLO: Final[int] = HANDSHAKE_CLIENT_HELLO
    EXTENSION_SNI: Final[int] = EXTENSION_SNI
    SNI_TYPE_HOSTNAME: Final[int] = SNI_TYPE_HOSTNAME

    @staticmethod
    def is_tls_client_hello(payload, length: int | None = None) -> bool:
        """Check whether ``payload`` looks like a TLS Client Hello.

        Mirrors ``isTLSClientHello``.  Accepts SSL 3.0 (0x0300) through
        TLS 1.3 (0x0304) in the *record* version field.
        """
        if length is None:
            length = len(payload)

        # Minimum TLS record: 5 bytes header + 4 bytes handshake header
        if length < 9:
            return False

        # Byte 0: Content Type (should be 0x16 = Handshake)
        if payload[0] != CONTENT_TYPE_HANDSHAKE:
            return False

        # Bytes 1-2: TLS Version. We accept 0x0300 (SSL 3.0) - 0x0304 (TLS 1.3)
        version = read_uint16_be(payload, 1)
        if version < 0x0300 or version > 0x0304:
            return False

        # Bytes 3-4: Record length.  length >= 9 here, so `length - 5` cannot
        # underflow despite being unsigned arithmetic in C++.
        record_length = read_uint16_be(payload, 3)
        if record_length > length - 5:
            return False

        # Byte 5: Handshake Type (should be 0x01 = Client Hello)
        if payload[5] != HANDSHAKE_CLIENT_HELLO:
            return False

        return True

    @staticmethod
    def extract(payload, length: int | None = None) -> str | None:
        """Extract the SNI hostname, or ``None`` if there isn't one.

        Mirrors ``std::optional<std::string> SNIExtractor::extract``.

        The walk is bounds-checked at every variable-length field, in the same
        places and the same order as the original, so malformed input is
        rejected identically.
        """
        if length is None:
            length = len(payload)

        if not SNIExtractor.is_tls_client_hello(payload, length):
            return None

        # Skip TLS record header (5 bytes)
        offset = 5

        # Skip handshake header: type (1) + length (3).
        # UPSTREAM BUG (harmless): handshake_length is computed and never used.
        # Kept because the read is what advances nothing -- the offset jump
        # below is a fixed 4 either way.
        _handshake_length = read_uint24_be(payload, offset + 1)
        offset += 4

        # Client Hello body: client version (2 bytes)
        offset += 2

        # Random (32 bytes)
        offset += 32

        # Session ID
        if offset >= length:
            return None
        session_id_length = payload[offset]
        offset += 1 + session_id_length

        # Cipher suites
        if offset + 2 > length:
            return None
        cipher_suites_length = read_uint16_be(payload, offset)
        offset += 2 + cipher_suites_length

        # Compression methods
        if offset >= length:
            return None
        compression_methods_length = payload[offset]
        offset += 1 + compression_methods_length

        # Extensions
        if offset + 2 > length:
            return None
        extensions_length = read_uint16_be(payload, offset)
        offset += 2

        extensions_end = offset + extensions_length
        if extensions_end > length:
            extensions_end = length  # Truncated, but try to parse anyway

        # Parse extensions to find SNI
        while offset + 4 <= extensions_end:
            extension_type = read_uint16_be(payload, offset)
            extension_length = read_uint16_be(payload, offset + 2)
            offset += 4

            if offset + extension_length > extensions_end:
                break

            if extension_type == EXTENSION_SNI:
                # SNI extension found. Structure:
                #   SNI List Length (2) | SNI Type (1) | SNI Length (2) | Value
                #
                # FIXED (was UPSTREAM BUG): every rejection path below used
                # `break` rather than `continue`, so one malformed or
                # non-hostname SNI entry abandoned the entire extension list
                # and the packet went unclassified.  A server-name entry of a
                # type other than `host_name` is legal in TLS and is exactly
                # the case that used to kill the whole walk.  Each path now
                # falls through to the next extension instead.
                name = SNIExtractor._read_sni_extension(
                    payload, offset, extension_length
                )
                if name is not None:
                    return name

            offset += extension_length

        return None

    @staticmethod
    def _read_sni_extension(payload, offset: int, extension_length: int) -> str | None:
        """Decode one SNI extension body, or ``None`` if it is unusable.

        Split out of :meth:`extract` so a rejection can skip just this
        extension instead of ending the walk.  All bounds were already
        established by the caller.
        """
        if extension_length < 5:
            return None

        sni_list_length = read_uint16_be(payload, offset)
        if sni_list_length < 3:
            return None

        sni_type = payload[offset + 2]
        sni_length = read_uint16_be(payload, offset + 3)

        if sni_type != SNI_TYPE_HOSTNAME:
            return None
        # extension_length >= 5 was checked above, so this cannot underflow.
        if sni_length > extension_length - 5:
            return None

        return bytes(payload[offset + 5 : offset + 5 + sni_length]).decode(_NAME_ENCODING)

    @staticmethod
    def extract_extensions(payload, length: int | None = None) -> list[tuple[int, str]]:
        """Return all TLS extensions as ``(type, value)`` pairs.

        FIXED (was an UPSTREAM STUB): the C++ body declared an empty vector,
        carried the comment "... (abbreviated for brevity)", and returned it —
        so every caller got nothing.  It now performs the same walk as
        :meth:`extract` and reports each extension as
        ``(type, hex-encoded body)``.
        """
        if length is None:
            length = len(payload)

        extensions: list[tuple[int, str]] = []

        if not SNIExtractor.is_tls_client_hello(payload, length):
            return extensions

        offset = 5 + 4 + 2 + 32  # record + handshake headers, version, random

        if offset >= length:
            return extensions
        offset += 1 + payload[offset]  # session id

        if offset + 2 > length:
            return extensions
        offset += 2 + read_uint16_be(payload, offset)  # cipher suites

        if offset >= length:
            return extensions
        offset += 1 + payload[offset]  # compression methods

        if offset + 2 > length:
            return extensions
        extensions_length = read_uint16_be(payload, offset)
        offset += 2

        extensions_end = min(offset + extensions_length, length)

        while offset + 4 <= extensions_end:
            extension_type = read_uint16_be(payload, offset)
            extension_length = read_uint16_be(payload, offset + 2)
            offset += 4

            if offset + extension_length > extensions_end:
                break

            body = bytes(payload[offset : offset + extension_length])
            extensions.append((extension_type, body.hex()))
            offset += extension_length

        return extensions

    # C++-style aliases
    isTLSClientHello = staticmethod(is_tls_client_hello)
    extractExtensions = staticmethod(extract_extensions)


# ============================================================================
# QUIC SNI Extractor - For QUIC/HTTP3 traffic
# ============================================================================
class QUICSNIExtractor:
    """Best-effort SNI recovery from QUIC Initial packets.

    Mirrors ``class QUICSNIExtractor``.  The C++ comment is candid that this is
    a simplification: real QUIC carries the Client Hello inside CRYPTO frames
    of an AEAD-protected Initial packet, so a proper implementation must derive
    the Initial keys and decrypt first.  This one just scans the raw bytes for
    a plausible Client Hello, which succeeds only on unprotected or test
    traffic.
    """

    __slots__ = ()

    @staticmethod
    def is_quic_initial(payload, length: int | None = None) -> bool:
        """Check for a QUIC long-header packet.  Mirrors ``isQUICInitial``.

        Note how weak this test is: it requires 5 bytes and the long-header
        form bit, and explicitly does not validate the version.  Any UDP
        payload whose first byte has the high bit set passes.
        """
        if length is None:
            length = len(payload)

        if length < 5:
            return False

        # QUIC long header starts with the form bit set
        first_byte = payload[0]
        if (first_byte & 0x80) == 0:
            return False

        # Check for QUIC version (bytes 1-4)
        # Common versions: 0x00000001 (v1), 0xff000000+ (drafts)
        # We'll be lenient here
        return True

    @staticmethod
    def extract(payload, length: int | None = None) -> str | None:
        """Scan a QUIC Initial packet for an embedded TLS Client Hello.

        Mirrors ``QUICSNIExtractor::extract``: walk the buffer looking for a
        ``0x01`` (Client Hello handshake type) byte, then hand
        :meth:`SNIExtractor.extract` a window starting 5 bytes earlier, so the
        candidate byte lands where a TLS record's handshake type belongs.

        DELIBERATE DEVIATION -- the one place this port is not bug-compatible.
        The C++ loop starts at ``i = 0`` and calls
        ``SNIExtractor::extract(payload + i - 5, length - i + 5)``.  For
        ``i < 5`` that forms a pointer *before* the start of the buffer and a
        length reaching past its end, so the callee reads up to 5 bytes of
        memory the packet does not own.  That is undefined behaviour: there is
        no defined result to reproduce, and Python cannot address memory
        outside an object anyway.  This port therefore starts the scan at
        ``i = 5``.

        Observable impact is negligible: for ``i < 5`` the C++ would have to
        find stray heap bytes that happen to spell a valid TLS record header
        (0x16, a version in 0x0300-0x0304, and a consistent record length)
        before it could return anything -- and if it did, the result would be
        junk derived from unrelated memory.  For every ``i >= 5`` the two
        implementations agree exactly.
        """
        if length is None:
            length = len(payload)

        # QUIC Initial packets contain the TLS Client Hello inside CRYPTO
        # frames.  This is complex to parse properly due to QUIC framing, so
        # this is a simplified search for the SNI extension pattern.
        if not QUICSNIExtractor.is_quic_initial(payload, length):
            return None

        view = memoryview(payload)

        # C++: for (size_t i = 0; i + 50 < length; i++)
        # Port: start at 5 -- see the deviation note above.
        i = 5
        while i + 50 < length:
            if view[i] == HANDSHAKE_CLIENT_HELLO:  # Client Hello handshake type
                # Window is [i-5, length): the same bytes the C++ pointer pair
                # (payload + i - 5, length - i + 5) describes.
                result = SNIExtractor.extract(view[i - 5 :], length - i + 5)
                if result is not None:
                    return result
            i += 1

        return None

    # C++-style alias
    isQUICInitial = staticmethod(is_quic_initial)


# ============================================================================
# HTTP Host Header Extractor (for unencrypted HTTP)
# ============================================================================
class HTTPHostExtractor:
    """Extracts the ``Host:`` header from a plaintext HTTP request.

    Mirrors ``class HTTPHostExtractor``.
    """

    __slots__ = ()

    #: Four-byte method prefixes, exactly as the C++ array.  Note the trailing
    #: spaces and the truncated forms -- the comparison is always 4 bytes, so
    #: "DELE", "PATC" and "OPTI" match DELETE/PATCH/OPTIONS by prefix.
    _METHODS: Final[tuple[bytes, ...]] = (
        b"GET ",
        b"POST",
        b"PUT ",
        b"HEAD",
        b"DELE",
        b"PATC",
        b"OPTI",
    )

    @staticmethod
    def is_http_request(payload, length: int | None = None) -> bool:
        """Check for a known HTTP method prefix.  Mirrors ``isHTTPRequest``."""
        if length is None:
            length = len(payload)

        if length < 4:
            return False

        first_four = bytes(payload[:4])
        return any(first_four == method for method in HTTPHostExtractor._METHODS)

    @staticmethod
    def extract(payload, length: int | None = None) -> str | None:
        """Extract the Host header value, minus any ``:port`` suffix.

        FIXED (was UPSTREAM BUG), two defects:

        * The C++ matched a case-insensitive ``host:`` **anywhere** in the
          payload, so ``X-Forwarded-Host:`` — a header any client can set —
          won if it appeared first, and a ``host:`` inside a URL or request
          body would too.  The match is now anchored to the start of a header
          line (start of payload, or immediately after CR/LF), which is what
          RFC 9112 field parsing requires.
        * The loop bound was ``i + 6 < length`` (strict), so a Host header in
          the last 6 bytes of the payload was never examined.  It is now
          ``<=``.

        Still preserved: an empty value does not end the search, so a bare
        ``Host:`` followed by a later valid one still resolves.
        """
        if length is None:
            length = len(payload)

        if not HTTPHostExtractor.is_http_request(payload, length):
            return None

        # NOTE: the C++ declares `host_header`/`host_header_len` here but only
        # ever uses the length; the literal itself is dead. Kept as the bound.
        host_header_len = 6

        i = 0
        while i + host_header_len <= length:
            # Only at the start of a header line (start of payload, or just
            # after a CR/LF), so X-Forwarded-Host and friends cannot win.
            at_line_start = i == 0 or payload[i - 1] in (0x0A, 0x0D)

            # Check for header (case-insensitive "host:")
            if (
                at_line_start
                and payload[i] in (0x48, 0x68)  # 'H' or 'h'
                and payload[i + 1] in (0x6F, 0x4F)  # 'o' or 'O'
                and payload[i + 2] in (0x73, 0x53)  # 's' or 'S'
                and payload[i + 3] in (0x74, 0x54)  # 't' or 'T'
                and payload[i + 4] == 0x3A  # ':'
            ):
                # Skip "Host:" and any whitespace
                start = i + 5
                while start < length and payload[start] in (0x20, 0x09):  # ' ' or '\t'
                    start += 1

                # Find end of line
                end = start
                while end < length and payload[end] not in (0x0D, 0x0A):  # '\r' '\n'
                    end += 1

                if end > start:
                    host = bytes(payload[start:end]).decode(_NAME_ENCODING)

                    # Remove port if present
                    colon_pos = host.find(":")
                    if colon_pos != -1:
                        host = host[:colon_pos]

                    return host
                # Empty value: fall through and keep scanning (as C++ does).
            i += 1

        return None

    # C++-style alias
    isHTTPRequest = staticmethod(is_http_request)


# ============================================================================
# DNS Query Extractor (to correlate domain names)
# ============================================================================
class DNSExtractor:
    """Extracts the queried name from a DNS request.

    Mirrors ``class DNSExtractor``.
    """

    __slots__ = ()

    @staticmethod
    def is_dns_query(payload, length: int | None = None) -> bool:
        """Check for a DNS query (not a response).  Mirrors ``isDNSQuery``."""
        if length is None:
            length = len(payload)

        # Minimum DNS header is 12 bytes
        if length < 12:
            return False

        # Check QR bit (byte 2, bit 7) - should be 0 for query
        flags = payload[2]
        if flags & 0x80:
            return False  # This is a response, not a query

        # Check QDCOUNT (bytes 4-5) - should be > 0
        qdcount = (payload[4] << 8) | payload[5]
        if qdcount == 0:
            return False

        return True

    @staticmethod
    def extract_query(payload, length: int | None = None) -> str | None:
        """Extract the queried domain name, or ``None``.

        Mirrors ``extractQuery``: walk the length-prefixed labels from byte 12
        and join them with dots.

        Note that DNS name **compression pointers are not followed** -- a label
        length byte above 63 (the two high bits set marks a pointer) simply
        ends the walk, returning whatever was collected so far.  That is the
        original behaviour and is preserved.
        """
        if length is None:
            length = len(payload)

        if not DNSExtractor.is_dns_query(payload, length):
            return None

        # DNS query starts at byte 12
        offset = 12
        labels: list[str] = []

        while offset < length:
            label_length = payload[offset]

            if label_length == 0:
                # End of domain name
                break

            if label_length > 63:
                # Compression pointer or invalid
                break

            offset += 1
            if offset + label_length > length:
                break

            labels.append(bytes(payload[offset : offset + label_length]).decode(_NAME_ENCODING))
            offset += label_length

        domain = ".".join(labels)
        return domain if domain else None

    # C++-style aliases
    isDNSQuery = staticmethod(is_dns_query)
    extractQuery = staticmethod(extract_query)


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import struct

    def build_client_hello(hostname: bytes, tls_version: int = 0x0303) -> bytes:
        """Assemble a minimal but well-formed TLS Client Hello carrying an SNI."""
        sni_ext_body = (
            struct.pack(">H", len(hostname) + 3)  # SNI list length
            + b"\x00"  # name type: hostname
            + struct.pack(">H", len(hostname))
            + hostname
        )
        extensions = struct.pack(">HH", 0x0000, len(sni_ext_body)) + sni_ext_body
        body = (
            struct.pack(">H", tls_version)  # client version
            + b"\xAB" * 32  # random
            + b"\x00"  # session id length
            + struct.pack(">H", 2)  # cipher suites length
            + b"\x13\x01"  # one cipher suite
            + b"\x01\x00"  # compression methods
            + struct.pack(">H", len(extensions))
            + extensions
        )
        handshake = b"\x01" + struct.pack(">I", len(body))[1:] + body
        return b"\x16" + struct.pack(">H", tls_version) + struct.pack(">H", len(handshake)) + handshake

    hello = build_client_hello(b"www.example.com")
    assert SNIExtractor.is_tls_client_hello(hello)
    assert SNIExtractor.extract(hello) == "www.example.com"
    assert SNIExtractor.extract(hello[:20]) is None
    assert SNIExtractor.extract(b"") is None
    assert SNIExtractor.extract(b"\x16\x03\x03\x00\x04\x02\x00\x00\x00") is None  # not ClientHello
    exts = SNIExtractor.extract_extensions(hello)          # was an upstream stub
    assert exts and exts[0][0] == 0x0000, exts
    assert bytes.fromhex(exts[0][1]).endswith(b"www.example.com")
    print(f"TLS SNI            -> {SNIExtractor.extract(hello)!r}")

    # QUIC: long-header packet with a Client Hello buried inside
    quic = b"\xC0" + b"\x00\x00\x00\x01" + b"\x00" * 40 + hello
    assert QUICSNIExtractor.is_quic_initial(quic)
    print(f"QUIC SNI           -> {QUICSNIExtractor.extract(quic)!r}")

    # HTTP
    req = b"GET /index.html HTTP/1.1\r\nHost: example.com:8080\r\nAccept: */*\r\n\r\n"
    assert HTTPHostExtractor.is_http_request(req)
    assert HTTPHostExtractor.extract(req) == "example.com"
    assert HTTPHostExtractor.extract(b"NOPE / HTTP/1.1\r\nHost: a.com\r\n\r\n") is None
    # FIXED: only a real header line matches, so X-Forwarded-Host cannot win
    assert (
        HTTPHostExtractor.extract(b"POST /x\r\nX-Forwarded-Host: a.com\r\nHost: b.com\r\n\r\n")
        == "b.com"
    )
    # FIXED: a Host header in the last bytes of the payload is now seen
    assert HTTPHostExtractor.extract(b"GET /\r\nHost: z.io") == "z.io"
    # FIXED: a non-hostname server-name entry no longer kills the whole walk
    import struct as _s
    other = _s.pack(">H", 4) + b"\x02" + _s.pack(">H", 1) + b"\x41"   # type 2, not hostname
    ext_bad = _s.pack(">HH", 0x0000, len(other)) + other
    good = b"good.example"
    body = _s.pack(">H", len(good) + 3) + b"\x00" + _s.pack(">H", len(good)) + good
    ext_good = _s.pack(">HH", 0x0000, len(body)) + body
    allx = ext_bad + ext_good
    ch = (_s.pack(">H", 0x0303) + b"\xAB" * 32 + b"\x00" + _s.pack(">H", 2) + b"\x13\x01"
          + b"\x01\x00" + _s.pack(">H", len(allx)) + allx)
    hs = b"\x01" + _s.pack(">I", len(ch))[1:] + ch
    rec = b"\x16\x03\x03" + _s.pack(">H", len(hs)) + hs
    assert SNIExtractor.extract(rec) == "good.example", SNIExtractor.extract(rec)
    print(f"HTTP Host          -> {HTTPHostExtractor.extract(req)!r}")

    # DNS
    query = (
        struct.pack(">HHHHHH", 0x1234, 0x0100, 1, 0, 0, 0)
        + b"\x03www\x07example\x03com\x00"
        + struct.pack(">HH", 1, 1)
    )
    assert DNSExtractor.is_dns_query(query)
    assert DNSExtractor.extract_query(query) == "www.example.com"
    response = bytearray(query)
    response[2] = 0x81  # QR bit -> response
    assert DNSExtractor.extract_query(bytes(response)) is None
    print(f"DNS query          -> {DNSExtractor.extract_query(query)!r}")

    print("sni_extractor.py self-test OK")
