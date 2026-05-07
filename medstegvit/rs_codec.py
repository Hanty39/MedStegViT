# =============================================================================
# rs_codec.py — Reed-Solomon kódování a dekódování payloadu
# =============================================================================

from reedsolo import RSCodec

rs = RSCodec(32)


def rs_encode(data: bytes) -> bytes:
    """
    Zakóduje 32 datových bajtů Reed-Solomonovým kódem na 64 bajtů.
    """
    assert len(data) == 32  # SHA-256 vždy vrací přesně 32 bajtů
    return rs.encode(data)


def rs_decode(codeword: bytes) -> bytes:
    """
    Dekóduje 64bajtové kódové slovo a opraví případné chyby.
    """
    decoded, _, _ = rs.decode(codeword)
    return decoded