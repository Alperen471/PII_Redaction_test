"""Deterministic format validators for the regex baseline (plan v4 §5.1)."""
from __future__ import annotations


def digits_only(s: str) -> str:
    return "".join(ch for ch in s if ch.isdigit())


def tckn_checksum(match: str) -> bool:
    """Turkish national id (TCKN) checksum."""
    d = digits_only(match)
    if len(d) != 11 or d[0] == "0":
        return False
    n = [int(c) for c in d]
    odd = n[0] + n[2] + n[4] + n[6] + n[8]
    even = n[1] + n[3] + n[5] + n[7]
    d10 = (odd * 7 - even) % 10
    if d10 != n[9]:
        return False
    d11 = sum(n[:10]) % 10
    return d11 == n[10]


def luhn(match: str) -> bool:
    d = digits_only(match)
    if not 13 <= len(d) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(d)):
        x = int(ch)
        if i % 2 == 1:
            x *= 2
            if x > 9:
                x -= 9
        total += x
    return total % 10 == 0


_IBAN_ALPHA = {c: str(i + 10) for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}


def iban_mod97(match: str) -> bool:
    raw = "".join(ch for ch in match if ch.isalnum()).upper()
    if len(raw) != 26 or not raw.startswith("TR"):
        return False
    rearranged = raw[4:] + raw[:4]
    digits = "".join(_IBAN_ALPHA.get(ch, ch) for ch in rearranged)
    if not digits.isdigit():
        return False
    return int(digits) % 97 == 1


def tr_phone(match: str) -> bool:
    d = digits_only(match)
    if d.startswith("90"):
        d = d[2:]
    elif d.startswith("0"):
        d = d[1:]
    if len(d) != 10:
        return False
    return d[0] in "2345"


VALIDATORS = {
    "none": lambda _s: True,
    "tckn_checksum": tckn_checksum,
    "luhn": luhn,
    "iban_mod97": iban_mod97,
    "tr_phone": tr_phone,
}
