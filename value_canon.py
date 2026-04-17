"""
Value canonicalizer — normalizes raw attribute values for comparison.

Used by scoring (_gemini_attr_block, _numeric_attr_score) to make
"3 шт" == "3 штуки" == "три" and "1500 Вт" == "1.5 кВт" and "щётка" == "щетка".

Two entry points:
  canon_string(s)              — strings: NFKD, ё→е, lowercase, strip punct
  canon_numeric(value, attr)   — numbers: parse "3 шт"/"1.5 кВт", convert to base unit

Both functions are forgiving — return None if input is unparseable.
"""

from __future__ import annotations
import re
import unicodedata
from typing import Any

# ── String canonicalization ──────────────────────────────────────────

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def canon_string(s: Any) -> str:
    """Normalize string: NFKD, ё→е, lowercase, strip punct, collapse whitespace."""
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("ё", "е").replace("Ё", "Е")
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


# ── Number+unit canonicalization ─────────────────────────────────────

# Maps a unit token (lowercased, no spaces) → (canonical_dimension, factor_to_base)
# Base units: m (length), w (power), g (mass), l (volume), v (voltage),
#             a (current), pa (pressure), s (time), pcs (count)
_UNIT_MAP: dict[str, tuple[str, float]] = {
    # length → meters
    "мм": ("m", 0.001), "mm": ("m", 0.001),
    "см": ("m", 0.01),  "cm": ("m", 0.01),
    "дм": ("m", 0.1),
    "м":  ("m", 1.0),   "m":  ("m", 1.0), "метр": ("m", 1.0),
    "метров": ("m", 1.0), "метра": ("m", 1.0),
    "км": ("m", 1000.0),

    # power → watts
    "вт": ("w", 1.0), "w": ("w", 1.0), "ватт": ("w", 1.0),
    "квт": ("w", 1000.0), "kw": ("w", 1000.0),
    "мвт": ("w", 1_000_000.0), "mw": ("w", 1_000_000.0),

    # mass → grams
    "мг": ("g", 0.001),
    "г":  ("g", 1.0), "g": ("g", 1.0), "грамм": ("g", 1.0),
    "кг": ("g", 1000.0), "kg": ("g", 1000.0),
    "т":  ("g", 1_000_000.0),

    # volume → liters
    "мл": ("l", 0.001), "ml": ("l", 0.001),
    "л":  ("l", 1.0), "l": ("l", 1.0), "литр": ("l", 1.0),
    "м3": ("l", 1000.0),

    # voltage → volts
    "в": ("v", 1.0), "v": ("v", 1.0), "вольт": ("v", 1.0),
    "кв": ("v", 1000.0), "kv": ("v", 1000.0),

    # current → amps
    "а": ("a", 1.0), "a": ("a", 1.0), "ампер": ("a", 1.0),
    "ма": ("a", 0.001),

    # pressure → pascals
    "па": ("pa", 1.0), "pa": ("pa", 1.0),
    "кпа": ("pa", 1000.0),
    "мпа": ("pa", 1_000_000.0), "mpa": ("pa", 1_000_000.0),
    "бар": ("pa", 100_000.0), "bar": ("pa", 100_000.0),
    "атм": ("pa", 101_325.0),

    # time → seconds
    "с":   ("s", 1.0), "сек": ("s", 1.0), "s": ("s", 1.0),
    "мин": ("s", 60.0), "min": ("s", 60.0),
    "ч":   ("s", 3600.0), "час": ("s", 3600.0), "h": ("s", 3600.0),

    # count → pieces
    "шт":      ("pcs", 1.0), "штук": ("pcs", 1.0), "штука": ("pcs", 1.0),
    "штуки":   ("pcs", 1.0), "штуку": ("pcs", 1.0), "штукой": ("pcs", 1.0),
    "pcs":     ("pcs", 1.0), "pc": ("pcs", 1.0),
    "пр":      ("pcs", 1.0), "пред": ("pcs", 1.0),
    "предмет": ("pcs", 1.0), "предметов": ("pcs", 1.0), "предмета": ("pcs", 1.0),
}

# Words → digits (Russian)
_WORD_NUM = {
    "ноль": 0, "один": 1, "одна": 1, "одно": 1,
    "два": 2, "две": 2, "три": 3, "четыре": 4,
    "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9,
    "десять": 10, "одиннадцать": 11, "двенадцать": 12,
    "сто": 100, "тысяча": 1000,
}

_NUM_UNIT_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*([a-zа-я]+)?",
    re.IGNORECASE,
)


def canon_numeric(value: Any, hint_unit: str | None = None) -> tuple[float, str] | None:
    """
    Parse value into (canonical_value_in_base_unit, dimension).
    Examples:
      "1.5 кВт"  → (1500.0, 'w')
      "3 шт"     → (3.0, 'pcs')
      "три"      → (3.0, 'pcs')   # only with hint_unit='pcs' or attr context
      15         → (15.0, '')      # bare number, no unit
      "10 м"     → (10.0, 'm')

    Returns None if unparseable.
    """
    if value is None or isinstance(value, bool):
        return None

    # Bare number (int/float)
    if isinstance(value, (int, float)):
        v = float(value)
        if hint_unit:
            mapped = _UNIT_MAP.get(canon_string(hint_unit).replace(" ", ""))
            if mapped:
                dim, factor = mapped
                return v * factor, dim
        return v, ""

    # String form — light normalization (preserve dot/comma for decimals)
    s = str(value)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("ё", "е").replace("Ё", "Е").lower().strip()
    if not s:
        return None

    # Word number ("три")
    if s in _WORD_NUM:
        return float(_WORD_NUM[s]), "pcs"

    # Try regex: digit + optional unit
    m = _NUM_UNIT_RE.search(s)
    if not m:
        return None

    try:
        num = float(m.group(1).replace(",", "."))
    except ValueError:
        return None

    unit_tok = (m.group(2) or "").strip()
    if not unit_tok and hint_unit:
        unit_tok = canon_string(hint_unit).replace(" ", "")

    if unit_tok:
        mapped = _UNIT_MAP.get(unit_tok)
        if mapped:
            dim, factor = mapped
            return num * factor, dim

    return num, ""


def canon_for_compare(value: Any, hint_unit: str | None = None) -> Any:
    """
    Return a canonical form suitable for equality/range comparison.
    Numbers → float in base unit. Strings → normalized string.
    Used by scoring to compare values.
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        result = canon_numeric(value, hint_unit)
        return result[0] if result else float(value)
    # Try numeric parse first (handles "3 шт", "1.5 кВт")
    result = canon_numeric(value, hint_unit)
    if result is not None:
        return result[0]
    return canon_string(value)
