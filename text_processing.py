"""
Text processing — normalization, tokenization, feature extraction.

Unified tokenization for the whole pipeline (no separate stem vs lemma paths).
"""
from __future__ import annotations
import re
import logging
from collections import Counter

import pymorphy3

log = logging.getLogger("cluster_engine.text")

# ── Morphological analyzer (lazy init) ─────────────────────────────────

_morph = None
_lemma_cache: dict[str, str] = {}


def _get_morph():
    global _morph
    if _morph is None:
        _morph = pymorphy3.MorphAnalyzer()
    return _morph


def lemmatize(word: str) -> str:
    w = word.lower().strip()
    if w not in _lemma_cache:
        _lemma_cache[w] = _get_morph().parse(w)[0].normal_form
    return _lemma_cache[w]


# ── Stop words ─────────────────────────────────────────────────────────

STOP_WORDS = frozenset({
    "и", "в", "на", "с", "по", "для", "из", "от", "до", "за", "к", "не",
    "или", "но", "а", "о", "у", "же", "бы", "ли", "ни", "то", "это",
    "как", "что", "так", "все", "уже", "он", "она", "они", "мы", "вы",
    "его", "её", "их", "мой", "ваш", "наш", "свой",
    "быть", "мочь", "также", "еще", "ещё", "при", "через",
    "the", "a", "an", "and", "or", "for", "in", "on", "with", "of", "to",
    "is", "are", "was", "were", "be", "been", "being",
    # marketplace noise
    "артикул", "товар", "шт", "штук", "упак", "упаковка", "набор",
})


# ── Regex patterns ─────────────────────────────────────────────────────

_RE_CLEAN = re.compile(r"[^a-zA-Zа-яА-ЯёЁ0-9\s/\-]")
_RE_SPACES = re.compile(r"\s+")
_RE_NUM_UNIT = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(мм|см|м|мг|г|кг|мл|л|вт|квт|гб|тб|мб|шт|мач|мп|гц"
    r"|mm|cm|kg|ml|gb|tb|mb|w|kw|hz|fps)\b",
    re.I,
)


# ── Strip numbers for clean embedding ─────────────────────────────────

# Numbers with units (50мм, 219шт, 1.5кВт, etc.)
_RE_NUM_WITH_UNIT = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:мм|см|м|мг|г|кг|мл|л|вт|квт|гб|тб|мб|шт|штук|мач|мп|гц"
    r"|предмет\w*|ступен\w*|секци\w*|луч\w*|звень\w*"
    r"|mm|cm|kg|ml|gb|tb|mb|w|kw|hz|pcs|pc)\b",
    re.I,
)
# Standalone numbers: only digits surrounded by spaces/punctuation (not touching letters)
_RE_STANDALONE_NUM = re.compile(r"(?<![A-Za-zА-Яа-яёЁ0-9])\d+(?:[.,]\d+)?(?![A-Za-zА-Яа-яёЁ0-9\-])")
# Dimensions like 2x5, 4х3
_RE_DIMENSIONS = re.compile(r"\b\d+\s*[xхXХ×]\s*\d+(?:\s*[xхXХ×]\s*\d+)*\b")


def strip_numbers(text: str) -> str:
    """
    Remove numbers from product name for clean text embedding.
    Strips: "50мм", "219шт", "3 ступени", "2x5", standalone digits.
    Keeps: model names (NV1150, DKAT168), brand codes.
    """
    if not text:
        return ""
    t = text
    t = _RE_DIMENSIONS.sub("", t)       # 2x5, 4х3
    t = _RE_NUM_WITH_UNIT.sub("", t)    # 50мм, 219шт, 3 ступени
    t = _RE_STANDALONE_NUM.sub("", t)   # standalone numbers
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ── Core functions ─────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Basic text normalization."""
    if not text:
        return ""
    t = text.lower().strip()
    t = _RE_CLEAN.sub(" ", t)
    t = _RE_SPACES.sub(" ", t).strip()
    return t


def normalize_brand(brand: str) -> str:
    """Normalize brand name for comparison."""
    if not brand:
        return ""
    b = brand.lower().strip()
    # Remove common suffixes
    for sfx in (" llc", " inc", " ltd", " gmbh", " ооо", " оао", " зао", " ип"):
        if b.endswith(sfx):
            b = b[: -len(sfx)].strip()
    b = _RE_CLEAN.sub("", b)
    return b.strip()


def tokenize(text: str) -> list[str]:
    """Tokenize and lemmatize text. Returns list of tokens."""
    if not text:
        return []
    t = normalize_text(text)
    words = t.split()
    tokens = []
    for w in words:
        if len(w) < 2:
            continue
        # Keep pure numbers (model numbers, sizes)
        if w.isdigit():
            if len(w) >= 2:
                tokens.append(w)
            continue
        # Keep alphanumeric tokens as-is (model codes like "a2780")
        if any(c.isdigit() for c in w) and any(c.isalpha() for c in w):
            tokens.append(w.lower())
            continue
        lem = lemmatize(w)
        if lem in STOP_WORDS or len(lem) < 2:
            continue
        tokens.append(lem)
    return tokens


def extract_numeric_attrs(text: str) -> dict[str, float]:
    """
    Extract numeric attributes: "500 мл" → {"мл": 500.0}.
    Returns dict of unit → value.
    """
    if not text:
        return {}
    t = normalize_text(text)
    attrs = {}
    for m in _RE_NUM_UNIT.finditer(t):
        val = float(m.group(1).replace(",", "."))
        unit = m.group(2).lower()
        if unit not in attrs:  # first occurrence wins
            attrs[unit] = val
    return attrs


def token_overlap(tokens_a: list[str], tokens_b: list[str]) -> float:
    """Jaccard-like overlap between two token lists."""
    if not tokens_a or not tokens_b:
        return 0.0
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return intersection / union


def brand_match(brand_a: str, brand_b: str) -> float:
    """
    Brand similarity: 1.0 = exact, 0.5 = one contains the other, 0.0 = different.
    """
    a = normalize_brand(brand_a)
    b = normalize_brand(brand_b)
    if not a or not b:
        return 0.0  # unknown brand = no signal
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.5
    return 0.0


def price_ratio(price_a: float, price_b: float) -> float:
    """
    Price divergence ratio. Returns max(a/b, b/a).
    1.0 = same price, higher = more divergent.
    """
    if price_a <= 0 or price_b <= 0:
        return 1.0  # no signal
    return max(price_a / price_b, price_b / price_a)
