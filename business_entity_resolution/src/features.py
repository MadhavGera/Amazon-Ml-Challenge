"""
features.py — Feature engineering for entity-pair classification.

Person B track (Phase B2).

This module consumes:
  - The Candidate pairs DataFrame (Schema 3, from blocking.py)
  - Normalized records DataFrames (Schema 2, from normalize.py)

And produces the Feature table (Schema 4), which is the direct input to
train.py and predict.py.

All feature column names are defined in config.FEATURE_TABLE_COLUMNS.
All computations handle missing/empty values gracefully without producing NaNs.

Feature groups:
1. Name similarity:
   - lev_dist_raw (normalized Levenshtein distance on raw_name, 0=identical)
   - jaro_winkler_raw (Jaro-Winkler similarity on raw_name, 1=identical)
   - token_sort_ratio_norm (RapidFuzz token_sort_ratio on name_core, in [0, 1])
   - token_set_ratio_norm (RapidFuzz token_set_ratio on name_core, in [0, 1])
   - tfidf_cosine_char (character 2-4 gram cosine similarity on name_core)
   - tfidf_cosine_word (word 1-2 gram cosine similarity on name_core)
   - jaccard_tokens_full (Jaccard similarity on raw_name tokens)
   - jaccard_tokens_core (Jaccard similarity on name_core tokens)
   - first_token_match (binary 1.0/0.0: first token of name_core matches)
   - name_len_diff_ratio (relative length difference on name_core)

2. Address similarity:
   - addr_lev_dist (normalized Levenshtein distance on address_expanded)
   - addr_jaro_winkler (Jaro-Winkler similarity on address_expanded)
   - addr_token_sort_ratio (RapidFuzz token_sort_ratio on address_expanded)
   - addr_token_set_ratio (RapidFuzz token_set_ratio on address_expanded)
   - addr_jaccard (Jaccard similarity on address_expanded tokens)
   - house_number_match (binary 1.0/0.0: non-empty house numbers match)
   - postal_prefix_match (binary 1.0/0.0: non-empty postal prefixes match)
   - city_token_overlap (heuristic overlap between city-region tokens)
   - landmark_flag (binary 1.0/0.0: True if EITHER record has a non-empty landmark)

3. Structural / meta features:
   - same_country_flag (binary 1.0/0.0: country_norm matches and non-empty)
   - candidate_score_gap (top-1 blocking_score for the S1 entity minus candidate's score)

4. Blocking provenance:
   - blocked_by_name_token (binary 1.0/0.0)
   - blocked_by_snm (binary 1.0/0.0)
   - blocked_by_char_tfidf (binary 1.0/0.0)
   - blocked_by_word_tfidf (binary 1.0/0.0)
   - blocked_by_addr_token (binary 1.0/0.0)
   - blocked_by_minhash (binary 1.0/0.0)
"""

from __future__ import annotations

import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

# Ensure src/ is on the path when imported or run standalone
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    CAND_COL_BLOCKING_SCORE,
    CAND_COL_BLOCKING_STRATEGY_FLAGS,
    CAND_COL_CANDIDATE_ENTITY_ID,
    CAND_COL_SOURCE1_ENTITY_ID,
    FEAT_COL_ADDR_JARO_WINKLER,
    FEAT_COL_ADDR_JACCARD,
    FEAT_COL_ADDR_LEV_DIST,
    FEAT_COL_ADDR_TOKEN_SET_RATIO,
    FEAT_COL_ADDR_TOKEN_SORT_RATIO,
    FEAT_COL_BLOCKED_BY_ADDR_TOKEN,
    FEAT_COL_BLOCKED_BY_CHAR_TFIDF,
    FEAT_COL_BLOCKED_BY_MINHASH,
    FEAT_COL_BLOCKED_BY_NAME_TOKEN,
    FEAT_COL_BLOCKED_BY_SNM,
    FEAT_COL_BLOCKED_BY_WORD_TFIDF,
    FEAT_COL_CANDIDATE_ENTITY_ID,
    FEAT_COL_CANDIDATE_SCORE_GAP,
    FEAT_COL_CITY_TOKEN_OVERLAP,
    FEAT_COL_FIRST_TOKEN_MATCH,
    FEAT_COL_HOUSE_NUMBER_MATCH,
    FEAT_COL_JACCARD_TOKENS_CORE,
    FEAT_COL_JACCARD_TOKENS_FULL,
    FEAT_COL_JARO_WINKLER_RAW,
    FEAT_COL_LANDMARK_FLAG,
    FEAT_COL_LEV_DIST_RAW,
    FEAT_COL_NAME_LEN_DIFF_RATIO,
    FEAT_COL_POSTAL_PREFIX_MATCH,
    FEAT_COL_SAME_COUNTRY_FLAG,
    FEAT_COL_SOURCE1_ENTITY_ID,
    FEAT_COL_TFIDF_COSINE_CHAR,
    FEAT_COL_TFIDF_COSINE_WORD,
    FEAT_COL_TOKEN_SET_RATIO_NORM,
    FEAT_COL_TOKEN_SORT_RATIO_NORM,
    FEATURE_COLUMNS,
    FEATURE_TABLE_COLUMNS,
    NORM_COL_ADDRESS_EXPANDED,
    NORM_COL_COUNTRY_NORM,
    NORM_COL_ENTITY_ID,
    NORM_COL_HOUSE_NUMBER,
    NORM_COL_LANDMARK,
    NORM_COL_NAME_CORE,
    NORM_COL_NAME_EXPANDED,
    NORM_COL_POSTAL_PREFIX,
    NORM_COL_RAW_ADDRESS,
    NORM_COL_RAW_NAME,
)


# ─────────────────────────────────────────────────────────────────────────────
# TOKEN & N-GRAM UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric tokens."""
    if not text:
        return []
    return _WORD_RE.findall(text.lower())


def _char_ngrams(text: str, n_min: int = 2, n_max: int = 4) -> Counter[str]:
    """Extract character n-grams of lengths n_min through n_max."""
    cleaned = text.strip().lower()
    if not cleaned:
        return Counter()
    counts: Counter[str] = Counter()
    length = len(cleaned)
    for n in range(n_min, min(n_max + 1, length + 1)):
        for i in range(length - n + 1):
            counts[cleaned[i : i + n]] += 1
    return counts


def _word_ngrams(tokens: list[str], n_min: int = 1, n_max: int = 2) -> Counter[str]:
    """Extract word n-grams from token list."""
    if not tokens:
        return Counter()
    counts: Counter[str] = Counter()
    num_tokens = len(tokens)
    for n in range(n_min, min(n_max + 1, num_tokens + 1)):
        for i in range(num_tokens - n + 1):
            ngram = " ".join(tokens[i : i + n])
            counts[ngram] += 1
    return counts


def _cosine_similarity(c1: Counter[str], c2: Counter[str]) -> float:
    """
    Compute cosine similarity between two term-frequency counters.
    Returns 0.0 if either counter is empty.
    """
    if not c1 or not c2:
        return 0.0
    dot_product = sum(count * c2[term] for term, count in c1.items() if term in c2)
    if dot_product == 0:
        return 0.0
    norm1 = math.sqrt(sum(count * count for count in c1.values()))
    norm2 = math.sqrt(sum(count * count for count in c2.values()))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(dot_product / (norm1 * norm2))


def _jaccard_similarity(tokens1: Sequence[str], tokens2: Sequence[str]) -> float:
    """Compute Jaccard similarity of two token sequences."""
    s1 = set(tokens1)
    s2 = set(tokens2)
    if not s1 and not s2:
        return 0.0
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    return float(intersection / union) if union > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 1. PURE FEATURE FUNCTIONS (PAIRWISE)
# ─────────────────────────────────────────────────────────────────────────────

def compute_lev_dist_raw(s1: str, s2: str) -> float:
    """
    Normalized Levenshtein distance on raw_name (0.0 = identical, 1.0 = completely different).
    If both strings are empty, returns 0.0.
    """
    s1_str = str(s1) if s1 is not None else ""
    s2_str = str(s2) if s2 is not None else ""
    if not s1_str and not s2_str:
        return 0.0
    return float(Levenshtein.normalized_distance(s1_str, s2_str))


def compute_jaro_winkler_raw(s1: str, s2: str) -> float:
    """
    Jaro-Winkler similarity on raw_name (1.0 = identical, 0.0 = completely different).
    If both strings are empty, returns 1.0.
    """
    s1_str = str(s1) if s1 is not None else ""
    s2_str = str(s2) if s2 is not None else ""
    if not s1_str and not s2_str:
        return 1.0
    return float(JaroWinkler.similarity(s1_str, s2_str))


def compute_token_sort_ratio_norm(s1: str, s2: str) -> float:
    """RapidFuzz token_sort_ratio normalized to [0, 1]."""
    s1_str = str(s1) if s1 is not None else ""
    s2_str = str(s2) if s2 is not None else ""
    return float(fuzz.token_sort_ratio(s1_str, s2_str) / 100.0)


def compute_token_set_ratio_norm(s1: str, s2: str) -> float:
    """RapidFuzz token_set_ratio normalized to [0, 1]."""
    s1_str = str(s1) if s1 is not None else ""
    s2_str = str(s2) if s2 is not None else ""
    return float(fuzz.token_set_ratio(s1_str, s2_str) / 100.0)


def compute_tfidf_cosine_char(s1: str, s2: str) -> float:
    """Character n-gram (2-4) TF cosine similarity between two strings."""
    c1 = _char_ngrams(str(s1) if s1 is not None else "")
    c2 = _char_ngrams(str(s2) if s2 is not None else "")
    return _cosine_similarity(c1, c2)


def compute_tfidf_cosine_word(s1: str, s2: str) -> float:
    """Word n-gram (1-2) TF cosine similarity between two strings."""
    t1 = _tokenize(str(s1) if s1 is not None else "")
    t2 = _tokenize(str(s2) if s2 is not None else "")
    w1 = _word_ngrams(t1)
    w2 = _word_ngrams(t2)
    return _cosine_similarity(w1, w2)


def compute_jaccard_tokens(s1: str, s2: str) -> float:
    """Jaccard similarity on word token sets."""
    t1 = _tokenize(str(s1) if s1 is not None else "")
    t2 = _tokenize(str(s2) if s2 is not None else "")
    return _jaccard_similarity(t1, t2)


def compute_first_token_match(s1: str, s2: str) -> float:
    """Binary 1.0/0.0: first token of both strings match and is non-empty."""
    t1 = _tokenize(str(s1) if s1 is not None else "")
    t2 = _tokenize(str(s2) if s2 is not None else "")
    if t1 and t2 and t1[0] == t2[0]:
        return 1.0
    return 0.0


def compute_name_len_diff_ratio(s1: str, s2: str) -> float:
    """
    Relative length difference between two strings:
    abs(len(a) - len(b)) / max(len(a), len(b), 1)
    """
    len1 = len(str(s1)) if s1 is not None else 0
    len2 = len(str(s2)) if s2 is not None else 0
    max_len = max(len1, len2, 1)
    return float(abs(len1 - len2) / max_len)


def compute_house_number_match(hn1: str, hn2: str) -> float:
    """
    Binary 1.0/0.0: house_number fields are non-empty and equal.
    Both-empty returns 0.0 (absence of house number is not evidence of match).
    """
    h1 = str(hn1).strip().lower() if hn1 is not None else ""
    h2 = str(hn2).strip().lower() if hn2 is not None else ""
    if h1 and h2 and h1 == h2:
        return 1.0
    return 0.0


def compute_postal_prefix_match(p1: str, p2: str) -> float:
    """
    Binary 1.0/0.0: postal_prefix fields are non-empty and equal.
    Both-empty returns 0.0 (absence of postal prefix is not evidence of match).
    """
    post1 = str(p1).strip().lower() if p1 is not None else ""
    post2 = str(p2).strip().lower() if p2 is not None else ""
    if post1 and post2 and post1 == post2:
        return 1.0
    return 0.0


def _extract_city_tokens(address: str) -> set[str]:
    """
    Heuristic extraction of city-position tokens from an address string.

    Heuristic design:
    1. If address contains commas, the city token(s) commonly reside in the
       penultimate or middle segment (before state/postal/country).
    2. If no commas, we examine the final 2-4 tokens of the address, filtering out
       pure numbers (postal codes / suite numbers) and standard address stop tokens
       (e.g., 'st', 'ave', 'rd', 'suite', 'floor', 'usa', 'us', 'uk', 'in').
    """
    addr = str(address).strip().lower() if address is not None else ""
    if not addr:
        return set()

    stop_tokens = {
        "st", "street", "ave", "avenue", "rd", "road", "blvd", "boulevard",
        "dr", "drive", "ln", "lane", "way", "hwy", "highway", "pkwy", "parkway",
        "suite", "ste", "fl", "floor", "unit", "apt", "apartment", "building",
        "bldg", "no", "near", "opp", "opposite", "behind", "us", "usa", "in",
        "uk", "ca", "de", "fr",
    }

    if "," in addr:
        segments = [seg.strip() for seg in addr.split(",") if seg.strip()]
        if len(segments) >= 2:
            # Segment before last (or second segment in 2-segment address)
            target_segment = segments[-2] if len(segments) > 2 else segments[-1]
            tokens = _tokenize(target_segment)
            city_tokens = {t for t in tokens if t not in stop_tokens and not t.isdigit()}
            if city_tokens:
                return city_tokens

    # Fallback to trailing tokens
    all_tokens = _tokenize(addr)
    if not all_tokens:
        return set()
    # Take up to 4 trailing tokens
    trailing = all_tokens[-min(4, len(all_tokens)) :]
    return {t for t in trailing if t not in stop_tokens and not t.isdigit()}


def compute_city_token_overlap(addr1: str, addr2: str) -> float:
    """
    Token overlap between candidate city tokens extracted from address_expanded.
    Returns Jaccard similarity between the extracted city token sets.
    """
    c1 = _extract_city_tokens(addr1)
    c2 = _extract_city_tokens(addr2)
    if not c1 or not c2:
        return 0.0
    return float(len(c1 & c2) / len(c1 | c2))


def compute_landmark_flag(lm1: str, lm2: str) -> float:
    """
    Binary 1.0/0.0: True if EITHER record's landmark field is non-empty.
    Signals landmark-based addressing is present on either side.
    """
    l1 = str(lm1).strip() if lm1 is not None else ""
    l2 = str(lm2).strip() if lm2 is not None else ""
    return 1.0 if (len(l1) > 0 or len(l2) > 0) else 0.0


def compute_same_country_flag(c1: str, c2: str) -> float:
    """
    Binary 1.0/0.0: True if country_norm fields match and are non-empty.
    """
    cnt1 = str(c1).strip().upper() if c1 is not None else ""
    cnt2 = str(c2).strip().upper() if c2 is not None else ""
    if cnt1 and cnt2 and cnt1 == cnt2:
        return 1.0
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. VECTORIZED FEATURE GROUPS (FAST BATCH COMPUTATION)
# ─────────────────────────────────────────────────────────────────────────────

def compute_name_features(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all name-similarity features for a merged pair DataFrame.

    Expects columns:
        raw_name_s1, name_expanded_s1, name_core_s1,
        raw_name_c,  name_expanded_c,  name_core_c.

    Returns DataFrame with columns:
        lev_dist_raw, jaro_winkler_raw, token_sort_ratio_norm,
        token_set_ratio_norm, tfidf_cosine_char, tfidf_cosine_word,
        jaccard_tokens_full, jaccard_tokens_core,
        first_token_match, name_len_diff_ratio.
    """
    raw_s1 = pairs.get("raw_name_s1", pd.Series("", index=pairs.index)).fillna("").astype(str).tolist()
    raw_c = pairs.get("raw_name_c", pd.Series("", index=pairs.index)).fillna("").astype(str).tolist()
    core_s1 = pairs.get("name_core_s1", pd.Series("", index=pairs.index)).fillna("").astype(str).tolist()
    core_c = pairs.get("name_core_c", pd.Series("", index=pairs.index)).fillna("").astype(str).tolist()

    lev_dist_raw = [compute_lev_dist_raw(a, b) for a, b in zip(raw_s1, raw_c)]
    jaro_winkler_raw = [compute_jaro_winkler_raw(a, b) for a, b in zip(raw_s1, raw_c)]
    token_sort_ratio_norm = [compute_token_sort_ratio_norm(a, b) for a, b in zip(core_s1, core_c)]
    token_set_ratio_norm = [compute_token_set_ratio_norm(a, b) for a, b in zip(core_s1, core_c)]
    tfidf_cosine_char = [compute_tfidf_cosine_char(a, b) for a, b in zip(core_s1, core_c)]
    tfidf_cosine_word = [compute_tfidf_cosine_word(a, b) for a, b in zip(core_s1, core_c)]
    jaccard_tokens_full = [compute_jaccard_tokens(a, b) for a, b in zip(raw_s1, raw_c)]
    jaccard_tokens_core = [compute_jaccard_tokens(a, b) for a, b in zip(core_s1, core_c)]
    first_token_match = [compute_first_token_match(a, b) for a, b in zip(core_s1, core_c)]
    name_len_diff_ratio = [compute_name_len_diff_ratio(a, b) for a, b in zip(core_s1, core_c)]

    return pd.DataFrame(
        {
            FEAT_COL_LEV_DIST_RAW: lev_dist_raw,
            FEAT_COL_JARO_WINKLER_RAW: jaro_winkler_raw,
            FEAT_COL_TOKEN_SORT_RATIO_NORM: token_sort_ratio_norm,
            FEAT_COL_TOKEN_SET_RATIO_NORM: token_set_ratio_norm,
            FEAT_COL_TFIDF_COSINE_CHAR: tfidf_cosine_char,
            FEAT_COL_TFIDF_COSINE_WORD: tfidf_cosine_word,
            FEAT_COL_JACCARD_TOKENS_FULL: jaccard_tokens_full,
            FEAT_COL_JACCARD_TOKENS_CORE: jaccard_tokens_core,
            FEAT_COL_FIRST_TOKEN_MATCH: first_token_match,
            FEAT_COL_NAME_LEN_DIFF_RATIO: name_len_diff_ratio,
        },
        index=pairs.index,
    )


def compute_address_features(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all address-similarity features for a merged pair DataFrame.

    Expects columns:
        address_expanded_s1, house_number_s1, postal_prefix_s1, landmark_s1,
        address_expanded_c,  house_number_c,  postal_prefix_c,  landmark_c.

    Returns DataFrame with columns:
        addr_lev_dist, addr_jaro_winkler, addr_token_sort_ratio,
        addr_token_set_ratio, addr_jaccard, house_number_match,
        postal_prefix_match, city_token_overlap, landmark_flag.
    """
    addr_s1 = pairs.get("address_expanded_s1", pd.Series("", index=pairs.index)).fillna("").astype(str).tolist()
    addr_c = pairs.get("address_expanded_c", pd.Series("", index=pairs.index)).fillna("").astype(str).tolist()
    hn_s1 = pairs.get("house_number_s1", pd.Series("", index=pairs.index)).fillna("").astype(str).tolist()
    hn_c = pairs.get("house_number_c", pd.Series("", index=pairs.index)).fillna("").astype(str).tolist()
    post_s1 = pairs.get("postal_prefix_s1", pd.Series("", index=pairs.index)).fillna("").astype(str).tolist()
    post_c = pairs.get("postal_prefix_c", pd.Series("", index=pairs.index)).fillna("").astype(str).tolist()
    lm_s1 = pairs.get("landmark_s1", pd.Series("", index=pairs.index)).fillna("").astype(str).tolist()
    lm_c = pairs.get("landmark_c", pd.Series("", index=pairs.index)).fillna("").astype(str).tolist()

    addr_lev_dist = [compute_lev_dist_raw(a, b) for a, b in zip(addr_s1, addr_c)]
    addr_jaro_winkler = [compute_jaro_winkler_raw(a, b) for a, b in zip(addr_s1, addr_c)]
    addr_token_sort_ratio = [compute_token_sort_ratio_norm(a, b) for a, b in zip(addr_s1, addr_c)]
    addr_token_set_ratio = [compute_token_set_ratio_norm(a, b) for a, b in zip(addr_s1, addr_c)]
    addr_jaccard = [compute_jaccard_tokens(a, b) for a, b in zip(addr_s1, addr_c)]
    house_number_match = [compute_house_number_match(a, b) for a, b in zip(hn_s1, hn_c)]
    postal_prefix_match = [compute_postal_prefix_match(a, b) for a, b in zip(post_s1, post_c)]
    city_token_overlap = [compute_city_token_overlap(a, b) for a, b in zip(addr_s1, addr_c)]
    landmark_flag = [compute_landmark_flag(a, b) for a, b in zip(lm_s1, lm_c)]

    return pd.DataFrame(
        {
            FEAT_COL_ADDR_LEV_DIST: addr_lev_dist,
            FEAT_COL_ADDR_JARO_WINKLER: addr_jaro_winkler,
            FEAT_COL_ADDR_TOKEN_SORT_RATIO: addr_token_sort_ratio,
            FEAT_COL_ADDR_TOKEN_SET_RATIO: addr_token_set_ratio,
            FEAT_COL_ADDR_JACCARD: addr_jaccard,
            FEAT_COL_HOUSE_NUMBER_MATCH: house_number_match,
            FEAT_COL_POSTAL_PREFIX_MATCH: postal_prefix_match,
            FEAT_COL_CITY_TOKEN_OVERLAP: city_token_overlap,
            FEAT_COL_LANDMARK_FLAG: landmark_flag,
        },
        index=pairs.index,
    )


def compute_meta_features(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Compute same_country_flag and candidate_score_gap.

    candidate_score_gap = top-1 blocking_score for this source1_entity_id
                          minus this pair's blocking_score.
    """
    country_s1 = pairs.get("country_norm_s1", pd.Series("", index=pairs.index)).fillna("").astype(str).tolist()
    country_c = pairs.get("country_norm_c", pd.Series("", index=pairs.index)).fillna("").astype(str).tolist()

    same_country_flag = [compute_same_country_flag(a, b) for a, b in zip(country_s1, country_c)]

    # Compute candidate_score_gap via groupby-transform
    if CAND_COL_BLOCKING_SCORE in pairs.columns and CAND_COL_SOURCE1_ENTITY_ID in pairs.columns:
        scores = pd.to_numeric(pairs[CAND_COL_BLOCKING_SCORE], errors="coerce").fillna(0.0)
        top_scores = scores.groupby(pairs[CAND_COL_SOURCE1_ENTITY_ID]).transform("max")
        score_gap = (top_scores - scores).astype(float).tolist()
    else:
        score_gap = [0.0] * len(pairs)

    return pd.DataFrame(
        {
            FEAT_COL_SAME_COUNTRY_FLAG: same_country_flag,
            FEAT_COL_CANDIDATE_SCORE_GAP: score_gap,
        },
        index=pairs.index,
    )


def unpack_blocking_flags(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the comma-separated blocking_strategy_flags column into binary
    indicator columns (blocked_by_name_token, blocked_by_snm, etc.).
    """
    flags_raw = (
        pairs.get(CAND_COL_BLOCKING_STRATEGY_FLAGS, pd.Series("", index=pairs.index))
        .fillna("")
        .astype(str)
    )

    # Fast set-membership checks
    def _parse_flags(cell: str) -> set[str]:
        return {f.strip().lower() for f in cell.split(",") if f.strip()}

    flag_sets = [_parse_flags(f) for f in flags_raw]

    return pd.DataFrame(
        {
            FEAT_COL_BLOCKED_BY_NAME_TOKEN: [1.0 if "name_token" in s else 0.0 for s in flag_sets],
            FEAT_COL_BLOCKED_BY_SNM: [1.0 if "snm" in s else 0.0 for s in flag_sets],
            FEAT_COL_BLOCKED_BY_CHAR_TFIDF: [1.0 if "char_tfidf" in s else 0.0 for s in flag_sets],
            FEAT_COL_BLOCKED_BY_WORD_TFIDF: [1.0 if "word_tfidf" in s else 0.0 for s in flag_sets],
            FEAT_COL_BLOCKED_BY_ADDR_TOKEN: [1.0 if "addr_token" in s else 0.0 for s in flag_sets],
            FEAT_COL_BLOCKED_BY_MINHASH: [1.0 if "minhash" in s else 0.0 for s in flag_sets],
        },
        index=pairs.index,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. MAIN ASSEMBLY FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_table(
    candidate_pairs: pd.DataFrame,
    source1_norm: pd.DataFrame,
    candidates_norm: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Join normalized records onto candidate pairs and compute all features.

    Parameters
    ----------
    candidate_pairs : pd.DataFrame
        Schema 3 output containing:
            source1_entity_id, candidate_entity_id,
            blocking_strategy_flags, blocking_score
    source1_norm : pd.DataFrame
        Normalized records (Schema 2) containing source1 records (or all records).
    candidates_norm : pd.DataFrame | None
        Normalized records (Schema 2) for candidate records (source2/source3).
        If None, lookups for candidate records will be performed against source1_norm.

    Returns
    -------
    pd.DataFrame
        Schema 4 feature table with columns config.FEATURE_TABLE_COLUMNS.
        Guaranteed zero null/NaN values.
    """
    if candidate_pairs.empty:
        # Return empty DataFrame with exact Schema 4 columns and float dtypes
        df_empty = pd.DataFrame(columns=list(FEATURE_TABLE_COLUMNS))
        for col in FEATURE_COLUMNS:
            df_empty[col] = df_empty[col].astype(float)
        return df_empty

    # Combine normalized pools if two separate DataFrames were passed
    if candidates_norm is not None:
        norm_pool = pd.concat([source1_norm, candidates_norm], ignore_index=True)
        # Drop duplicates on entity_id to ensure 1-to-1 join
        norm_pool = norm_pool.drop_duplicates(subset=[NORM_COL_ENTITY_ID])
    else:
        norm_pool = source1_norm.drop_duplicates(subset=[NORM_COL_ENTITY_ID])

    # Left join source1 records
    merged = candidate_pairs.merge(
        norm_pool,
        left_on=CAND_COL_SOURCE1_ENTITY_ID,
        right_on=NORM_COL_ENTITY_ID,
        how="left",
        suffixes=("", "_s1"),
    )
    # Rename joined columns with _s1 suffix if not already present
    rename_s1 = {
        NORM_COL_RAW_NAME: "raw_name_s1",
        NORM_COL_NAME_EXPANDED: "name_expanded_s1",
        NORM_COL_NAME_CORE: "name_core_s1",
        NORM_COL_RAW_ADDRESS: "raw_address_s1",
        NORM_COL_ADDRESS_EXPANDED: "address_expanded_s1",
        NORM_COL_HOUSE_NUMBER: "house_number_s1",
        NORM_COL_POSTAL_PREFIX: "postal_prefix_s1",
        NORM_COL_LANDMARK: "landmark_s1",
        NORM_COL_COUNTRY_NORM: "country_norm_s1",
    }
    merged = merged.rename(columns=rename_s1)

    # Left join candidate records
    merged = merged.merge(
        norm_pool,
        left_on=CAND_COL_CANDIDATE_ENTITY_ID,
        right_on=NORM_COL_ENTITY_ID,
        how="left",
        suffixes=("", "_c"),
    )
    rename_c = {
        NORM_COL_RAW_NAME: "raw_name_c",
        NORM_COL_NAME_EXPANDED: "name_expanded_c",
        NORM_COL_NAME_CORE: "name_core_c",
        NORM_COL_RAW_ADDRESS: "raw_address_c",
        NORM_COL_ADDRESS_EXPANDED: "address_expanded_c",
        NORM_COL_HOUSE_NUMBER: "house_number_c",
        NORM_COL_POSTAL_PREFIX: "postal_prefix_c",
        NORM_COL_LANDMARK: "landmark_c",
        NORM_COL_COUNTRY_NORM: "country_norm_c",
    }
    merged = merged.rename(columns=rename_c)

    # Compute feature groups
    name_feats = compute_name_features(merged)
    addr_feats = compute_address_features(merged)
    meta_feats = compute_meta_features(merged)
    block_feats = unpack_blocking_flags(merged)

    # Assemble full table
    feature_table = pd.concat(
        [
            merged[[CAND_COL_SOURCE1_ENTITY_ID, CAND_COL_CANDIDATE_ENTITY_ID]],
            name_feats,
            addr_feats,
            meta_feats,
            block_feats,
        ],
        axis=1,
    )

    # Ensure column ordering strictly matches FEATURE_TABLE_COLUMNS
    feature_table = feature_table[list(FEATURE_TABLE_COLUMNS)]

    # Cast all feature columns to float and fill any residual NaNs with safe defaults (0.0)
    for col in FEATURE_COLUMNS:
        feature_table[col] = pd.to_numeric(feature_table[col], errors="coerce").fillna(0.0).astype(float)

    return feature_table


# ─────────────────────────────────────────────────────────────────────────────
# 4. MOCK DATA VALIDATION SUITE
# ─────────────────────────────────────────────────────────────────────────────

def _create_mock_dataset() -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """
    Construct rich mock dataset covering:
    1. Multi-match group (1 S1 entity with 3 true matches and 1 decoy candidate)
    2. True match with legal suffix / abbreviation variation
    3. Hard negative (name lookalike, different address & city)
    4. Landmark address example (unstructured address, no house number)
    5. Missing / empty field example (tests null safety defaults)
    """
    # 1. Normalized records
    norm_records = [
        # S1-001 multi-match group
        {
            "entity_id": "S1-001",
            "raw_name": "Starbucks Coffee Co",
            "name_expanded": "Starbucks Coffee Company",
            "name_core": "Starbucks Coffee",
            "raw_address": "100 Pine Street Suite 200 Seattle WA",
            "address_expanded": "100 Pine Street Suite 200 Seattle WA",
            "house_number": "100",
            "postal_prefix": "98101",
            "landmark": "",
            "country_norm": "US",
        },
        {
            "entity_id": "S2-001",
            "raw_name": "Starbucks Coffee",
            "name_expanded": "Starbucks Coffee",
            "name_core": "Starbucks Coffee",
            "raw_address": "100 Pine St Ste 200 Seattle WA",
            "address_expanded": "100 Pine Street Suite 200 Seattle WA",
            "house_number": "100",
            "postal_prefix": "98101",
            "landmark": "",
            "country_norm": "US",
        },
        {
            "entity_id": "S2-002",
            "raw_name": "Starbucks Corp",
            "name_expanded": "Starbucks Corporation",
            "name_core": "Starbucks",
            "raw_address": "100 Pine Street Seattle Washington",
            "address_expanded": "100 Pine Street Seattle Washington",
            "house_number": "100",
            "postal_prefix": "98101",
            "landmark": "",
            "country_norm": "US",
        },
        {
            "entity_id": "S3-001",
            "raw_name": "Starbucks #142",
            "name_expanded": "Starbucks Number 142",
            "name_core": "Starbucks",
            "raw_address": "100 Pine St Seattle WA",
            "address_expanded": "100 Pine Street Seattle WA",
            "house_number": "100",
            "postal_prefix": "98101",
            "landmark": "",
            "country_norm": "US",
        },
        {
            "entity_id": "S3-002",
            "raw_name": "Pine Street Bakery LLC",
            "name_expanded": "Pine Street Bakery Limited Liability Company",
            "name_core": "Pine Street Bakery",
            "raw_address": "100 Pine St Ste 100 Seattle WA",
            "address_expanded": "100 Pine Street Suite 100 Seattle WA",
            "house_number": "100",
            "postal_prefix": "98101",
            "landmark": "",
            "country_norm": "US",
        },
        # S1-002: True match with abbreviation/legal noise
        {
            "entity_id": "S1-002",
            "raw_name": "Acme Industrial Supplies Inc",
            "name_expanded": "Acme Industrial Supplies Incorporated",
            "name_core": "Acme Industrial Supplies",
            "raw_address": "450 Industrial Parkway Cleveland OH",
            "address_expanded": "450 Industrial Parkway Cleveland Ohio",
            "house_number": "450",
            "postal_prefix": "44101",
            "landmark": "",
            "country_norm": "US",
        },
        {
            "entity_id": "S2-003",
            "raw_name": "Acme Industrial Supplies LLC",
            "name_expanded": "Acme Industrial Supplies Limited Liability Company",
            "name_core": "Acme Industrial Supplies",
            "raw_address": "450 Industrial Pkwy Cleveland Ohio",
            "address_expanded": "450 Industrial Parkway Cleveland Ohio",
            "house_number": "450",
            "postal_prefix": "44101",
            "landmark": "",
            "country_norm": "US",
        },
        # S1-003: Hard negative (lookalike name, completely different location)
        {
            "entity_id": "S1-003",
            "raw_name": "Target Logistics Solutions",
            "name_expanded": "Target Logistics Solutions",
            "name_core": "Target Logistics Solutions",
            "raw_address": "1200 Commerce Blvd Dallas TX",
            "address_expanded": "1200 Commerce Boulevard Dallas Texas",
            "house_number": "1200",
            "postal_prefix": "75201",
            "landmark": "",
            "country_norm": "US",
        },
        {
            "entity_id": "S2-004",
            "raw_name": "Target Logistics Group",
            "name_expanded": "Target Logistics Group",
            "name_core": "Target Logistics Group",
            "raw_address": "800 Pacific Hwy San Diego CA",
            "address_expanded": "800 Pacific Highway San Diego California",
            "house_number": "800",
            "postal_prefix": "92101",
            "landmark": "",
            "country_norm": "US",
        },
        # S1-004: Landmark address example (unstructured, no house number)
        {
            "entity_id": "S1-004",
            "raw_name": "Sharma Sweets & Caterers",
            "name_expanded": "Sharma Sweets and Caterers",
            "name_core": "Sharma Sweets",
            "raw_address": "Near Metro Station Sector 18 Noida UP",
            "address_expanded": "Near Metro Station Sector 18 Noida Uttar Pradesh",
            "house_number": "",
            "postal_prefix": "201301",
            "landmark": "Metro Station",
            "country_norm": "IN",
        },
        {
            "entity_id": "S3-003",
            "raw_name": "Sharma Sweets",
            "name_expanded": "Sharma Sweets",
            "name_core": "Sharma Sweets",
            "raw_address": "Opposite Metro Gate 2 Sector 18 Noida",
            "address_expanded": "Opposite Metro Gate 2 Sector 18 Noida",
            "house_number": "",
            "postal_prefix": "201301",
            "landmark": "Metro Gate 2",
            "country_norm": "IN",
        },
        # S1-005: Missing / empty field example
        {
            "entity_id": "S1-005",
            "raw_name": "Apex Dynamics",
            "name_expanded": "Apex Dynamics",
            "name_core": "Apex Dynamics",
            "raw_address": "",
            "address_expanded": "",
            "house_number": "",
            "postal_prefix": "",
            "landmark": "",
            "country_norm": "",
        },
        {
            "entity_id": "S2-005",
            "raw_name": "Apex Motors Ltd",
            "name_expanded": "Apex Motors Limited",
            "name_core": "Apex Motors",
            "raw_address": "12 High St London",
            "address_expanded": "12 High Street London",
            "house_number": "12",
            "postal_prefix": "EC1A",
            "landmark": "",
            "country_norm": "GB",
        },
    ]

    # 2. Candidate pairs (Schema 3)
    pairs = [
        # S1-001 multi-match group (3 true matches, 1 decoy)
        {
            "source1_entity_id": "S1-001",
            "candidate_entity_id": "S2-001",
            "blocking_strategy_flags": "name_token,snm,char_tfidf",
            "blocking_score": 0.98,
            "label": "TRUE_MATCH (Top-1)",
        },
        {
            "source1_entity_id": "S1-001",
            "candidate_entity_id": "S2-002",
            "blocking_strategy_flags": "name_token,char_tfidf",
            "blocking_score": 0.88,
            "label": "TRUE_MATCH (Top-2)",
        },
        {
            "source1_entity_id": "S1-001",
            "candidate_entity_id": "S3-001",
            "blocking_strategy_flags": "name_token,word_tfidf",
            "blocking_score": 0.82,
            "label": "TRUE_MATCH (Top-3)",
        },
        {
            "source1_entity_id": "S1-001",
            "candidate_entity_id": "S3-002",
            "blocking_strategy_flags": "addr_token",
            "blocking_score": 0.45,
            "label": "FALSE_MATCH (Decoy/Address overlap)",
        },
        # S1-002: True match
        {
            "source1_entity_id": "S1-002",
            "candidate_entity_id": "S2-003",
            "blocking_strategy_flags": "name_token,snm,char_tfidf,word_tfidf",
            "blocking_score": 0.95,
            "label": "TRUE_MATCH (Legal suffix variant)",
        },
        # S1-003: Hard negative
        {
            "source1_entity_id": "S1-003",
            "candidate_entity_id": "S2-004",
            "blocking_strategy_flags": "name_token",
            "blocking_score": 0.70,
            "label": "FALSE_MATCH (Lookalike name, diff city)",
        },
        # S1-004: Landmark true match
        {
            "source1_entity_id": "S1-004",
            "candidate_entity_id": "S3-003",
            "blocking_strategy_flags": "name_token,minhash",
            "blocking_score": 0.91,
            "label": "TRUE_MATCH (Landmark addressing)",
        },
        # S1-005: Missing fields
        {
            "source1_entity_id": "S1-005",
            "candidate_entity_id": "S2-005",
            "blocking_strategy_flags": "char_tfidf",
            "blocking_score": 0.30,
            "label": "FALSE_MATCH (Empty S1 address/country)",
        },
    ]

    df_norm = pd.DataFrame(norm_records)
    df_pairs = pd.DataFrame(pairs)
    return df_pairs, df_norm, pairs


def _run_validation() -> None:
    """Run full validation of build_feature_table on mock data."""
    print("=" * 75)
    print("  features.py — Mock Dataset Validation Suite (Phase B2)")
    print("=" * 75)

    df_pairs, df_norm, pair_meta = _create_mock_dataset()

    print(f"\n1. Building feature table for {len(df_pairs)} mock candidate pairs...")
    feat_df = build_feature_table(df_pairs, df_norm)

    print(f"   Shape: {feat_df.shape}")
    print(f"   Columns ({len(feat_df.columns)}): {list(feat_df.columns)}")

    # Check 1: Column presence and ordering
    assert list(feat_df.columns) == list(FEATURE_TABLE_COLUMNS), (
        f"Columns mismatch!\nExpected: {list(FEATURE_TABLE_COLUMNS)}\nGot: {list(feat_df.columns)}"
    )
    print("   [PASS] Exact schema column match.")

    # Check 2: Zero NaNs
    nan_counts = feat_df.isna().sum().sum()
    assert nan_counts == 0, f"Found {nan_counts} NaNs in feature table!"
    print(f"   [PASS] Zero NaNs across all columns.")

    # Print feature inspection for every pair
    print("\n" + "=" * 75)
    print("  2. Feature Inspection by Pair")
    print("=" * 75)

    for idx, row in feat_df.iterrows():
        meta = pair_meta[idx]
        s1_id = row[FEAT_COL_SOURCE1_ENTITY_ID]
        c_id = row[FEAT_COL_CANDIDATE_ENTITY_ID]
        label = meta["label"]

        print(f"\nPair #{idx+1}: {s1_id} -> {c_id}  [{label}]")
        print("-" * 65)
        print(f"  Name features:")
        print(f"    lev_dist_raw          : {row[FEAT_COL_LEV_DIST_RAW]:.4f}")
        print(f"    jaro_winkler_raw      : {row[FEAT_COL_JARO_WINKLER_RAW]:.4f}")
        print(f"    token_sort_ratio_norm : {row[FEAT_COL_TOKEN_SORT_RATIO_NORM]:.4f}")
        print(f"    token_set_ratio_norm  : {row[FEAT_COL_TOKEN_SET_RATIO_NORM]:.4f}")
        print(f"    tfidf_cosine_char     : {row[FEAT_COL_TFIDF_COSINE_CHAR]:.4f}")
        print(f"    tfidf_cosine_word     : {row[FEAT_COL_TFIDF_COSINE_WORD]:.4f}")
        print(f"    jaccard_tokens_full   : {row[FEAT_COL_JACCARD_TOKENS_FULL]:.4f}")
        print(f"    jaccard_tokens_core   : {row[FEAT_COL_JACCARD_TOKENS_CORE]:.4f}")
        print(f"    first_token_match     : {row[FEAT_COL_FIRST_TOKEN_MATCH]:.1f}")
        print(f"    name_len_diff_ratio   : {row[FEAT_COL_NAME_LEN_DIFF_RATIO]:.4f}")
        print(f"  Address features:")
        print(f"    addr_lev_dist         : {row[FEAT_COL_ADDR_LEV_DIST]:.4f}")
        print(f"    addr_jaro_winkler     : {row[FEAT_COL_ADDR_JARO_WINKLER]:.4f}")
        print(f"    addr_token_sort_ratio : {row[FEAT_COL_ADDR_TOKEN_SORT_RATIO]:.4f}")
        print(f"    addr_token_set_ratio  : {row[FEAT_COL_ADDR_TOKEN_SET_RATIO]:.4f}")
        print(f"    addr_jaccard          : {row[FEAT_COL_ADDR_JACCARD]:.4f}")
        print(f"    house_number_match    : {row[FEAT_COL_HOUSE_NUMBER_MATCH]:.1f}")
        print(f"    postal_prefix_match   : {row[FEAT_COL_POSTAL_PREFIX_MATCH]:.1f}")
        print(f"    city_token_overlap    : {row[FEAT_COL_CITY_TOKEN_OVERLAP]:.4f}")
        print(f"    landmark_flag         : {row[FEAT_COL_LANDMARK_FLAG]:.1f}")
        print(f"  Meta & Provenance features:")
        print(f"    same_country_flag     : {row[FEAT_COL_SAME_COUNTRY_FLAG]:.1f}")
        print(f"    candidate_score_gap   : {row[FEAT_COL_CANDIDATE_SCORE_GAP]:.4f}")
        print(
            f"    provenance flags      : name_token={row[FEAT_COL_BLOCKED_BY_NAME_TOKEN]:.0f}, "
            f"snm={row[FEAT_COL_BLOCKED_BY_SNM]:.0f}, char_tfidf={row[FEAT_COL_BLOCKED_BY_CHAR_TFIDF]:.0f}, "
            f"word_tfidf={row[FEAT_COL_BLOCKED_BY_WORD_TFIDF]:.0f}, addr_token={row[FEAT_COL_BLOCKED_BY_ADDR_TOKEN]:.0f}, "
            f"minhash={row[FEAT_COL_BLOCKED_BY_MINHASH]:.0f}"
        )

    # Separation Sanity Check
    print("\n" + "=" * 75)
    print("  3. Separation Sanity Check (True Match vs Hard Negative)")
    print("=" * 75)

    # True Match 1 vs Decoy for S1-001
    tm1 = feat_df.iloc[0]
    decoy = feat_df.iloc[3]
    print(f"Multi-Match Group S1-001:")
    print(f"  True Match (S2-001)  -> tfidf_cosine_char: {tm1[FEAT_COL_TFIDF_COSINE_CHAR]:.4f}, token_sort: {tm1[FEAT_COL_TOKEN_SORT_RATIO_NORM]:.4f}, score_gap: {tm1[FEAT_COL_CANDIDATE_SCORE_GAP]:.4f}")
    print(f"  Decoy Match (S3-002) -> tfidf_cosine_char: {decoy[FEAT_COL_TFIDF_COSINE_CHAR]:.4f}, token_sort: {decoy[FEAT_COL_TOKEN_SORT_RATIO_NORM]:.4f}, score_gap: {decoy[FEAT_COL_CANDIDATE_SCORE_GAP]:.4f}")
    assert tm1[FEAT_COL_TFIDF_COSINE_CHAR] > decoy[FEAT_COL_TFIDF_COSINE_CHAR], "Separation failure on tfidf_cosine_char!"
    assert tm1[FEAT_COL_TOKEN_SORT_RATIO_NORM] > decoy[FEAT_COL_TOKEN_SORT_RATIO_NORM], "Separation failure on token_sort!"
    assert decoy[FEAT_COL_CANDIDATE_SCORE_GAP] > tm1[FEAT_COL_CANDIDATE_SCORE_GAP], "Score gap failure!"
    print("  [PASS] True match vs Decoy clearly separated.")

    # True Match S1-002 vs Hard Negative S1-003
    tm2 = feat_df.iloc[4]
    hn = feat_df.iloc[5]
    print(f"\nLocation Mismatch Check:")
    print(f"  True Match (S1-002 -> S2-003) -> addr_token_sort: {tm2[FEAT_COL_ADDR_TOKEN_SORT_RATIO]:.4f}, city_overlap: {tm2[FEAT_COL_CITY_TOKEN_OVERLAP]:.4f}, house_num_match: {tm2[FEAT_COL_HOUSE_NUMBER_MATCH]:.1f}")
    print(f"  Lookalike (S1-003 -> S2-004)  -> addr_token_sort: {hn[FEAT_COL_ADDR_TOKEN_SORT_RATIO]:.4f}, city_overlap: {hn[FEAT_COL_CITY_TOKEN_OVERLAP]:.4f}, house_num_match: {hn[FEAT_COL_HOUSE_NUMBER_MATCH]:.1f}")
    assert tm2[FEAT_COL_ADDR_TOKEN_SORT_RATIO] > hn[FEAT_COL_ADDR_TOKEN_SORT_RATIO], "Separation failure on addr_token_sort!"
    assert tm2[FEAT_COL_CITY_TOKEN_OVERLAP] > hn[FEAT_COL_CITY_TOKEN_OVERLAP], "Separation failure on city_token_overlap!"
    print("  [PASS] True match vs Lookalike location clearly separated.")

    # Landmark Check S1-004
    lm_pair = feat_df.iloc[6]
    print(f"\nLandmark Address Check:")
    print(f"  Landmark Pair (S1-004 -> S3-003) -> landmark_flag: {lm_pair[FEAT_COL_LANDMARK_FLAG]:.1f}, house_num_match: {lm_pair[FEAT_COL_HOUSE_NUMBER_MATCH]:.1f}")
    assert lm_pair[FEAT_COL_LANDMARK_FLAG] == 1.0, "Landmark flag should be 1.0!"
    assert lm_pair[FEAT_COL_HOUSE_NUMBER_MATCH] == 0.0, "Both-empty house numbers should be 0.0!"
    print("  [PASS] Landmark flag active and both-empty house numbers correctly handled as 0.0.")

    # Null Safety Check S1-005
    null_pair = feat_df.iloc[7]
    print(f"\nNull Safety Check:")
    print(f"  Degraded Pair (S1-005 -> S2-005) -> same_country_flag: {null_pair[FEAT_COL_SAME_COUNTRY_FLAG]:.1f}, addr_jaccard: {null_pair[FEAT_COL_ADDR_JACCARD]:.4f}")
    assert null_pair[FEAT_COL_SAME_COUNTRY_FLAG] == 0.0, "Missing country should yield 0.0!"
    assert null_pair[FEAT_COL_ADDR_JACCARD] == 0.0, "Empty address should yield 0.0!"
    print("  [PASS] Null/missing fields safely evaluated without NaNs.")

    print("\n" + "=" * 75)
    print("  ALL VALIDATION CHECKS PASSED")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    _run_validation()
