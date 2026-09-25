"""
features.py — Feature engineering for entity-pair classification.

Phase 4 implementation target.

This module consumes:
  - The Candidate pairs DataFrame (Schema 3, from blocking.py)
  - Both Normalized record DataFrames (Schema 2, from normalize.py)

And produces the Feature table (Schema 4), which is the direct input to
train.py and predict.py.

All feature column names are defined in config.FEATURE_TABLE_COLUMNS.
Feature functions must return values in the dtypes documented in docs/schemas.md.

Feature groups
--------------
1. Name similarity   — Levenshtein, Jaro-Winkler, token ratios, TF-IDF cosine,
                       Jaccard, first-token match, length-diff ratio.
2. Address similarity — Same set of metrics applied to address fields, plus
                        structured sub-field comparisons (house_number,
                        postal_prefix, city token overlap, landmark flag).
3. Meta / structural — same_country_flag, candidate_score_gap.
4. Blocking provenance — Binary flags derived from blocking_strategy_flags.
"""

from __future__ import annotations

import pandas as pd


def build_feature_table(
    candidate_pairs: pd.DataFrame,
    source1_norm: pd.DataFrame,
    candidates_norm: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join normalised records onto candidate pairs and compute all features.

    Parameters
    ----------
    candidate_pairs : pd.DataFrame
        Schema 3 output from blocking.build_candidate_pairs.
    source1_norm : pd.DataFrame
        Normalised source1 records (Schema 2).
    candidates_norm : pd.DataFrame
        Normalised source2/source3 records (Schema 2).

    Returns
    -------
    pd.DataFrame
        Schema 4 feature table with columns config.FEATURE_TABLE_COLUMNS.

    Notes
    -----
    Implemented in Phase 4.
    """
    raise NotImplementedError("build_feature_table: implement in Phase 4")


# ── Name similarity ───────────────────────────────────────────────────────────

def compute_name_features(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all name-similarity features for the pair DataFrame.

    Expects columns: raw_name_s1, name_expanded_s1, name_core_s1,
                     raw_name_c,  name_expanded_c,  name_core_c.

    Returns DataFrame with columns:
        lev_dist_raw, jaro_winkler_raw, token_sort_ratio_norm,
        token_set_ratio_norm, tfidf_cosine_char, tfidf_cosine_word,
        jaccard_tokens_full, jaccard_tokens_core,
        first_token_match, name_len_diff_ratio.

    Notes
    -----
    Implemented in Phase 4.
    """
    raise NotImplementedError("compute_name_features: implement in Phase 4")


# ── Address similarity ────────────────────────────────────────────────────────

def compute_address_features(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all address-similarity features for the pair DataFrame.

    Expects columns: address_expanded_s1, house_number_s1, postal_prefix_s1,
                     address_expanded_c,  house_number_c,  postal_prefix_c,
                     landmark_s1, landmark_c.

    Returns DataFrame with columns:
        addr_lev_dist, addr_jaro_winkler, addr_token_sort_ratio,
        addr_token_set_ratio, addr_jaccard, house_number_match,
        postal_prefix_match, city_token_overlap, landmark_flag.

    Notes
    -----
    Implemented in Phase 4.
    """
    raise NotImplementedError("compute_address_features: implement in Phase 4")


# ── Meta / structural features ────────────────────────────────────────────────

def compute_meta_features(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Compute same_country_flag and candidate_score_gap.

    candidate_score_gap = top-1 blocking_score for this source1_entity_id
                          minus this pair's blocking_score.

    Notes
    -----
    Implemented in Phase 4.
    """
    raise NotImplementedError("compute_meta_features: implement in Phase 4")


# ── Blocking provenance features ──────────────────────────────────────────────

def unpack_blocking_flags(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the comma-separated blocking_strategy_flags column into binary
    indicator columns (blocked_by_name_token, blocked_by_snm, …).

    Notes
    -----
    Implemented in Phase 4.
    """
    raise NotImplementedError("unpack_blocking_flags: implement in Phase 4")
