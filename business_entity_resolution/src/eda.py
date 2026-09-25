"""
eda.py — Exploratory data analysis utilities.

Phase 1 / Phase 2 implementation target (run early, results inform blocking design).

Planned analyses
----------------
1. Field-level completeness — % missing / empty per column across all three sources.
2. Name length distribution — histogram of raw_name token counts.
3. Address length distribution — similar for raw_address.
4. Country distribution — value counts + normalisation sanity check.
5. Token frequency — top-N name tokens (to inform stopword list).
6. Cross-source overlap heuristics — estimate how many pairs share at least one
   name token (gives blocking recall upper bound without feature engineering).
7. Ground-truth match-ratio distribution — how many matches does each source1
   entity have?  Are there 1-to-many or many-to-many cases?
8. Duplicate / near-duplicate detection within a single source — flag for cleaning.
"""

from __future__ import annotations

import pandas as pd


def field_completeness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame reporting, per column: total rows, non-empty count,
    empty count, and % complete.

    Notes
    -----
    Implemented in Phase 1.
    """
    raise NotImplementedError("field_completeness: implement in Phase 1")


def name_length_distribution(df: pd.DataFrame) -> pd.Series:
    """
    Return a Series of value counts of token count per record in raw_name.

    Notes
    -----
    Implemented in Phase 1.
    """
    raise NotImplementedError("name_length_distribution: implement in Phase 1")


def top_name_tokens(df: pd.DataFrame, top_n: int = 50) -> pd.Series:
    """
    Return the *top_n* most frequent tokens across all raw_name values.
    Useful for building the stopword / legal-suffix list.

    Notes
    -----
    Implemented in Phase 1.
    """
    raise NotImplementedError("top_name_tokens: implement in Phase 1")


def country_distribution(df: pd.DataFrame) -> pd.Series:
    """
    Return value counts of the raw `country` field.

    Notes
    -----
    Implemented in Phase 1.
    """
    raise NotImplementedError("country_distribution: implement in Phase 1")


def cross_source_token_overlap(
    source1_norm: pd.DataFrame,
    candidates_norm: pd.DataFrame,
) -> float:
    """
    Estimate the fraction of source1 entities that share at least one
    significant name token with at least one candidate entity.

    This is an upper-bound estimate of the name-token blocking recall.

    Notes
    -----
    Implemented in Phase 2.
    """
    raise NotImplementedError("cross_source_token_overlap: implement in Phase 2")


def ground_truth_match_ratio(ground_truth: pd.DataFrame) -> pd.Series:
    """
    Return a Series of value counts of the number of matched entities per
    source1_entity_id in the ground-truth file.

    Notes
    -----
    Implemented in Phase 2.
    """
    raise NotImplementedError("ground_truth_match_ratio: implement in Phase 2")
