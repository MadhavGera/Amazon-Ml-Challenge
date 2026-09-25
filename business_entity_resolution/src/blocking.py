"""
blocking.py — Candidate pair generation (blocking / indexing stage).

Phase 3 implementation target.

This module consumes Normalized record DataFrames (Schema 2) and produces the
Candidate pairs schema (Schema 3).

Strategy catalogue
------------------
1. name_token   — Inverted-index on tokenised name_expanded; retrieve pairs
                  sharing at least one significant token.
2. snm          — Sorted-Neighbourhood Method on a composite sort key
                  (name_core + postal_prefix).  Window size = SNM_WINDOW_SIZE.
3. char_tfidf   — TF-IDF (char n-grams, TFIDF_NGRAM_RANGE_CHAR) + approximate
                  nearest-neighbours (sklearn NearestNeighbors or similar).
4. word_tfidf   — TF-IDF (word n-grams, TFIDF_NGRAM_RANGE_WORD) + kNN.
5. addr_token   — Inverted-index on tokenised address_expanded.
6. minhash      — MinHash / LSH (datasketch) fallback for large candidate sets
                  where candidate_set_size >= CANDIDATE_SIZE_THRESHOLD.

The gating logic between strategies 3/4 and 6 is:
    if len(corpus) < CANDIDATE_SIZE_THRESHOLD:
        use TF-IDF + kNN (strategies 3 & 4)
    else:
        use MinHash / LSH (strategy 6)

All strategies emit (source1_entity_id, candidate_entity_id) pairs.  The
combiner merges them, sets blocking_strategy_flags, and computes blocking_score.
"""

from __future__ import annotations

import pandas as pd


def build_candidate_pairs(
    source1_norm: pd.DataFrame,
    source2_norm: pd.DataFrame,
    source3_norm: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Run all configured blocking strategies and return the merged candidate-pair
    DataFrame (Schema 3).

    Parameters
    ----------
    source1_norm : pd.DataFrame
        Normalised records from source1.
    source2_norm : pd.DataFrame
        Normalised records from source2.
    source3_norm : pd.DataFrame | None
        Normalised records from source3, if available.

    Returns
    -------
    pd.DataFrame
        Columns: source1_entity_id, candidate_entity_id,
                 blocking_strategy_flags, blocking_score.

    Notes
    -----
    Implemented in Phase 3.
    """
    raise NotImplementedError("build_candidate_pairs: implement in Phase 3")


def name_token_block(
    source1_norm: pd.DataFrame,
    candidates_norm: pd.DataFrame,
) -> pd.DataFrame:
    """
    Inverted-index blocking on tokenised name_expanded.

    Returns pairs sharing at least one significant (non-stopword) name token.

    Notes
    -----
    Implemented in Phase 3.
    """
    raise NotImplementedError("name_token_block: implement in Phase 3")


def snm_block(
    source1_norm: pd.DataFrame,
    candidates_norm: pd.DataFrame,
    window_size: int,
) -> pd.DataFrame:
    """
    Sorted-Neighbourhood Method blocking.

    Sorts both DataFrames on (name_core, postal_prefix), then slides a window
    of size *window_size* and emits all cross-source pairs within each window.

    Notes
    -----
    Implemented in Phase 3.
    """
    raise NotImplementedError("snm_block: implement in Phase 3")


def tfidf_knn_block(
    source1_norm: pd.DataFrame,
    candidates_norm: pd.DataFrame,
    ngram_range: tuple[int, int],
    analyzer: str,
    top_k: int,
) -> pd.DataFrame:
    """
    TF-IDF vectorisation + approximate k-nearest-neighbours blocking.

    *analyzer* is either "char" or "word".
    Uses sklearn TfidfVectorizer + NearestNeighbors (or equivalent).

    Notes
    -----
    Implemented in Phase 3.
    """
    raise NotImplementedError("tfidf_knn_block: implement in Phase 3")


def addr_token_block(
    source1_norm: pd.DataFrame,
    candidates_norm: pd.DataFrame,
) -> pd.DataFrame:
    """
    Inverted-index blocking on tokenised address_expanded.

    Notes
    -----
    Implemented in Phase 3.
    """
    raise NotImplementedError("addr_token_block: implement in Phase 3")


def minhash_lsh_block(
    source1_norm: pd.DataFrame,
    candidates_norm: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """
    MinHash / LSH blocking (datasketch) for large corpora.

    Used when len(candidates_norm) >= CANDIDATE_SIZE_THRESHOLD.

    Notes
    -----
    Implemented in Phase 3.
    """
    raise NotImplementedError("minhash_lsh_block: implement in Phase 3")


def merge_strategy_outputs(
    strategy_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Union all strategy-specific pair DataFrames, deduplicate, set
    blocking_strategy_flags (comma-joined strategy names), and compute
    blocking_score (max across contributing strategies per pair).

    Parameters
    ----------
    strategy_dfs : dict[str, pd.DataFrame]
        Mapping strategy_name → pairs DataFrame with at minimum columns
        (source1_entity_id, candidate_entity_id, raw_score).

    Returns
    -------
    pd.DataFrame
        Schema 3: (source1_entity_id, candidate_entity_id,
                   blocking_strategy_flags, blocking_score).

    Notes
    -----
    Implemented in Phase 3.
    """
    raise NotImplementedError("merge_strategy_outputs: implement in Phase 3")


def evaluate_blocking(
    candidate_pairs: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> dict[str, float]:
    """
    Compute blocking recall and pair-completeness metrics.

    Recall = fraction of true match pairs that appear in candidate_pairs.

    Parameters
    ----------
    candidate_pairs : pd.DataFrame
        Schema 3 output from build_candidate_pairs.
    ground_truth : pd.DataFrame
        Schema 1b ground-truth file.

    Returns
    -------
    dict[str, float]
        Keys: "recall", "reduction_ratio", "pair_completeness".

    Notes
    -----
    Implemented in Phase 3.
    """
    raise NotImplementedError("evaluate_blocking: implement in Phase 3")
