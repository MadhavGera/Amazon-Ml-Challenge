"""
metrics.py — Evaluation metrics for the entity resolution pipeline.

Phase 5 implementation target (used throughout for offline evaluation).

Metrics to implement
--------------------
- Pair-level precision, recall, F1 (binary classification on candidate pairs).
- Entity-level precision, recall, F1 (after aggregating predictions per source1 entity).
- Blocking recall / pair-completeness (used in Phase 3 to evaluate candidate generation).
- Reduction ratio (fraction of the O(n²) universe that blocking discards).
- Challenge-specific metric wrapper (to be confirmed once the challenge metric
  is finalised — assume F1 on the matching_results output for now).
"""

from __future__ import annotations

import pandas as pd


def pair_precision_recall_f1(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> dict[str, float]:
    """
    Compute pair-level precision, recall, and F1 score.

    Parameters
    ----------
    predictions : pd.DataFrame
        Columns: source1_entity_id, candidate_entity_id, is_match (bool/int).
    ground_truth : pd.DataFrame
        Schema 1b ground-truth DataFrame.

    Returns
    -------
    dict[str, float]
        Keys: "precision", "recall", "f1".

    Notes
    -----
    Implemented in Phase 5.
    """
    raise NotImplementedError("pair_precision_recall_f1: implement in Phase 5")


def entity_level_f1(
    matching_results: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> dict[str, float]:
    """
    Compute entity-level F1 after grouping predictions per source1_entity_id.

    Notes
    -----
    Implemented in Phase 5.
    """
    raise NotImplementedError("entity_level_f1: implement in Phase 5")


def blocking_recall(
    candidate_pairs: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> dict[str, float]:
    """
    Compute blocking recall (fraction of true pairs covered) and reduction ratio.

    Notes
    -----
    Implemented in Phase 3.
    """
    raise NotImplementedError("blocking_recall: implement in Phase 3")


def challenge_metric(
    matching_results: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> float:
    """
    Wrapper around the official challenge evaluation metric.

    Delegates to entity_level_f1 until the official metric is confirmed.

    Notes
    -----
    Implemented in Phase 5.
    """
    raise NotImplementedError("challenge_metric: implement in Phase 5")
