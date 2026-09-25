"""
error_analysis.py — Post-prediction error analysis utilities.

Phase 7 implementation target.

Planned analyses
----------------
1. False negatives (missed matches) — examine whether they cluster around:
   - low blocking_score (blocking failure)
   - high lev_dist_raw (genuinely hard name variation)
   - specific countries or address patterns
   - rare tokens in name_core
2. False positives (spurious matches) — examine:
   - common confusable name tokens (e.g. generic names like "City Hotel")
   - cases where address similarity pulled in wrong pairs
3. Threshold sensitivity — plot precision/recall curve as MATCH_THRESHOLD varies.
4. Feature importance cross-check — compare SHAP values for FP/FN subsets vs
   overall distribution.
5. Per-source breakdown — does model performance differ significantly across
   source1 vs source2 vs source3 origin of candidate?
6. Hard-negative mining — surface high-confidence FPs for manual inspection
   and potential training set augmentation.
"""

from __future__ import annotations

import pandas as pd


def false_negatives(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return the subset of true positive pairs that were predicted negative.

    Notes
    -----
    Implemented in Phase 7.
    """
    raise NotImplementedError("false_negatives: implement in Phase 7")


def false_positives(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return the subset of predicted positive pairs that are actually negative.

    Notes
    -----
    Implemented in Phase 7.
    """
    raise NotImplementedError("false_positives: implement in Phase 7")


def threshold_sensitivity(
    predictions_with_proba: pd.DataFrame,
    ground_truth: pd.DataFrame,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """
    Compute precision, recall, and F1 at each candidate threshold.

    Parameters
    ----------
    predictions_with_proba : pd.DataFrame
        Columns: source1_entity_id, candidate_entity_id, match_prob.
    ground_truth : pd.DataFrame
        Schema 1b ground-truth.
    thresholds : list[float] | None
        Thresholds to evaluate.  Defaults to np.linspace(0.05, 0.95, 37).

    Returns
    -------
    pd.DataFrame
        Columns: threshold, precision, recall, f1.

    Notes
    -----
    Implemented in Phase 7.
    """
    raise NotImplementedError("threshold_sensitivity: implement in Phase 7")


def hard_negatives(
    predictions_with_proba: pd.DataFrame,
    ground_truth: pd.DataFrame,
    top_n: int = 200,
) -> pd.DataFrame:
    """
    Return the *top_n* highest-confidence false positives for manual review.

    Notes
    -----
    Implemented in Phase 7.
    """
    raise NotImplementedError("hard_negatives: implement in Phase 7")


def per_source_breakdown(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute precision/recall/F1 broken down by the source of the candidate entity.

    Notes
    -----
    Implemented in Phase 7.
    """
    raise NotImplementedError("per_source_breakdown: implement in Phase 7")
