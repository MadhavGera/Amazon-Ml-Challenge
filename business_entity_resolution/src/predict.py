"""
predict.py — Inference pipeline for generating submission files.

Phase 5/6 implementation target.

Consumes a trained model (LightGBM Booster) and the test Feature table to
produce the two submission TSVs:
  - output/matching_results.tsv   (source1_entity_id, matched_entity_ids)
  - output/candidate_pairs.tsv    (source1_entity_id, candidate_entity_ids)

Planned steps
-------------
1. Load the serialised model from OUTPUT_DIR/model.txt.
2. Load the test feature table (Schema 4, labels absent).
3. Score each pair: model.predict_proba → match probability.
4. Apply MATCH_THRESHOLD to produce binary predictions.
5. Optionally apply MARGIN_THRESHOLD (Phase 6) for confidence gating.
6. Aggregate per source1_entity_id → matched_entity_ids (pipe-separated).
7. Write both submission TSVs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_model(model_path: str | Path) -> object:
    """
    Load and return a serialised LightGBM Booster from *model_path*.

    Notes
    -----
    Implemented in Phase 5.
    """
    raise NotImplementedError("load_model: implement in Phase 5")


def predict_pairs(
    model: object,
    feature_table: pd.DataFrame,
    match_threshold: float | None = None,
    margin_threshold: float | None = None,
) -> pd.DataFrame:
    """
    Score candidate pairs and apply decision thresholds.

    Parameters
    ----------
    model : object
        Trained LightGBM Booster.
    feature_table : pd.DataFrame
        Schema 4 feature table (test split, no labels).
    match_threshold : float | None
        Probability threshold for positive prediction.  Uses config.MATCH_THRESHOLD
        if None.
    margin_threshold : float | None
        Margin-based confidence gate (Phase 6).  Uses config.MARGIN_THRESHOLD if None.

    Returns
    -------
    pd.DataFrame
        Columns: source1_entity_id, candidate_entity_id, match_prob, is_match (bool).

    Notes
    -----
    Implemented in Phase 5/6.
    """
    raise NotImplementedError("predict_pairs: implement in Phase 5")


def build_submission(
    predictions: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> None:
    """
    Aggregate predictions and write both submission TSVs to *output_dir*.

    Notes
    -----
    Implemented in Phase 5.
    """
    raise NotImplementedError("build_submission: implement in Phase 5")
