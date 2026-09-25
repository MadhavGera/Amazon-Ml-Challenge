"""
train.py — Model training pipeline.

Phase 5 implementation target.

Consumes the Feature table (Schema 4) with ground-truth labels and trains a
LightGBM binary classifier to predict whether a candidate pair is a true match.

Planned steps
-------------
1. Label generation — join Schema 4 pairs against train_ground_truth to produce
   binary labels (1 = match, 0 = non-match).
2. Train/validation split — stratified split, preserving source1_entity_id groups.
3. Class-imbalance handling — positive:negative ratio is typically 1:50 to 1:200;
   use scale_pos_weight or threshold tuning.
4. Model training — LightGBM LGBMClassifier with cross-validation.
5. Threshold search — on the validation set, find MATCH_THRESHOLD that maximises
   the challenge evaluation metric (F1 or similar).  Write result to config or
   a separate tuning artefact.
6. Model serialisation — save to OUTPUT_DIR/model.txt (LightGBM native format).
7. Feature importance logging — write to OUTPUT_DIR/feature_importance.csv.
"""

from __future__ import annotations

import pandas as pd


def make_training_labels(
    feature_table: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join the feature table against ground truth to add a binary `label` column.

    Parameters
    ----------
    feature_table : pd.DataFrame
        Schema 4 feature table.
    ground_truth : pd.DataFrame
        Schema 1b ground-truth DataFrame.

    Returns
    -------
    pd.DataFrame
        Feature table with an additional `label` column (int, 0 or 1).

    Notes
    -----
    Implemented in Phase 5.
    """
    raise NotImplementedError("make_training_labels: implement in Phase 5")


def train_model(
    feature_table: pd.DataFrame,
    ground_truth: pd.DataFrame,
    output_path: str | None = None,
) -> object:
    """
    Train a LightGBM classifier and serialise it to *output_path*.

    Parameters
    ----------
    feature_table : pd.DataFrame
        Labelled feature table (Schema 4 + label column).
    ground_truth : pd.DataFrame
        Ground-truth DataFrame (used for label generation and stratification).
    output_path : str | None
        Where to save the model (defaults to config.OUTPUT_MODEL_PATH).

    Returns
    -------
    object
        Trained LightGBM Booster instance.

    Notes
    -----
    Implemented in Phase 5.
    """
    raise NotImplementedError("train_model: implement in Phase 5")


def find_match_threshold(
    model: object,
    val_feature_table: pd.DataFrame,
    val_labels: pd.Series,
) -> float:
    """
    Search for the optimal MATCH_THRESHOLD on the validation set.

    Returns the threshold value that maximises the primary evaluation metric.

    Notes
    -----
    Implemented in Phase 5.
    """
    raise NotImplementedError("find_match_threshold: implement in Phase 5")
