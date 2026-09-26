"""
metrics.py - Official evaluation metric for the business entity resolution pipeline.

Implements the exact macro-averaged F0.5 scoring described in the problem statement:

    F_0.5 = (1.25 x Precision x Recall) / (0.25 x Precision + Recall)

computed PER Source-1 entity, then macro-averaged across all S1 entities.

Per-entity edge-case rules
--------------------------
  GT empty  & PRED empty     -> F0.5 = 1.0  (correct singleton)
  GT empty  & PRED non-empty -> F0.5 = 0.0  (false merge on a true singleton)
  GT non-empty & PRED empty  -> F0.5 = 0.0  (missed match; precision undefined)
  Otherwise:
      precision = |pred & gt| / |pred|
      recall    = |pred & gt| / |gt|
      F0.5 = 0.0 if both are 0, else (1.25 x P x R) / (0.25 x P + R)

Usage
-----
  from src.metrics import compute_macro_f05, load_predictions_tsv, load_ground_truth_tsv

  gt   = load_ground_truth_tsv(TRAIN_GROUND_TRUTH_PATH)
  pred = load_predictions_tsv(OUTPUT_MATCHING_RESULTS_PATH)
  result = compute_macro_f05(pred, gt)
  print(result["macro_f05"])

Run standalone tests
--------------------
  python -m src.metrics
  python src/metrics.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

# Ensure src/ is importable regardless of invocation directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    GT_COL_MATCHED_ENTITY_IDS,
    GT_COL_SOURCE1_ENTITY_ID,
)


# ---------------------------------------------------------------------------
# CORE METRIC
# ---------------------------------------------------------------------------

def _f05_for_entity(pred: set, gt: set) -> float:
    """
    Compute F0.5 for a single Source-1 entity.

    Parameters
    ----------
    pred : set[str]
        Predicted matched entity IDs.
    gt : set[str]
        Ground-truth matched entity IDs.

    Returns
    -------
    float : F0.5 in [0.0, 1.0]
    """
    gt_empty   = len(gt)   == 0
    pred_empty = len(pred) == 0

    if gt_empty and pred_empty:
        return 1.0        # correct singleton
    if gt_empty and not pred_empty:
        return 0.0        # false merge on true singleton
    if not gt_empty and pred_empty:
        return 0.0        # missed match; precision undefined -> 0

    tp        = len(pred & gt)
    precision = tp / len(pred)
    recall    = tp / len(gt)

    denom = 0.25 * precision + recall
    if denom == 0.0:
        return 0.0
    return (1.25 * precision * recall) / denom


def compute_macro_f05(
    predictions:  dict,
    ground_truth: dict,
) -> dict:
    """
    Compute macro-averaged F0.5 across all Source-1 entities.

    Every id present in ground_truth is scored.  If a Source-1 id is absent
    from predictions it is treated as an empty prediction (not an error).

    Parameters
    ----------
    predictions : dict[str, set[str]]
        Maps source1_entity_id -> set of predicted matched_entity_ids.
    ground_truth : dict[str, set[str]]
        Maps source1_entity_id -> set of true matched_entity_ids.

    Returns
    -------
    dict with keys:
        macro_f05          float  - primary challenge metric
        per_entity_scores  dict   - per-entity F0.5 values
        micro_precision    float  - TP / (TP + FP) across all entities
        micro_recall       float  - TP / (TP + FN) across all entities
        singleton_count    int    - number of true singletons (gt empty)
        singleton_accuracy float  - fraction of true singletons predicted empty
    """
    if not ground_truth:
        raise ValueError("ground_truth must not be empty.")

    per_entity_scores = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0
    singleton_count   = 0
    singleton_correct = 0

    for s1_id, gt in ground_truth.items():
        pred = predictions.get(s1_id, set())

        if len(gt) == 0:
            singleton_count += 1
            if len(pred) == 0:
                singleton_correct += 1

        per_entity_scores[s1_id] = _f05_for_entity(pred, gt)

        if len(gt) > 0 or len(pred) > 0:
            tp = len(pred & gt)
            total_tp += tp
            total_fp += len(pred) - tp
            total_fn += len(gt)  - tp

    macro_f05 = (
        sum(per_entity_scores.values()) / len(per_entity_scores)
        if per_entity_scores else 0.0
    )
    micro_precision = (
        total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    )
    micro_recall = (
        total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    )
    singleton_accuracy = (
        singleton_correct / singleton_count if singleton_count > 0 else 0.0
    )

    return {
        "macro_f05":          macro_f05,
        "per_entity_scores":  per_entity_scores,
        "micro_precision":    micro_precision,
        "micro_recall":       micro_recall,
        "singleton_count":    singleton_count,
        "singleton_accuracy": singleton_accuracy,
    }


# ---------------------------------------------------------------------------
# TSV I/O HELPERS
# ---------------------------------------------------------------------------

def _parse_id_list(cell: str) -> set:
    """
    Parse a comma- or pipe-separated cell into a set of entity IDs.
    Empty / whitespace-only cells return an empty set (singleton).
    """
    cell = cell.strip()
    if not cell:
        return set()
    delimiter = "," if "," in cell else "|"
    return {tok.strip() for tok in cell.split(delimiter) if tok.strip()}


def load_predictions_tsv(path) -> dict:
    """
    Load a predictions TSV (matching_results.tsv format) into a dict-of-sets.

    Expected columns: source1_entity_id, matched_entity_ids
    Empty matched_entity_ids cells are treated as empty sets (singleton preds).

    Parameters
    ----------
    path : str | Path

    Returns
    -------
    dict[str, set[str]]

    Raises
    ------
    FileNotFoundError, ValueError
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")

    result = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None or GT_COL_SOURCE1_ENTITY_ID not in reader.fieldnames:
            raise ValueError(
                f"Predictions file missing column '{GT_COL_SOURCE1_ENTITY_ID}': {path}"
            )
        for row in reader:
            s1_id      = row[GT_COL_SOURCE1_ENTITY_ID].strip()
            raw_ids    = row.get(GT_COL_MATCHED_ENTITY_IDS, "")
            result[s1_id] = _parse_id_list(raw_ids)
    return result


def load_ground_truth_tsv(path) -> dict:
    """
    Load train_ground_truth.tsv into a dict-of-sets.

    Expected columns: source1_entity_id, matched_entity_ids
    Empty matched_entity_ids cells -> true singletons (empty set).

    Parameters
    ----------
    path : str | Path

    Returns
    -------
    dict[str, set[str]]

    Raises
    ------
    FileNotFoundError, ValueError
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {path}")

    result = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None or GT_COL_SOURCE1_ENTITY_ID not in reader.fieldnames:
            raise ValueError(
                f"Ground-truth file missing column '{GT_COL_SOURCE1_ENTITY_ID}': {path}"
            )
        for row in reader:
            s1_id   = row[GT_COL_SOURCE1_ENTITY_ID].strip()
            raw_ids = row.get(GT_COL_MATCHED_ENTITY_IDS, "")
            result[s1_id] = _parse_id_list(raw_ids)
    return result


# ---------------------------------------------------------------------------
# LEGACY / PHASE-5 STUBS (kept from original scaffold)
# ---------------------------------------------------------------------------

def pair_precision_recall_f1(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> dict:
    """
    Compute pair-level precision, recall, and F1 score.

    Parameters
    ----------
    predictions : pd.DataFrame
        Columns: source1_entity_id, candidate_entity_id, is_match (bool/int).
    ground_truth : pd.DataFrame
        Schema 1b ground-truth DataFrame.

    Notes
    -----
    Implemented in Phase 5.
    """
    raise NotImplementedError("pair_precision_recall_f1: implement in Phase 5")


def entity_level_f1(
    matching_results: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> dict:
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
) -> dict:
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


# ---------------------------------------------------------------------------
# STANDALONE UNIT TESTS
# ---------------------------------------------------------------------------

def _run_tests() -> None:
    """
    Standalone test suite for compute_macro_f05 / _f05_for_entity.
    Prints PASS/FAIL for each case and exits with code 1 on failure.
    """
    TOL = 0.001
    all_passed = True

    def _check(label: str, got: float, expected: float, tol: float = TOL) -> bool:
        ok = abs(got - expected) <= tol
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}]  {label}")
        print(f"           got={got:.6f}  expected={expected:.6f}  tol={tol}")
        return ok

    print("=" * 65)
    print("  metrics.py - Unit Tests")
    print("=" * 65)

    # ── Test 1: Worked example from the problem statement ──────────────────
    print("\nTest 1 - Worked example (S1-00001)")
    pred_1 = {"S1-00001": {"S2-00047", "S2-00193", "S3-00812"}}
    gt_1   = {"S1-00001": {"S2-00047", "S3-00812"}}
    #
    # TP = {S2-00047, S3-00812} = 2
    # Precision = 2/3 ~ 0.667
    # Recall    = 2/2 = 1.000
    # F0.5 = (1.25 * 2/3 * 1.0) / (0.25 * 2/3 + 1.0)
    #       = 0.8333 / 1.1667 ~ 0.7143
    result_1 = compute_macro_f05(pred_1, gt_1)
    f05_1    = result_1["per_entity_scores"]["S1-00001"]
    macro_1  = result_1["macro_f05"]

    exp_p    = 2 / 3
    exp_r    = 1.0
    exp_f05  = (1.25 * exp_p * exp_r) / (0.25 * exp_p + exp_r)

    print(f"  Precision (expected): {exp_p:.6f}")
    print(f"  Recall    (expected): {exp_r:.6f}")
    print(f"  F0.5      (expected): {exp_f05:.6f}")
    ok1a = _check("per-entity F0.5 ~ 0.714", f05_1,   exp_f05)
    ok1b = _check("macro_f05 == per_entity (single)",  macro_1, exp_f05)
    print(f"  micro_precision : {result_1['micro_precision']:.6f}")
    print(f"  micro_recall    : {result_1['micro_recall']:.6f}")
    print(f"  singleton_count : {result_1['singleton_count']}")
    if not (ok1a and ok1b):
        all_passed = False

    # ── Test 2: Correct singleton ──────────────────────────────────────────
    print("\nTest 2 - Correct singleton (GT={}, PRED={})")
    pred_2 = {"S1-00002": set()}
    gt_2   = {"S1-00002": set()}
    result_2 = compute_macro_f05(pred_2, gt_2)
    ok2  = _check("F0.5 == 1.0 (correct singleton)",  result_2["macro_f05"],         1.0)
    ok2b = _check("singleton_count == 1",              float(result_2["singleton_count"]), 1.0)
    ok2c = _check("singleton_accuracy == 1.0",         result_2["singleton_accuracy"], 1.0)
    if not (ok2 and ok2b and ok2c):
        all_passed = False

    # ── Test 3: False merge on true singleton ─────────────────────────────
    print("\nTest 3 - False merge on true singleton (GT={}, PRED={S2-00099})")
    pred_3 = {"S1-00003": {"S2-00099"}}
    gt_3   = {"S1-00003": set()}
    result_3 = compute_macro_f05(pred_3, gt_3)
    ok3  = _check("F0.5 == 0.0 (false merge)",        result_3["macro_f05"],         0.0)
    ok3b = _check("singleton_count == 1",              float(result_3["singleton_count"]), 1.0)
    ok3c = _check("singleton_accuracy == 0.0",         result_3["singleton_accuracy"], 0.0)
    if not (ok3 and ok3b and ok3c):
        all_passed = False

    # ── Test 4: Missed match ───────────────────────────────────────────────
    print("\nTest 4 - Missed match (GT={S2-00100}, PRED={})")
    pred_4 = {"S1-00004": set()}
    gt_4   = {"S1-00004": {"S2-00100"}}
    result_4 = compute_macro_f05(pred_4, gt_4)
    ok4 = _check("F0.5 == 0.0 (missed match)",        result_4["macro_f05"],         0.0)
    if not ok4:
        all_passed = False

    # ── Test 5: Multi-entity macro-average ────────────────────────────────
    print("\nTest 5 - Multi-entity macro-average")
    # S1-00001: F0.5 ~ 0.7143  (from Test 1)
    # S1-00002: F0.5 = 1.0     (correct singleton)
    # Macro = (0.7143 + 1.0) / 2 ~ 0.8571
    pred_5 = {
        "S1-00001": {"S2-00047", "S2-00193", "S3-00812"},
        "S1-00002": set(),
    }
    gt_5 = {
        "S1-00001": {"S2-00047", "S3-00812"},
        "S1-00002": set(),
    }
    result_5      = compute_macro_f05(pred_5, gt_5)
    expected_mac5 = (exp_f05 + 1.0) / 2.0
    ok5a = _check(f"macro_f05 ~ {expected_mac5:.4f}", result_5["macro_f05"],         expected_mac5)
    ok5b = _check("singleton_count == 1",              float(result_5["singleton_count"]), 1.0)
    ok5c = _check("singleton_accuracy == 1.0",         result_5["singleton_accuracy"], 1.0)
    if not (ok5a and ok5b and ok5c):
        all_passed = False

    # ── Test 6: S1 id absent from predictions ─────────────────────────────
    print("\nTest 6 - S1 id absent from predictions -> treated as empty prediction")
    pred_6 = {}
    gt_6   = {"S1-00004": {"S2-00100"}}
    result_6 = compute_macro_f05(pred_6, gt_6)
    ok6 = _check("F0.5 == 0.0 (absent id = empty pred)", result_6["macro_f05"], 0.0)
    if not ok6:
        all_passed = False

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    if all_passed:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 65)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    _run_tests()
