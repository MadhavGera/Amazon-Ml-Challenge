"""
predict.py — Inference pipeline, post-processing, and submission generation.

Person B track (Phase B4).

This module implements:
1. Multi-match threshold application (keep all candidates >= threshold, not top-1).
2. Relative margin check (confidence gating on ambiguous top candidates).
3. One-S1-per-record deduplication enforcement (validated safe in B1 EDA).
4. Singleton rate diagnostic monitoring.
5. Submission TSV writers (matching_results.tsv and candidate_pairs.tsv) with
   mandatory s1_order parameter ensuring zero-candidate S1 entities are never omitted.
6. Local submission validator mirroring challenge evaluation checks.

Usage
-----
    python src/predict.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    CAND_COL_CANDIDATE_ENTITY_ID,
    CAND_COL_SOURCE1_ENTITY_ID,
    FEATURE_COLUMNS,
    OUTPUT_CANDIDATE_PAIRS_PATH,
    OUTPUT_MATCHING_RESULTS_PATH,
    SUBMISSION_CANDIDATE_COLS,
    SUBMISSION_MATCHING_COLS,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. THRESHOLD APPLICATION (MULTI-MATCH SUPPORT)
# ─────────────────────────────────────────────────────────────────────────────

def apply_threshold(
    scored_candidates_df: pd.DataFrame,
    threshold: float = 0.65,
    prob_col: str = "prob",
    all_s1_ids: Sequence[str] | None = None,
) -> dict[str, set[str]]:
    """
    Apply probability threshold to scored candidate pairs.

    Keeps EVERY candidate at or above the threshold for each Source 1 entity,
    supporting 0, 1, or multiple matches per S1 entity.

    Parameters
    ----------
    scored_candidates_df : pd.DataFrame
        DataFrame with columns: source1_entity_id, candidate_entity_id, and prob_col.
    threshold : float
        Decision threshold (default: 0.65).
    prob_col : str
        Name of the probability column.
    all_s1_ids : Sequence[str] | None
        Optional complete list of S1 IDs to guarantee in the returned dictionary.

    Returns
    -------
    dict[str, set[str]]
        Mapping source1_entity_id -> set of predicted matched entity IDs.
    """
    predictions: dict[str, set[str]] = {}

    if all_s1_ids is not None:
        for s1_id in all_s1_ids:
            predictions[s1_id] = set()
    else:
        for s1_id in scored_candidates_df[CAND_COL_SOURCE1_ENTITY_ID].unique():
            predictions[s1_id] = set()

    if scored_candidates_df.empty:
        return predictions

    # Filter candidate pairs at or above threshold
    passed = scored_candidates_df[scored_candidates_df[prob_col] >= threshold]
    for _, row in passed.iterrows():
        s1 = row[CAND_COL_SOURCE1_ENTITY_ID]
        c = row[CAND_COL_CANDIDATE_ENTITY_ID]
        if s1 not in predictions:
            predictions[s1] = set()
        predictions[s1].add(c)

    return predictions


# ─────────────────────────────────────────────────────────────────────────────
# 2. RELATIVE MARGIN CHECK (CONSERVATIVE AMBIGUITY GATING)
# ─────────────────────────────────────────────────────────────────────────────

def apply_margin_check(
    scored_candidates_df: pd.DataFrame,
    threshold: float = 0.65,
    margin: float = 0.08,
    margin_penalty: float = 0.10,
    prob_col: str = "prob",
    all_s1_ids: Sequence[str] | None = None,
) -> dict[str, set[str]]:
    """
    Apply relative margin confidence gating for ambiguous candidate sets.

    Logic:
    ------
    For each S1 entity:
    1. Sort its candidates by predicted probability in descending order: p1 >= p2 >= ...
    2. If there are >= 2 candidates and (p1 - p2) < margin:
       The top candidate is considered "borderline" / ambiguous.
       It is accepted ONLY if p1 >= (threshold + margin_penalty) (stricter bar).
       Otherwise, candidate 1 is rejected.
    3. If there is only 1 candidate (or p1 - p2 >= margin):
       Candidate 1 is evaluated against the standard base threshold (p1 >= threshold).
    4. All lower-ranked candidates (p2, p3, ...) are evaluated against the standard
       base threshold (pi >= threshold) as normal.

    Concrete Example:
    -----------------
    Suppose base threshold = 0.65, margin = 0.10, margin_penalty = 0.10.
    - S1-0001 has Cand A (prob=0.68) and Cand B (prob=0.64).
      Gap = 0.68 - 0.64 = 0.04 < 0.10 (margin triggered).
      Strict threshold = 0.65 + 0.10 = 0.75.
      Cand A has 0.68 < 0.75 -> REJECTED.
      Cand B has 0.64 < 0.65 -> REJECTED.
      Result for S1-0001: set() (singleton predicted instead of noisy false merge).

    Parameters
    ----------
    scored_candidates_df : pd.DataFrame
        DataFrame with source1_entity_id, candidate_entity_id, prob_col.
    threshold : float
        Base matching threshold.
    margin : float
        Minimum required probability gap between top-1 and top-2 candidates.
    margin_penalty : float
        Added penalty to base threshold for borderline top-1 candidates.
    prob_col : str
        Name of the probability column.
    all_s1_ids : Sequence[str] | None
        Optional complete list of S1 IDs.

    Returns
    -------
    dict[str, set[str]]
        Mapping source1_entity_id -> set of predicted matched entity IDs.
    """
    predictions: dict[str, set[str]] = {}

    if all_s1_ids is not None:
        for s1_id in all_s1_ids:
            predictions[s1_id] = set()
    else:
        for s1_id in scored_candidates_df[CAND_COL_SOURCE1_ENTITY_ID].unique():
            predictions[s1_id] = set()

    if scored_candidates_df.empty:
        return predictions

    strict_threshold = threshold + margin_penalty

    # Group candidate scores by S1 entity
    s1_groups = scored_candidates_df.groupby(CAND_COL_SOURCE1_ENTITY_ID)

    for s1_id, group in s1_groups:
        if s1_id not in predictions:
            predictions[s1_id] = set()

        sorted_candidates = group.sort_values(by=prob_col, ascending=False).to_dict(orient="records")
        if not sorted_candidates:
            continue

        n_cands = len(sorted_candidates)
        top_cand = sorted_candidates[0]
        p1 = top_cand[prob_col]
        c1 = top_cand[CAND_COL_CANDIDATE_ENTITY_ID]

        is_ambiguous = False
        if n_cands >= 2:
            p2 = sorted_candidates[1][prob_col]
            if (p1 - p2) < margin:
                is_ambiguous = True

        # Evaluate top-1 candidate
        if is_ambiguous:
            if p1 >= strict_threshold:
                predictions[s1_id].add(c1)
        else:
            if p1 >= threshold:
                predictions[s1_id].add(c1)

        # Evaluate remaining candidates against base threshold
        for cand in sorted_candidates[1:]:
            if cand[prob_col] >= threshold:
                predictions[s1_id].add(cand[CAND_COL_CANDIDATE_ENTITY_ID])

    return predictions


# ─────────────────────────────────────────────────────────────────────────────
# 3. ONE-S1-PER-RECORD DEDUPLICATION ENFORCEMENT
# ─────────────────────────────────────────────────────────────────────────────

def enforce_one_s1_per_record(
    predictions: dict[str, set[str]],
    scored_candidates_df: pd.DataFrame,
    prob_col: str = "prob",
) -> tuple[dict[str, set[str]], int]:
    """
    Enforce the rule that no single candidate (S2/S3 entity) can be assigned
    to more than one Source 1 entity.

    If an S2/S3 candidate ID is predicted under multiple S1 entities, it is kept
    ONLY under the S1 entity where it achieved the highest matching probability,
    and removed from all others.

    Parameters
    ----------
    predictions : dict[str, set[str]]
        Predicted match sets per S1 entity.
    scored_candidates_df : pd.DataFrame
        Pairwise scored DataFrame with prob_col.
    prob_col : str
        Probability column name.

    Returns
    -------
    cleaned_predictions : dict[str, set[str]]
        De-duplicated predictions dictionary.
    conflict_count : int
        Number of candidate conflict instances resolved.
    """
    pair_probs: dict[tuple[str, str], float] = {}
    for _, row in scored_candidates_df[[CAND_COL_SOURCE1_ENTITY_ID, CAND_COL_CANDIDATE_ENTITY_ID, prob_col]].iterrows():
        pair_probs[(row[CAND_COL_SOURCE1_ENTITY_ID], row[CAND_COL_CANDIDATE_ENTITY_ID])] = float(row[prob_col])

    cand_to_s1s: dict[str, list[str]] = defaultdict(list)
    for s1_id, cands in predictions.items():
        for c in cands:
            cand_to_s1s[c].append(s1_id)

    cleaned_predictions = {s1: set(cands) for s1, cands in predictions.items()}
    conflict_count = 0

    for cand_id, s1_list in cand_to_s1s.items():
        if len(s1_list) > 1:
            conflict_count += 1
            best_s1 = max(s1_list, key=lambda s1: pair_probs.get((s1, cand_id), -1.0))
            for s1 in s1_list:
                if s1 != best_s1:
                    cleaned_predictions[s1].discard(cand_id)

    return cleaned_predictions, conflict_count


# ─────────────────────────────────────────────────────────────────────────────
# 4. SINGLETON DIAGNOSTIC MONITORING
# ─────────────────────────────────────────────────────────────────────────────

def count_predicted_singletons(predictions: dict[str, set[str]]) -> int:
    """Return count of entities with empty prediction set."""
    return sum(1 for cands in predictions.values() if len(cands) == 0)


def report_singleton_rate(
    predictions: dict[str, set[str]],
    benchmark_rate: float = 0.0558,
) -> float:
    """
    Compute predicted singleton rate and print diagnostic comparison against
    the ground-truth empirical rate (5.58%).
    """
    total = len(predictions)
    if total == 0:
        return 0.0
    singletons = count_predicted_singletons(predictions)
    rate = singletons / total
    print(f"\n  [Singleton Diagnostic]")
    print(f"    Predicted singletons : {singletons:,d} / {total:,d} ({rate:.2%})")
    print(f"    Ground-truth benchmark: {benchmark_rate:.2%}")
    print(f"    Difference           : {rate - benchmark_rate:+.2%}")
    return rate


# ─────────────────────────────────────────────────────────────────────────────
# 5. SUBMISSION TSV WRITERS (MANDATORY S1_ORDER)
# ─────────────────────────────────────────────────────────────────────────────

def write_matching_results_tsv(
    predictions: dict[str, set[str]],
    output_path: str | Path,
    s1_order: Sequence[str],
) -> None:
    """
    Write matching_results.tsv in the exact challenge submission format.

    Parameters
    ----------
    predictions : dict[str, set[str]]
        Mapping source1_entity_id -> set of predicted matched entity IDs.
    output_path : str | Path
        Destination path for matching_results.tsv.
    s1_order : Sequence[str]
        REQUIRED complete ordered list of all Source 1 test entities. Every S1 entity
        in this sequence is guaranteed to have exactly one row in the output file,
        even if it has zero predicted matches (empty string).
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with out_p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE)
        writer.writerow(list(SUBMISSION_MATCHING_COLS))
        for s1 in s1_order:
            matched_set = predictions.get(s1, set())
            matched_str = ",".join(sorted(matched_set))
            writer.writerow([s1, matched_str])


def write_candidate_pairs_tsv(
    candidate_pairs_df: pd.DataFrame,
    output_path: str | Path,
    s1_order: Sequence[str],
) -> None:
    """
    Write candidate_pairs.tsv in the exact challenge submission format.

    Parameters
    ----------
    candidate_pairs_df : pd.DataFrame
        DataFrame containing candidate pairs with source1_entity_id and candidate_entity_id.
    output_path : str | Path
        Destination path for candidate_pairs.tsv.
    s1_order : Sequence[str]
        REQUIRED complete ordered list of all Source 1 test entities. Every S1 entity
        in this sequence is guaranteed to have exactly one row in the output file,
        even if it has zero nominated candidates (empty string).
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    cand_map: dict[str, set[str]] = defaultdict(set)
    if not candidate_pairs_df.empty:
        for _, row in candidate_pairs_df[[CAND_COL_SOURCE1_ENTITY_ID, CAND_COL_CANDIDATE_ENTITY_ID]].iterrows():
            s1 = row[CAND_COL_SOURCE1_ENTITY_ID]
            c = row[CAND_COL_CANDIDATE_ENTITY_ID]
            cand_map[s1].add(c)

    with out_p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE)
        writer.writerow(list(SUBMISSION_CANDIDATE_COLS))
        for s1 in s1_order:
            cands = cand_map.get(s1, set())
            cands_str = ",".join(sorted(cands))
            writer.writerow([s1, cands_str])


# ─────────────────────────────────────────────────────────────────────────────
# 6. LOCAL SUBMISSION VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

def validate_submission_locally(
    matching_results_path: str | Path,
    candidate_pairs_path: str | Path,
    test_source1_ids: set[str],
    test_source2_ids: set[str],
    test_source3_ids: set[str],
) -> tuple[bool, list[str]]:
    """
    Strict local validator for competition submission files.

    Validates:
    1. Both files exist and are non-empty.
    2. Exact required TSV header columns.
    3. Exactly one row per test source1_entity_id (no duplicates, no missing).
    4. No self-matches (S1 id matching itself).
    5. All candidate and matched IDs belong to source2 or source3 test sets.
    6. No duplicate IDs within any single row.
    7. Matched entity IDs are a strict subset of candidate entity IDs for every S1 entity.
    """
    issues: list[str] = []
    p_match = Path(matching_results_path)
    p_cand = Path(candidate_pairs_path)

    if not p_match.exists():
        issues.append(f"matching_results.tsv not found at {p_match}")
    if not p_cand.exists():
        issues.append(f"candidate_pairs.tsv not found at {p_cand}")
    if issues:
        return False, issues

    valid_cands_pool = test_source2_ids | test_source3_ids

    # 1. Validate matching_results.tsv
    match_s1_seen: set[str] = set()
    matches_per_s1: dict[str, set[str]] = {}

    with p_match.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header != list(SUBMISSION_MATCHING_COLS):
            issues.append(f"matching_results.tsv header invalid: {header} (expected {list(SUBMISSION_MATCHING_COLS)})")

        for line_num, row in enumerate(reader, start=2):
            if len(row) != 2:
                issues.append(f"matching_results.tsv line {line_num}: invalid column count ({len(row)} != 2)")
                continue
            s1_id, match_str = row[0].strip(), row[1].strip()

            if s1_id in match_s1_seen:
                issues.append(f"matching_results.tsv line {line_num}: duplicate source1_entity_id '{s1_id}'")
            match_s1_seen.add(s1_id)

            if s1_id not in test_source1_ids:
                issues.append(f"matching_results.tsv line {line_num}: unknown source1_entity_id '{s1_id}'")

            matched_ids = [m.strip() for m in match_str.split(",") if m.strip()] if match_str else []
            if len(matched_ids) != len(set(matched_ids)):
                issues.append(f"matching_results.tsv line {line_num}: duplicate IDs in matched list for '{s1_id}'")

            for m in matched_ids:
                if m == s1_id:
                    issues.append(f"matching_results.tsv line {line_num}: self-match detected '{s1_id}' -> '{m}'")
                if m not in valid_cands_pool:
                    issues.append(f"matching_results.tsv line {line_num}: matched id '{m}' not in test candidate sources")

            matches_per_s1[s1_id] = set(matched_ids)

    missing_s1_match = test_source1_ids - match_s1_seen
    if missing_s1_match:
        issues.append(f"matching_results.tsv is missing {len(missing_s1_match)} test S1 entities (e.g. {list(missing_s1_match)[:3]})")

    # 2. Validate candidate_pairs.tsv
    cand_s1_seen: set[str] = set()
    cands_per_s1: dict[str, set[str]] = {}

    with p_cand.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header != list(SUBMISSION_CANDIDATE_COLS):
            issues.append(f"candidate_pairs.tsv header invalid: {header} (expected {list(SUBMISSION_CANDIDATE_COLS)})")

        for line_num, row in enumerate(reader, start=2):
            if len(row) != 2:
                issues.append(f"candidate_pairs.tsv line {line_num}: invalid column count ({len(row)} != 2)")
                continue
            s1_id, cand_str = row[0].strip(), row[1].strip()

            if s1_id in cand_s1_seen:
                issues.append(f"candidate_pairs.tsv line {line_num}: duplicate source1_entity_id '{s1_id}'")
            cand_s1_seen.add(s1_id)

            if s1_id not in test_source1_ids:
                issues.append(f"candidate_pairs.tsv line {line_num}: unknown source1_entity_id '{s1_id}'")

            cand_ids = [c.strip() for c in cand_str.split(",") if c.strip()] if cand_str else []
            if len(cand_ids) != len(set(cand_ids)):
                issues.append(f"candidate_pairs.tsv line {line_num}: duplicate IDs in candidate list for '{s1_id}'")

            for c in cand_ids:
                if c not in valid_cands_pool:
                    issues.append(f"candidate_pairs.tsv line {line_num}: candidate id '{c}' not in test candidate sources")

            cands_per_s1[s1_id] = set(cand_ids)

    missing_s1_cand = test_source1_ids - cand_s1_seen
    if missing_s1_cand:
        issues.append(f"candidate_pairs.tsv is missing {len(missing_s1_cand)} test S1 entities (e.g. {list(missing_s1_cand)[:3]})")

    # 3. Cross-file check: matches must be subset of candidates
    for s1_id, matches in matches_per_s1.items():
        candidates = cands_per_s1.get(s1_id, set())
        invalid_matches = matches - candidates
        if invalid_matches:
            issues.append(f"S1 entity '{s1_id}' has matches not present in candidate list: {list(invalid_matches)[:3]}")

    is_valid = len(issues) == 0
    return is_valid, issues


# ─────────────────────────────────────────────────────────────────────────────
# 7. END-TO-END INFERENCE PIPELINE ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

def _run_predict_pipeline() -> None:
    """Run full inference pipeline on synthetic dataset as test harness."""
    from metrics import compute_macro_f05
    from train import generate_synthetic_dataset, predict_probabilities, split_train_val_by_entity, train_model

    SEP = "=" * 75
    sep = "-" * 75

    print(f"\n{SEP}")
    print("  predict.py — End-to-End Inference & Validation Pipeline (Phase B4)")
    print(SEP)

    # 1. Generate synthetic dataset and train model
    print("\n[Step 1/6] Generating synthetic dataset and training model...")
    dataset_df, gt_dict, country_map = generate_synthetic_dataset(num_s1=1000, candidates_per_s1=18, random_state=123)
    train_df, val_df, train_gt, val_gt = split_train_val_by_entity(dataset_df, gt_dict, val_ratio=0.30, random_state=123)

    model = train_model(train_df)
    val_df["prob"] = predict_probabilities(model, val_df)

    test_s1_ids = set(val_gt.keys())

    # --- SIMULATE BLOCKING MISS (Zero-Candidate S1 entities) ---
    # Artificially remove candidate pairs for 3 S1 entities to test zero-candidate resilience
    zero_cand_s1_list = sorted(test_s1_ids)[:3]
    print(f"\n  [Edge Case Test] Simulating blocking miss: removing candidate pairs for 3 entities:")
    for z_id in zero_cand_s1_list:
        print(f"    - {z_id} (0 candidates nominated)")
    val_df_with_zero_cands = val_df[~val_df[CAND_COL_SOURCE1_ENTITY_ID].isin(zero_cand_s1_list)].copy()

    test_s2_ids = {c for c in val_df[CAND_COL_CANDIDATE_ENTITY_ID] if c.startswith("S2")}
    test_s3_ids = {c for c in val_df[CAND_COL_CANDIDATE_ENTITY_ID] if c.startswith("S3")}

    print(f"\n  Validation/Test slice: {len(test_s1_ids):,d} S1 entities, {len(val_df_with_zero_cands):,d} candidate pairs")

    # 2. Apply Base Thresholding
    print("\n[Step 2/6] Applying base threshold (threshold = 0.65)...")
    base_preds = apply_threshold(val_df_with_zero_cands, threshold=0.65, prob_col="prob", all_s1_ids=list(test_s1_ids))
    m_base = compute_macro_f05(base_preds, val_gt)
    print(f"  Base Threshold Macro F0.5 : {m_base['macro_f05']:.4f} (Prec: {m_base['micro_precision']:.4f}, Rec: {m_base['micro_recall']:.4f})")

    # 3. Apply Relative Margin Check
    print("\n[Step 3/6] Applying relative margin check (margin = 0.08, penalty = 0.10)...")
    margin_preds = apply_margin_check(
        val_df_with_zero_cands, threshold=0.65, margin=0.08, margin_penalty=0.10, prob_col="prob", all_s1_ids=list(test_s1_ids)
    )
    m_margin = compute_macro_f05(margin_preds, val_gt)
    print(f"  Margin-Gated Macro F0.5   : {m_margin['macro_f05']:.4f} (Prec: {m_margin['micro_precision']:.4f}, Rec: {m_margin['micro_recall']:.4f})")

    # 4. Enforce One-S1-Per-Record Dedup
    print("\n[Step 4/6] Enforcing one-S1-per-record deduplication rule...")
    final_preds, conflict_count = enforce_one_s1_per_record(margin_preds, val_df_with_zero_cands, prob_col="prob")
    print(f"  Candidate multi-match conflicts resolved: {conflict_count}")
    m_final = compute_macro_f05(final_preds, val_gt)
    print(f"  Final Post-Processed F0.5 : {m_final['macro_f05']:.4f}")

    # 5. Singleton Diagnostic Check
    print("\n[Step 5/6] Checking singleton rate diagnostics...")
    report_singleton_rate(final_preds, benchmark_rate=0.0558)

    # 6. Write submission TSVs and validate locally
    print("\n[Step 6/6] Writing submission TSVs and running local validation...")
    out_dir = Path(__file__).resolve().parents[1] / "output"
    out_match = out_dir / "matching_results.tsv"
    out_cand = out_dir / "candidate_pairs.tsv"

    s1_order_list = sorted(test_s1_ids)
    write_matching_results_tsv(final_preds, out_match, s1_order=s1_order_list)
    write_candidate_pairs_tsv(val_df_with_zero_cands, out_cand, s1_order=s1_order_list)

    print(f"  Saved: {out_match}")
    print(f"  Saved: {out_cand}")

    # Explicit assertion for zero-candidate rows
    with out_match.open("r", encoding="utf-8") as f:
        match_rows = dict(csv.reader(f, delimiter="\t"))
    with out_cand.open("r", encoding="utf-8") as f:
        cand_rows = dict(csv.reader(f, delimiter="\t"))

    for z_id in zero_cand_s1_list:
        assert z_id in match_rows, f"{z_id} missing from matching_results.tsv!"
        assert match_rows[z_id] == "", f"{z_id} should have empty match string!"
        assert z_id in cand_rows, f"{z_id} missing from candidate_pairs.tsv!"
        assert cand_rows[z_id] == "", f"{z_id} should have empty candidate string!"

    print(f"  [PASS] Confirmed zero-candidate entities successfully written with empty cells in both TSVs.")

    is_valid, issues = validate_submission_locally(
        matching_results_path=out_match,
        candidate_pairs_path=out_cand,
        test_source1_ids=test_s1_ids,
        test_source2_ids=test_s2_ids,
        test_source3_ids=test_s3_ids,
    )

    print(f"\n{sep}")
    print(f"  Local Validator Result: {'[PASS]' if is_valid else '[FAIL]'}")
    print(f"{sep}")
    if is_valid:
        print("  All local validation checks PASSED with 0 errors!")
    else:
        print(f"  Validation failed with {len(issues)} issues:")
        for idx, issue in enumerate(issues[:10], start=1):
            print(f"    {idx}. {issue}")

    assert is_valid, "Submission validation failed!"

    print(f"\n{SEP}")
    print("  PHASE B4 COMPLETE — PREDICT PIPELINE VALIDATED SUCCESSFULLY")
    print(f"{SEP}\n")


if __name__ == "__main__":
    _run_predict_pipeline()
