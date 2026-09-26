"""
train.py — Model training, threshold tuning, and cross-country evaluation.

Person B track (Phase B3).

This module implements:
1. Hardened synthetic dataset generation reflecting the empirical ground-truth distribution
   from Phase B1 (5.58% singletons, median 4 matches, max 11 matches) with multi-feature
   overlapping hard decoys.
2. Grouped train/validation split by source1_entity_id (no data leakage).
3. LightGBM binary classifier training on Schema 4 feature table.
4. Threshold sweep to maximize the official competition metric (macro F0.5 via metrics.py).
5. Cross-country generalization analysis (US, IN, and held-out FR proxy).

Usage
-----
    python src/train.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
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
    GT_COL_MATCHED_ENTITY_IDS,
    GT_COL_SOURCE1_ENTITY_ID,
    OUTPUT_MODEL_PATH,
)
from metrics import compute_macro_f05


# ─────────────────────────────────────────────────────────────────────────────
# 1. HARDENED SYNTHETIC DATASET GENERATION
# ─────────────────────────────────────────────────────────────────────────────

# Empirical distribution from eda_notes_B.md (Phase B1):
EMPIRICAL_MATCH_DISTRIBUTION: dict[int, float] = {
    0: 0.055848,
    1: 0.053995,
    2: 0.170028,
    3: 0.240546,
    4: 0.219372,
    5: 0.145892,
    6: 0.074708,
    7: 0.028986,
    8: 0.008465,
    9: 0.001905,
    10: 0.000242,
    11: 0.000017,
}


def generate_synthetic_dataset(
    num_s1: int = 1500,
    candidates_per_s1: int = 18,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, set[str]], dict[str, str]]:
    """
    Generate synthetic candidate pairs and Schema 4 feature table matching
    the real empirical ground-truth distribution from Phase B1.

    Includes multi-feature overlapping hard decoys (e.g. near-duplicate businesses,
    chain lookalikes, same-address different business) and noisy true matches
    so the classification problem is non-trivial and produces a realistic threshold curve.
    """
    rng = np.random.default_rng(random_state)

    # 1. Sample match counts according to empirical distribution
    match_counts_keys = list(EMPIRICAL_MATCH_DISTRIBUTION.keys())
    match_probs = np.array(list(EMPIRICAL_MATCH_DISTRIBUTION.values()))
    match_probs /= match_probs.sum()

    s1_match_counts = rng.choice(match_counts_keys, size=num_s1, p=match_probs)

    # 2. Assign country buckets (US: 45%, IN: 45%, FR: 10%)
    country_choices = ["US", "IN", "FR"]
    country_probs = [0.45, 0.45, 0.10]
    s1_countries = rng.choice(country_choices, size=num_s1, p=country_probs)

    rows = []
    ground_truth: dict[str, set[str]] = {}
    country_map: dict[str, str] = {}

    for i in range(num_s1):
        s1_id = f"S1-SYN-{i:05d}"
        country = s1_countries[i]
        n_true = s1_match_counts[i]
        country_map[s1_id] = country

        true_cand_ids: set[str] = set()

        n_candidates = max(candidates_per_s1, n_true)
        n_decoys = n_candidates - n_true

        raw_scores = []
        cand_rows = []

        # --- A. True Match Candidates ---
        for t_idx in range(n_true):
            src_prefix = "S2" if rng.random() < 0.484 else "S3"
            c_id = f"{src_prefix}-SYN-{i:05d}-{t_idx:02d}"
            true_cand_ids.add(c_id)

            is_noisy_match = rng.random() < 0.25
            is_landmark = 1.0 if (country == "IN" and rng.random() < 0.35) else (1.0 if rng.random() < 0.05 else 0.0)

            if is_noisy_match:
                # Noisy true match (heavy abbreviation or typo)
                name_sim = rng.uniform(0.62, 0.82)
                addr_sim = rng.uniform(0.50, 0.78) if not is_landmark else rng.uniform(0.40, 0.65)
                hn_match = 0.0 if (is_landmark or rng.random() < 0.30) else 1.0
                post_match = 1.0 if rng.random() < 0.80 else 0.0
                city_overlap = rng.uniform(0.50, 0.85)
                first_tok = 1.0 if rng.random() < 0.85 else 0.0
                blocking_score = float(rng.uniform(0.68, 0.85))
            else:
                # Standard true match
                name_sim = rng.uniform(0.78, 0.98)
                addr_sim = rng.uniform(0.72, 0.98) if not is_landmark else rng.uniform(0.55, 0.85)
                hn_match = 0.0 if is_landmark else (1.0 if rng.random() < 0.90 else 0.0)
                post_match = 1.0 if rng.random() < 0.94 else 0.0
                city_overlap = float(rng.uniform(0.75, 1.0))
                first_tok = 1.0 if rng.random() < 0.95 else 0.0
                blocking_score = float(rng.uniform(0.82, 0.99))

            raw_scores.append(blocking_score)

            cand_rows.append({
                FEAT_COL_SOURCE1_ENTITY_ID: s1_id,
                FEAT_COL_CANDIDATE_ENTITY_ID: c_id,
                "label": 1,
                "_raw_score": blocking_score,
                FEAT_COL_LEV_DIST_RAW: float(np.clip(1.0 - name_sim + rng.normal(0, 0.06), 0.0, 1.0)),
                FEAT_COL_JARO_WINKLER_RAW: float(np.clip(name_sim + rng.normal(0, 0.04), 0.0, 1.0)),
                FEAT_COL_TOKEN_SORT_RATIO_NORM: float(np.clip(name_sim + rng.normal(0, 0.05), 0.0, 1.0)),
                FEAT_COL_TOKEN_SET_RATIO_NORM: float(np.clip(name_sim + rng.uniform(0.0, 0.12), 0.0, 1.0)),
                FEAT_COL_TFIDF_COSINE_CHAR: float(np.clip(name_sim + rng.normal(0, 0.05), 0.0, 1.0)),
                FEAT_COL_TFIDF_COSINE_WORD: float(np.clip(name_sim + rng.normal(0, 0.06), 0.0, 1.0)),
                FEAT_COL_JACCARD_TOKENS_FULL: float(np.clip(name_sim * 0.85 + rng.normal(0, 0.06), 0.0, 1.0)),
                FEAT_COL_JACCARD_TOKENS_CORE: float(np.clip(name_sim + rng.normal(0, 0.05), 0.0, 1.0)),
                FEAT_COL_FIRST_TOKEN_MATCH: first_tok,
                FEAT_COL_NAME_LEN_DIFF_RATIO: float(np.clip(rng.uniform(0.0, 0.25), 0.0, 1.0)),
                FEAT_COL_ADDR_LEV_DIST: float(np.clip(1.0 - addr_sim + rng.normal(0, 0.06), 0.0, 1.0)),
                FEAT_COL_ADDR_JARO_WINKLER: float(np.clip(addr_sim + rng.normal(0, 0.04), 0.0, 1.0)),
                FEAT_COL_ADDR_TOKEN_SORT_RATIO: float(np.clip(addr_sim + rng.normal(0, 0.06), 0.0, 1.0)),
                FEAT_COL_ADDR_TOKEN_SET_RATIO: float(np.clip(addr_sim + rng.uniform(0.0, 0.10), 0.0, 1.0)),
                FEAT_COL_ADDR_JACCARD: float(np.clip(addr_sim * 0.9 + rng.normal(0, 0.06), 0.0, 1.0)),
                FEAT_COL_HOUSE_NUMBER_MATCH: hn_match,
                FEAT_COL_POSTAL_PREFIX_MATCH: post_match,
                FEAT_COL_CITY_TOKEN_OVERLAP: float(np.clip(city_overlap, 0.0, 1.0)),
                FEAT_COL_LANDMARK_FLAG: is_landmark,
                FEAT_COL_SAME_COUNTRY_FLAG: 1.0 if rng.random() < 0.98 else 0.0,
                FEAT_COL_BLOCKED_BY_NAME_TOKEN: 1.0 if rng.random() < 0.92 else 0.0,
                FEAT_COL_BLOCKED_BY_SNM: 1.0 if rng.random() < 0.80 else 0.0,
                FEAT_COL_BLOCKED_BY_CHAR_TFIDF: 1.0 if rng.random() < 0.88 else 0.0,
                FEAT_COL_BLOCKED_BY_WORD_TFIDF: 1.0 if rng.random() < 0.68 else 0.0,
                FEAT_COL_BLOCKED_BY_ADDR_TOKEN: 1.0 if rng.random() < 0.65 else 0.0,
                FEAT_COL_BLOCKED_BY_MINHASH: 1.0 if (is_landmark or rng.random() < 0.4) else 0.0,
            })

        # --- B. Decoy Candidates (Negatives & Multi-Feature Hard Negatives) ---
        for d_idx in range(n_decoys):
            src_prefix = "S2" if rng.random() < 0.5 else "S3"
            c_id = f"{src_prefix}-SYN-DEC-{i:05d}-{d_idx:02d}"

            decoy_type = rng.choice(["multi_feature_hard", "name_hard", "addr_hard", "easy"], p=[0.40, 0.20, 0.15, 0.25])

            if decoy_type == "multi_feature_hard":
                # Multi-feature overlapping hard decoy: near-duplicate / competitor / branch
                name_sim = rng.uniform(0.78, 0.94)
                addr_sim = rng.uniform(0.55, 0.84)
                hn_match = 1.0 if rng.random() < 0.25 else 0.0
                post_match = 1.0 if rng.random() < 0.65 else 0.0
                city_overlap = rng.uniform(0.40, 0.85)
                first_tok = 1.0 if rng.random() < 0.80 else 0.0
                blocking_score = float(rng.uniform(0.65, 0.88))
                n_flags = rng.choice([2, 3])
            elif decoy_type == "name_hard":
                # Lookalike name, different location
                name_sim = rng.uniform(0.72, 0.90)
                addr_sim = rng.uniform(0.10, 0.38)
                hn_match = 0.0
                post_match = 0.0 if rng.random() < 0.85 else 1.0
                city_overlap = rng.uniform(0.0, 0.25)
                first_tok = 1.0 if rng.random() < 0.75 else 0.0
                blocking_score = float(rng.uniform(0.50, 0.72))
                n_flags = 2
            elif decoy_type == "addr_hard":
                # Same address / building, different business
                name_sim = rng.uniform(0.12, 0.38)
                addr_sim = rng.uniform(0.75, 0.95)
                hn_match = 1.0 if rng.random() < 0.80 else 0.0
                post_match = 1.0
                city_overlap = 1.0
                first_tok = 0.0
                blocking_score = float(rng.uniform(0.45, 0.68))
                n_flags = 2
            else:
                # Easy decoy
                name_sim = rng.uniform(0.05, 0.40)
                addr_sim = rng.uniform(0.05, 0.40)
                hn_match = 0.0
                post_match = 0.0 if rng.random() < 0.85 else 1.0
                city_overlap = rng.uniform(0.0, 0.15)
                first_tok = 0.0
                blocking_score = float(rng.uniform(0.15, 0.45))
                n_flags = 1

            raw_scores.append(blocking_score)

            cand_rows.append({
                FEAT_COL_SOURCE1_ENTITY_ID: s1_id,
                FEAT_COL_CANDIDATE_ENTITY_ID: c_id,
                "label": 0,
                "_raw_score": blocking_score,
                FEAT_COL_LEV_DIST_RAW: float(np.clip(1.0 - name_sim + rng.normal(0, 0.06), 0.0, 1.0)),
                FEAT_COL_JARO_WINKLER_RAW: float(np.clip(name_sim + rng.normal(0, 0.04), 0.0, 1.0)),
                FEAT_COL_TOKEN_SORT_RATIO_NORM: float(np.clip(name_sim + rng.normal(0, 0.05), 0.0, 1.0)),
                FEAT_COL_TOKEN_SET_RATIO_NORM: float(np.clip(name_sim + rng.uniform(0.0, 0.10), 0.0, 1.0)),
                FEAT_COL_TFIDF_COSINE_CHAR: float(np.clip(name_sim + rng.normal(0, 0.05), 0.0, 1.0)),
                FEAT_COL_TFIDF_COSINE_WORD: float(np.clip(name_sim + rng.normal(0, 0.06), 0.0, 1.0)),
                FEAT_COL_JACCARD_TOKENS_FULL: float(np.clip(name_sim * 0.85 + rng.normal(0, 0.06), 0.0, 1.0)),
                FEAT_COL_JACCARD_TOKENS_CORE: float(np.clip(name_sim + rng.normal(0, 0.05), 0.0, 1.0)),
                FEAT_COL_FIRST_TOKEN_MATCH: first_tok,
                FEAT_COL_NAME_LEN_DIFF_RATIO: float(np.clip(rng.uniform(0.05, 0.50), 0.0, 1.0)),
                FEAT_COL_ADDR_LEV_DIST: float(np.clip(1.0 - addr_sim + rng.normal(0, 0.06), 0.0, 1.0)),
                FEAT_COL_ADDR_JARO_WINKLER: float(np.clip(addr_sim + rng.normal(0, 0.04), 0.0, 1.0)),
                FEAT_COL_ADDR_TOKEN_SORT_RATIO: float(np.clip(addr_sim + rng.normal(0, 0.06), 0.0, 1.0)),
                FEAT_COL_ADDR_TOKEN_SET_RATIO: float(np.clip(addr_sim + rng.uniform(0.0, 0.08), 0.0, 1.0)),
                FEAT_COL_ADDR_JACCARD: float(np.clip(addr_sim * 0.85 + rng.normal(0, 0.06), 0.0, 1.0)),
                FEAT_COL_HOUSE_NUMBER_MATCH: hn_match,
                FEAT_COL_POSTAL_PREFIX_MATCH: post_match,
                FEAT_COL_CITY_TOKEN_OVERLAP: float(np.clip(city_overlap, 0.0, 1.0)),
                FEAT_COL_LANDMARK_FLAG: 1.0 if rng.random() < 0.08 else 0.0,
                FEAT_COL_SAME_COUNTRY_FLAG: 1.0 if rng.random() < 0.85 else 0.0,
                FEAT_COL_BLOCKED_BY_NAME_TOKEN: 1.0 if (n_flags >= 1 and name_sim > 0.4) else 0.0,
                FEAT_COL_BLOCKED_BY_SNM: 1.0 if (n_flags >= 2 and name_sim > 0.6) else 0.0,
                FEAT_COL_BLOCKED_BY_CHAR_TFIDF: 1.0 if (n_flags >= 2 and name_sim > 0.5) else 0.0,
                FEAT_COL_BLOCKED_BY_WORD_TFIDF: 1.0 if (n_flags >= 3 and name_sim > 0.65) else 0.0,
                FEAT_COL_BLOCKED_BY_ADDR_TOKEN: 1.0 if (addr_sim > 0.5 and rng.random() < 0.8) else 0.0,
                FEAT_COL_BLOCKED_BY_MINHASH: 1.0 if rng.random() < 0.15 else 0.0,
            })

        max_score = max(raw_scores) if raw_scores else 0.0
        for r in cand_rows:
            r[FEAT_COL_CANDIDATE_SCORE_GAP] = float(max_score - r.pop("_raw_score"))
            rows.append(r)

        ground_truth[s1_id] = true_cand_ids

    df = pd.DataFrame(rows)
    return df, ground_truth, country_map


# ─────────────────────────────────────────────────────────────────────────────
# 2. GROUPED TRAIN / VALIDATION SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def split_train_val_by_entity(
    feature_table: pd.DataFrame,
    ground_truth: dict[str, set[str]],
    val_ratio: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, set[str]], dict[str, set[str]]]:
    """
    Split feature table strictly by source1_entity_id so no entity has pairs
    spanning across both train and validation sets (no data leakage).

    Stratifies by match-count bucket (0, 1, 2, 3+).
    """
    rng = np.random.default_rng(random_state)

    buckets: dict[str, list[str]] = {"0": [], "1": [], "2": [], "3+": []}
    for s1_id, gt_set in ground_truth.items():
        k = len(gt_set)
        if k == 0:
            buckets["0"].append(s1_id)
        elif k == 1:
            buckets["1"].append(s1_id)
        elif k == 2:
            buckets["2"].append(s1_id)
        else:
            buckets["3+"].append(s1_id)

    train_s1_set: set[str] = set()
    val_s1_set: set[str] = set()

    for b_name, s1_list in buckets.items():
        shuffled = list(s1_list)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_ratio))
        val_s1_set.update(shuffled[:n_val])
        train_s1_set.update(shuffled[n_val:])

    train_df = feature_table[feature_table[CAND_COL_SOURCE1_ENTITY_ID].isin(train_s1_set)].copy()
    val_df = feature_table[feature_table[CAND_COL_SOURCE1_ENTITY_ID].isin(val_s1_set)].copy()

    train_gt = {k: v for k, v in ground_truth.items() if k in train_s1_set}
    val_gt = {k: v for k, v in ground_truth.items() if k in val_s1_set}

    return train_df, val_df, train_gt, val_gt


# ─────────────────────────────────────────────────────────────────────────────
# 3. LIGHTGBM TRAINING & PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    train_feature_table: pd.DataFrame,
    feature_cols: list[str] | None = None,
    output_path: str | Path | None = None,
    **kwargs,
) -> lgb.LGBMClassifier:
    """
    Train a LightGBM binary classifier on the training feature table.
    """
    if feature_cols is None:
        feature_cols = list(FEATURE_COLUMNS)

    X_train = train_feature_table[feature_cols]
    y_train = train_feature_table["label"]

    default_params = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
    }
    default_params.update(kwargs)

    model = lgb.LGBMClassifier(**default_params)
    model.fit(X_train, y_train)

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(str(out_p))

    return model


def predict_probabilities(
    model: lgb.LGBMClassifier,
    feature_table: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> np.ndarray:
    """Score matching probability for every candidate pair."""
    if feature_cols is None:
        feature_cols = list(FEATURE_COLUMNS)
    X = feature_table[feature_cols]
    return model.predict_proba(X)[:, 1]


# ─────────────────────────────────────────────────────────────────────────────
# 4. THRESHOLD SWEEP & EVALUATION AGAINST MACRO F0.5
# ─────────────────────────────────────────────────────────────────────────────

def sweep_thresholds(
    model: lgb.LGBMClassifier,
    val_feature_table: pd.DataFrame,
    val_ground_truth: dict[str, set[str]],
    thresholds: Sequence[float] | None = None,
) -> tuple[float, float, pd.DataFrame]:
    """
    Sweep decision thresholds against the official macro F0.5 competition metric.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.05, 1.00, 0.05)]

    val_df = val_feature_table.copy()
    val_df["prob"] = predict_probabilities(model, val_df)

    sweep_records = []
    best_f05 = -1.0
    best_thresh = 0.50

    pairs_by_s1 = val_df[[CAND_COL_SOURCE1_ENTITY_ID, CAND_COL_CANDIDATE_ENTITY_ID, "prob"]].to_dict(
        orient="records"
    )

    s1_to_cand_probs: dict[str, list[tuple[str, float]]] = {s1: [] for s1 in val_ground_truth.keys()}
    for row in pairs_by_s1:
        s1 = row[CAND_COL_SOURCE1_ENTITY_ID]
        c = row[CAND_COL_CANDIDATE_ENTITY_ID]
        p = row["prob"]
        if s1 in s1_to_cand_probs:
            s1_to_cand_probs[s1].append((c, p))

    for thresh in thresholds:
        predictions: dict[str, set[str]] = {}
        for s1_id, cand_list in s1_to_cand_probs.items():
            matched = {c for c, p in cand_list if p >= thresh}
            predictions[s1_id] = matched

        metrics = compute_macro_f05(predictions, val_ground_truth)
        macro_f05 = metrics["macro_f05"]
        precision = metrics["micro_precision"]
        recall = metrics["micro_recall"]
        sing_acc = metrics["singleton_accuracy"]

        sweep_records.append({
            "threshold": thresh,
            "macro_f05": macro_f05,
            "micro_precision": precision,
            "micro_recall": recall,
            "singleton_accuracy": sing_acc,
        })

        if macro_f05 > best_f05:
            best_f05 = macro_f05
            best_thresh = thresh

    sweep_df = pd.DataFrame(sweep_records)
    return best_thresh, best_f05, sweep_df


def find_match_threshold(
    model: lgb.LGBMClassifier,
    val_feature_table: pd.DataFrame,
    val_ground_truth: dict[str, set[str]],
) -> float:
    """Wrapper returning the threshold that maximises macro F0.5."""
    best_thresh, _, _ = sweep_thresholds(model, val_feature_table, val_ground_truth)
    return best_thresh


# ─────────────────────────────────────────────────────────────────────────────
# 5. CROSS-COUNTRY EVALUATION (SYMMETRIC & INDEPENDENT SPLITS)
# ─────────────────────────────────────────────────────────────────────────────

def run_cross_country_evaluation(
    dataset_df: pd.DataFrame,
    ground_truth: dict[str, set[str]],
    country_map: dict[str, str],
    best_threshold: float,
) -> dict:
    """
    Evaluate cross-country generalization and quantify performance degradation.

    Symmetric split methodology:
    1. Scenario 1 (Train US+IN -> Test FR held-out proxy):
       Train on full US + IN data, evaluate on held-out France proxy.
    2. Scenario 2 (Train US -> Test IN domain adaptation):
       Train independently on full US data, evaluate domain transfer on full IN data.
    3. Scenario 3 (Train IN -> Test US domain adaptation):
       Train independently on full IN data, evaluate domain transfer on full US data.
    """
    df = dataset_df.copy()
    df["country"] = df[CAND_COL_SOURCE1_ENTITY_ID].map(country_map)

    # Country slices
    df_us = df[df["country"] == "US"].copy()
    df_in = df[df["country"] == "IN"].copy()
    df_fr = df[df["country"] == "FR"].copy()

    gt_us = {k: v for k, v in ground_truth.items() if country_map.get(k) == "US"}
    gt_in = {k: v for k, v in ground_truth.items() if country_map.get(k) == "IN"}
    gt_fr = {k: v for k, v in ground_truth.items() if country_map.get(k) == "FR"}

    # --- Scenario 1: Train US+IN -> Test FR ---
    train_us_in = pd.concat([df_us, df_in], ignore_index=True)
    model_us_in = train_model(train_us_in)
    val_fr = df_fr.copy()
    val_fr["prob"] = predict_probabilities(model_us_in, val_fr)

    preds_fr: dict[str, set[str]] = {s1: set() for s1 in gt_fr.keys()}
    for _, row in val_fr.iterrows():
        if row["prob"] >= best_threshold:
            preds_fr[row[CAND_COL_SOURCE1_ENTITY_ID]].add(row[CAND_COL_CANDIDATE_ENTITY_ID])
    m_fr = compute_macro_f05(preds_fr, gt_fr)

    # --- Scenario 2: Train US -> Test IN ---
    model_us = train_model(df_us)
    val_in = df_in.copy()
    val_in["prob"] = predict_probabilities(model_us, val_in)

    preds_in: dict[str, set[str]] = {s1: set() for s1 in gt_in.keys()}
    for _, row in val_in.iterrows():
        if row["prob"] >= best_threshold:
            preds_in[row[CAND_COL_SOURCE1_ENTITY_ID]].add(row[CAND_COL_CANDIDATE_ENTITY_ID])
    m_us_to_in = compute_macro_f05(preds_in, gt_in)

    # --- Scenario 3: Train IN -> Test US (Clean independent IN training) ---
    model_in = train_model(df_in)
    val_us = df_us.copy()
    val_us["prob"] = predict_probabilities(model_in, val_us)

    preds_us: dict[str, set[str]] = {s1: set() for s1 in gt_us.keys()}
    for _, row in val_us.iterrows():
        if row["prob"] >= best_threshold:
            preds_us[row[CAND_COL_SOURCE1_ENTITY_ID]].add(row[CAND_COL_CANDIDATE_ENTITY_ID])
    m_in_to_us = compute_macro_f05(preds_us, gt_us)

    return {
        "us_in_to_fr": m_fr,
        "us_to_in": m_us_to_in,
        "in_to_us": m_in_to_us,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. STANDALONE PIPELINE EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline() -> None:
    SEP = "=" * 75
    sep = "-" * 75

    print(f"\n{SEP}")
    print("  train.py — End-to-End Training & Evaluation Pipeline (Phase B3)")
    print(SEP)

    # 1. Generate synthetic dataset
    print("\n[Step 1/5] Generating synthetic dataset (1,500 entities, ~27k pairs)...")
    dataset_df, gt_dict, country_map = generate_synthetic_dataset(num_s1=1500, candidates_per_s1=18)

    total_s1 = len(gt_dict)
    counts = [len(gt) for gt in gt_dict.values()]
    hist = Counter(counts)

    print(f"\n{sep}")
    print("  Match-Count Distribution Comparison (Empirical B1 vs Synthetic B3)")
    print(sep)
    print(f"  {'#Matches':<10} {'Empirical %':>13} {'Synthetic Count':>17} {'Synthetic %':>13}")
    print(f"  {'-'*57}")
    for k in sorted(EMPIRICAL_MATCH_DISTRIBUTION.keys()):
        emp_pct = EMPIRICAL_MATCH_DISTRIBUTION[k] * 100
        syn_cnt = hist.get(k, 0)
        syn_pct = syn_cnt / total_s1 * 100
        print(f"  {k:<10} {emp_pct:>12.2f}% {syn_cnt:>17,d} {syn_pct:>12.2f}%")

    # 2. Split train/val by S1 entity
    print(f"\n[Step 2/5] Performing grouped train/val split (80/20 by S1 entity)...")
    train_df, val_df, train_gt, val_gt = split_train_val_by_entity(dataset_df, gt_dict, val_ratio=0.20)
    print(f"  Train pairs : {len(train_df):,d} across {len(train_gt):,d} S1 entities")
    print(f"  Val pairs   : {len(val_df):,d} across {len(val_gt):,d} S1 entities")

    train_s1 = set(train_df[CAND_COL_SOURCE1_ENTITY_ID])
    val_s1 = set(val_df[CAND_COL_SOURCE1_ENTITY_ID])
    assert len(train_s1 & val_s1) == 0, "Data leakage detected across train and val splits!"
    print("  [PASS] Zero data leakage: train and val entity sets are completely disjoint.")

    # 3. Train LightGBM model
    print(f"\n[Step 3/5] Training LightGBM binary classifier on {len(FEATURE_COLUMNS)} features...")
    model = train_model(train_df)
    print("  [PASS] LightGBM model trained successfully.")

    # 4. Threshold sweep
    print(f"\n[Step 4/5] Executing threshold sweep against macro F0.5 on validation set...")
    best_thresh, best_f05, sweep_df = sweep_thresholds(model, val_df, val_gt)

    print(f"\n{sep}")
    print("  Validation Threshold Sweep Results")
    print(sep)
    print(f"  {'Threshold':<10} {'Macro F0.5':>12} {'Precision':>12} {'Recall':>10} {'Singleton Acc':>15}")
    print(f"  {'-'*63}")
    for _, row in sweep_df.iterrows():
        t = row["threshold"]
        marker = " <== [OPTIMAL]" if abs(t - best_thresh) < 1e-4 else ""
        print(
            f"  {t:<10.2f} {row['macro_f05']:>12.4f} {row['micro_precision']:>12.4f} "
            f"{row['micro_recall']:>10.4f} {row['singleton_accuracy']:>15.2%}{marker}"
        )

    print(f"\n  >> BEST THRESHOLD: {best_thresh:.2f} (Macro F0.5 = {best_f05:.4f})")

    # 5. Cross-country evaluation
    print(f"\n[Step 5/5] Running cross-country generalization evaluation...")
    cc_res = run_cross_country_evaluation(dataset_df, gt_dict, country_map, best_thresh)

    fr_f05 = cc_res["us_in_to_fr"]["macro_f05"]
    us_to_in_f05 = cc_res["us_to_in"]["macro_f05"]
    in_to_us_f05 = cc_res["in_to_us"]["macro_f05"]

    delta_fr = fr_f05 - best_f05
    delta_us_in = us_to_in_f05 - best_f05
    delta_in_us = in_to_us_f05 - best_f05

    print(f"\n{sep}")
    print("  Cross-Country Evaluation Results (Symmetric Independent Splits)")
    print(sep)
    print(f"  1. Train (US + IN) -> Test FR (Held-out proxy):")
    print(f"     Macro F0.5 : {fr_f05:.4f}  (Delta vs same-dist: {delta_fr:+.4f})")
    print(f"     Precision  : {cc_res['us_in_to_fr']['micro_precision']:.4f}")
    print(f"     Recall     : {cc_res['us_in_to_fr']['micro_recall']:.4f}")
    print()
    print(f"  2. Train US -> Test IN:")
    print(f"     Macro F0.5 : {us_to_in_f05:.4f}  (Delta vs same-dist: {delta_us_in:+.4f})")
    print(f"     Precision  : {cc_res['us_to_in']['micro_precision']:.4f}")
    print(f"     Recall     : {cc_res['us_to_in']['micro_recall']:.4f}")
    print()
    print(f"  3. Train IN -> Test US:")
    print(f"     Macro F0.5 : {in_to_us_f05:.4f}  (Delta vs same-dist: {delta_in_us:+.4f})")
    print(f"     Precision  : {cc_res['in_to_us']['micro_precision']:.4f}")
    print(f"     Recall     : {cc_res['in_to_us']['micro_recall']:.4f}")

    print(f"\n{SEP}")
    print("  ALL TRAINING PIPELINE CHECKS COMPLETED SUCCESSFULLY")
    print(f"{SEP}\n")


if __name__ == "__main__":
    _run_pipeline()
