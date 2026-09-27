"""
blocking.py — Candidate pair generation (blocking / indexing stage).

Phase A3 implementation (Person A Track).

This module consumes Normalized record DataFrames (Schema 2) and produces the
Candidate pairs schema (Schema 3).

Strategy Catalogue
------------------
1. name_token   — Inverted-index on tokenised name_core with coarse address signal
                  (postal_prefix / house_number) and block-size guardrail for common names.
2. snm          — Sorted-Neighbourhood Method on (country_norm, name_core) with
                  sliding window of size SNM_WINDOW_SIZE.
3. char_tfidf   — TF-IDF (char 2-4 n-grams) on concatenated name + address + cosine kNN
                  (NearestNeighbors top-k per S1 entity).
4. word_tfidf   — TF-IDF (word 1-2 n-grams) on name_core + cosine kNN.
5. addr_token   — Inverted-index on (country, postal_prefix, house_number) with
                  fallback to landmark token keys for addresses without house numbers.
6. minhash      — MinHash / LSH (64 hash permutations, 16 bands) fallback for large
                  candidate sets where len(candidates) >= CANDIDATE_SIZE_THRESHOLD.

Combiner & Gating Logic
-----------------------
All strategies emit (source1_entity_id, candidate_entity_id, raw_score) pairs.
`merge_strategy_outputs` unions all strategy pairs, deduplicates, records
`blocking_strategy_flags` (comma-joined strategy names), and computes `blocking_score`.
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Ensure src/ is on the path regardless of invocation directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    CAND_COL_BLOCKING_SCORE,
    CAND_COL_BLOCKING_STRATEGY_FLAGS,
    CAND_COL_CANDIDATE_ENTITY_ID,
    CAND_COL_SOURCE1_ENTITY_ID,
    CANDIDATE_PAIR_COLUMNS,
    CANDIDATE_SIZE_THRESHOLD,
    GT_COL_MATCHED_ENTITY_IDS,
    GT_COL_SOURCE1_ENTITY_ID,
    NORM_COL_ADDRESS_EXPANDED,
    NORM_COL_COUNTRY_NORM,
    NORM_COL_ENTITY_ID,
    NORM_COL_HOUSE_NUMBER,
    NORM_COL_LANDMARK,
    NORM_COL_NAME_CORE,
    NORM_COL_NAME_EXPANDED,
    NORM_COL_POSTAL_PREFIX,
    SNM_WINDOW_SIZE,
    TFIDF_NGRAM_RANGE_CHAR,
    TFIDF_NGRAM_RANGE_WORD,
    TOP_K_CANDIDATES,
)


# ---------------------------------------------------------------------------
# 1. NAME TOKEN BLOCKING WITH BLOCK-SIZE GUARDRAIL
# ---------------------------------------------------------------------------

def name_token_block(
    source1_norm: pd.DataFrame,
    candidates_norm: pd.DataFrame,
    max_block_size: int = 50,
) -> pd.DataFrame:
    """
    Inverted-index blocking on significant name_core tokens with coarse address component.

    Guardrail Strategy
    ------------------
    For high-frequency tokens (appearing in > max_block_size candidate records,
    such as chain names or common generic business words), a coarse address component
    (postal_prefix or house_number) is required in the blocking key to avoid
    generating massive O(N^2) candidate blocks. Distinctive/rare tokens allow
    pure (country, token) matching.

    Parameters
    ----------
    source1_norm : pd.DataFrame
        Normalized Source 1 records.
    candidates_norm : pd.DataFrame
        Normalized candidate records.
    max_block_size : int, default 50
        Maximum candidate frequency threshold before requiring address conjunction.

    Returns
    -------
    pd.DataFrame
        Columns: source1_entity_id, candidate_entity_id, raw_score.
    """
    cand_index = defaultdict(list)
    cand_token_freq: Counter[str] = Counter()

    cand_records = []
    for idx, row in candidates_norm.iterrows():
        cid = row[NORM_COL_ENTITY_ID]
        country = row[NORM_COL_COUNTRY_NORM]
        name = row[NORM_COL_NAME_CORE]
        post = row[NORM_COL_POSTAL_PREFIX][:3] if row[NORM_COL_POSTAL_PREFIX] else ""
        hno = row[NORM_COL_HOUSE_NUMBER]

        tokens = [t for t in name.split() if len(t) >= 3]
        for t in set(tokens):
            cand_token_freq[t] += 1
        cand_records.append((cid, country, tokens, post, hno))

    # Populate inverted index with guardrails
    for cid, country, tokens, post, hno in cand_records:
        for t in set(tokens):
            freq = cand_token_freq[t]
            if freq > max_block_size:
                if post:
                    key = (country, t, "post", post)
                    cand_index[key].append(cid)
                elif hno:
                    key = (country, t, "hno", hno)
                    cand_index[key].append(cid)
                else:
                    if len(tokens) >= 2:
                        key = (country, tuple(sorted(tokens[:2])))
                        cand_index[key].append(cid)
            else:
                key = (country, t)
                cand_index[key].append(cid)

    # Query index with Source 1 entities
    pairs = []
    for idx, row in source1_norm.iterrows():
        s1_id = row[NORM_COL_ENTITY_ID]
        country = row[NORM_COL_COUNTRY_NORM]
        name = row[NORM_COL_NAME_CORE]
        post = row[NORM_COL_POSTAL_PREFIX][:3] if row[NORM_COL_POSTAL_PREFIX] else ""
        hno = row[NORM_COL_HOUSE_NUMBER]

        tokens = [t for t in name.split() if len(t) >= 3]
        matched_cands = set()

        for t in set(tokens):
            freq = cand_token_freq[t]
            if freq > max_block_size:
                if post:
                    key = (country, t, "post", post)
                    matched_cands.update(cand_index.get(key, []))
                if hno:
                    key = (country, t, "hno", hno)
                    matched_cands.update(cand_index.get(key, []))
                if len(tokens) >= 2:
                    key = (country, tuple(sorted(tokens[:2])))
                    matched_cands.update(cand_index.get(key, []))
            else:
                key = (country, t)
                matched_cands.update(cand_index.get(key, []))

        for cid in matched_cands:
            pairs.append((s1_id, cid, 0.80))

    if not pairs:
        return pd.DataFrame(columns=[CAND_COL_SOURCE1_ENTITY_ID, CAND_COL_CANDIDATE_ENTITY_ID, "raw_score"])

    df_pairs = pd.DataFrame(pairs, columns=[CAND_COL_SOURCE1_ENTITY_ID, CAND_COL_CANDIDATE_ENTITY_ID, "raw_score"])
    return df_pairs.drop_duplicates(subset=[CAND_COL_SOURCE1_ENTITY_ID, CAND_COL_CANDIDATE_ENTITY_ID])


# ---------------------------------------------------------------------------
# 2. SORTED NEIGHBORHOOD METHOD (SNM)
# ---------------------------------------------------------------------------

def snm_block(
    source1_norm: pd.DataFrame,
    candidates_norm: pd.DataFrame,
    window_size: int = SNM_WINDOW_SIZE,
) -> pd.DataFrame:
    """
    Sorted-Neighbourhood Method blocking.

    Sorts combined records alphabetically on (country_norm, name_core),
    then slides a window of size *window_size* and emits all cross-source pairs.

    Parameters
    ----------
    source1_norm : pd.DataFrame
        Normalized Source 1 records.
    candidates_norm : pd.DataFrame
        Normalized candidate records.
    window_size : int, default SNM_WINDOW_SIZE
        Number of adjacent positions to search.

    Returns
    -------
    pd.DataFrame
        Columns: source1_entity_id, candidate_entity_id, raw_score.
    """
    s1_df = source1_norm[[NORM_COL_ENTITY_ID, NORM_COL_COUNTRY_NORM, NORM_COL_NAME_CORE]].copy()
    s1_df["_source"] = "S1"

    cand_df = candidates_norm[[NORM_COL_ENTITY_ID, NORM_COL_COUNTRY_NORM, NORM_COL_NAME_CORE]].copy()
    cand_df["_source"] = "CAND"

    combined = pd.concat([s1_df, cand_df], ignore_index=True)
    combined = combined.sort_values(by=[NORM_COL_COUNTRY_NORM, NORM_COL_NAME_CORE]).reset_index(drop=True)

    pairs = []
    n = len(combined)
    records = combined[[NORM_COL_ENTITY_ID, NORM_COL_COUNTRY_NORM, "_source"]].values

    for i in range(n):
        src_i, country_i, type_i = records[i]
        if type_i != "S1":
            continue

        start = max(0, i - window_size)
        end = min(n, i + window_size + 1)

        for j in range(start, end):
            if i == j:
                continue
            src_j, country_j, type_j = records[j]
            if country_i == country_j and type_j == "CAND":
                dist = abs(i - j)
                score = round(1.0 - (dist / (window_size + 1)), 4)
                pairs.append((src_i, src_j, score))

    if not pairs:
        return pd.DataFrame(columns=[CAND_COL_SOURCE1_ENTITY_ID, CAND_COL_CANDIDATE_ENTITY_ID, "raw_score"])

    df_pairs = pd.DataFrame(pairs, columns=[CAND_COL_SOURCE1_ENTITY_ID, CAND_COL_CANDIDATE_ENTITY_ID, "raw_score"])
    return df_pairs.drop_duplicates(subset=[CAND_COL_SOURCE1_ENTITY_ID, CAND_COL_CANDIDATE_ENTITY_ID])


# ---------------------------------------------------------------------------
# 3 & 4. TF-IDF + kNN BLOCKING (CHAR & WORD)
# ---------------------------------------------------------------------------

def tfidf_knn_block(
    source1_norm: pd.DataFrame,
    candidates_norm: pd.DataFrame,
    ngram_range: tuple[int, int] = TFIDF_NGRAM_RANGE_CHAR,
    analyzer: str = "char",
    top_k: int = TOP_K_CANDIDATES,
) -> pd.DataFrame:
    """
    TF-IDF vectorisation + approximate k-nearest-neighbours blocking.

    Vectorizes normalized fields (combines name and address for character n-grams
    to capture script/OCR variations, or core name for word n-grams) and queries
    cosine kNN per country partition.

    Parameters
    ----------
    source1_norm : pd.DataFrame
        Normalized Source 1 records.
    candidates_norm : pd.DataFrame
        Normalized candidate records.
    ngram_range : tuple[int, int], default TFIDF_NGRAM_RANGE_CHAR
        N-gram min and max length.
    analyzer : str, default "char"
        "char" or "word".
    top_k : int, default TOP_K_CANDIDATES
        Number of nearest neighbors to retrieve per S1 query entity.

    Returns
    -------
    pd.DataFrame
        Columns: source1_entity_id, candidate_entity_id, raw_score.
    """
    pairs = []
    countries = set(source1_norm[NORM_COL_COUNTRY_NORM].unique())

    for country in countries:
        s1_sub = source1_norm[source1_norm[NORM_COL_COUNTRY_NORM] == country].reset_index(drop=True)
        cand_sub = candidates_norm[candidates_norm[NORM_COL_COUNTRY_NORM] == country].reset_index(drop=True)

        if len(s1_sub) == 0 or len(cand_sub) == 0:
            continue

        if analyzer == "char":
            s1_texts = (s1_sub[NORM_COL_NAME_EXPANDED] + " " + s1_sub[NORM_COL_ADDRESS_EXPANDED]).fillna("").tolist()
            cand_texts = (cand_sub[NORM_COL_NAME_EXPANDED] + " " + cand_sub[NORM_COL_ADDRESS_EXPANDED]).fillna("").tolist()
        else:
            s1_texts = s1_sub[NORM_COL_NAME_CORE].fillna("").tolist()
            cand_texts = cand_sub[NORM_COL_NAME_CORE].fillna("").tolist()

        vec = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=1,
            max_features=50000,
            sublinear_tf=True
        )

        all_texts = s1_texts + cand_texts
        vec.fit(all_texts)

        s1_mat = vec.transform(s1_texts)
        cand_mat = vec.transform(cand_texts)

        k = min(top_k, cand_mat.shape[0])
        nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=-1)
        nn.fit(cand_mat)

        distances, indices = nn.kneighbors(s1_mat)

        s1_ids = s1_sub[NORM_COL_ENTITY_ID].values
        cand_ids = cand_sub[NORM_COL_ENTITY_ID].values

        for row_i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
            s1_id = s1_ids[row_i]
            for d, cand_idx in zip(dist_row, idx_row):
                cand_id = cand_ids[cand_idx]
                sim = round(max(0.0, 1.0 - float(d)), 4)
                pairs.append((s1_id, cand_id, sim))

    if not pairs:
        return pd.DataFrame(columns=[CAND_COL_SOURCE1_ENTITY_ID, CAND_COL_CANDIDATE_ENTITY_ID, "raw_score"])

    df_pairs = pd.DataFrame(pairs, columns=[CAND_COL_SOURCE1_ENTITY_ID, CAND_COL_CANDIDATE_ENTITY_ID, "raw_score"])
    return df_pairs.drop_duplicates(subset=[CAND_COL_SOURCE1_ENTITY_ID, CAND_COL_CANDIDATE_ENTITY_ID])


# ---------------------------------------------------------------------------
# 5. ADDRESS TOKEN BLOCKING
# ---------------------------------------------------------------------------

def addr_token_block(
    source1_norm: pd.DataFrame,
    candidates_norm: pd.DataFrame,
    max_block_size: int = 50,
) -> pd.DataFrame:
    """
    Inverted-index blocking on address keys:
    (country, postal_prefix, house_number) with landmark fallback.

    Catches entities with significant name variation/rebranding but identical physical address.

    Parameters
    ----------
    source1_norm : pd.DataFrame
        Normalized Source 1 records.
    candidates_norm : pd.DataFrame
        Normalized candidate records.
    max_block_size : int, default 50
        Block-size cap per address/landmark key.

    Returns
    -------
    pd.DataFrame
        Columns: source1_entity_id, candidate_entity_id, raw_score.
    """
    cand_index = defaultdict(list)
    for idx, row in candidates_norm.iterrows():
        cid = row[NORM_COL_ENTITY_ID]
        country = row[NORM_COL_COUNTRY_NORM]
        post = row[NORM_COL_POSTAL_PREFIX]
        hno = row[NORM_COL_HOUSE_NUMBER]
        lmk = row[NORM_COL_LANDMARK]

        # Primary address key
        if post and hno:
            cand_index[(country, "post_hno", post, hno)].append(cid)
        elif hno:
            cand_index[(country, "hno", hno)].append(cid)

        # Landmark fallback key for addresses without house numbers
        if lmk:
            lmk_tokens = [t for t in lmk.split() if len(t) >= 4]
            if len(lmk_tokens) >= 2:
                cand_index[(country, "lmk", tuple(sorted(lmk_tokens[:2])))].append(cid)
            elif lmk_tokens:
                cand_index[(country, "lmk", lmk_tokens[0])].append(cid)

    pairs = []
    for idx, row in source1_norm.iterrows():
        s1_id = row[NORM_COL_ENTITY_ID]
        country = row[NORM_COL_COUNTRY_NORM]
        post = row[NORM_COL_POSTAL_PREFIX]
        hno = row[NORM_COL_HOUSE_NUMBER]
        lmk = row[NORM_COL_LANDMARK]

        matched_cands = set()
        if post and hno:
            matched_cands.update(cand_index.get((country, "post_hno", post, hno), []))
        elif hno:
            matched_cands.update(cand_index.get((country, "hno", hno), []))

        if lmk:
            lmk_tokens = [t for t in lmk.split() if len(t) >= 4]
            if len(lmk_tokens) >= 2:
                matched_cands.update(cand_index.get((country, "lmk", tuple(sorted(lmk_tokens[:2]))), []))
            elif lmk_tokens:
                matched_cands.update(cand_index.get((country, "lmk", lmk_tokens[0]), []))

        if len(matched_cands) > max_block_size:
            matched_cands = set(list(matched_cands)[:max_block_size])

        for cid in matched_cands:
            pairs.append((s1_id, cid, 0.85))

    if not pairs:
        return pd.DataFrame(columns=[CAND_COL_SOURCE1_ENTITY_ID, CAND_COL_CANDIDATE_ENTITY_ID, "raw_score"])

    df_pairs = pd.DataFrame(pairs, columns=[CAND_COL_SOURCE1_ENTITY_ID, CAND_COL_CANDIDATE_ENTITY_ID, "raw_score"])
    return df_pairs.drop_duplicates(subset=[CAND_COL_SOURCE1_ENTITY_ID, CAND_COL_CANDIDATE_ENTITY_ID])


# ---------------------------------------------------------------------------
# 6. MINHASH / LSH BLOCKING
# ---------------------------------------------------------------------------

def minhash_lsh_block(
    source1_norm: pd.DataFrame,
    candidates_norm: pd.DataFrame,
    num_perm: int = 64,
    num_bands: int = 16,
    top_k: int = TOP_K_CANDIDATES,
) -> pd.DataFrame:
    """
    MinHash / LSH blocking for Jaccard similarity across large corpora.

    Used as an efficient sub-linear time candidate generator when
    len(candidates_norm) >= CANDIDATE_SIZE_THRESHOLD.

    Parameters
    ----------
    source1_norm : pd.DataFrame
        Normalized Source 1 records.
    candidates_norm : pd.DataFrame
        Normalized candidate records.
    num_perm : int, default 64
        Number of hash permutations.
    num_bands : int, default 16
        Number of LSH bands (rows per band = num_perm // num_bands).
    top_k : int, default TOP_K_CANDIDATES
        Top candidates to return per S1 entity.

    Returns
    -------
    pd.DataFrame
        Columns: source1_entity_id, candidate_entity_id, raw_score.
    """
    rows_per_band = num_perm // num_bands
    rng = np.random.RandomState(42)

    p = 2**31 - 1
    a = rng.randint(1, p, size=num_perm, dtype=np.int64)
    b = rng.randint(0, p, size=num_perm, dtype=np.int64)

    def _compute_minhash(text: str) -> np.ndarray:
        if not text:
            return np.zeros(num_perm, dtype=np.int64)
        shingles = [text[i:i+3] for i in range(max(1, len(text)-2))]
        if not shingles:
            shingles = [text]
        shingle_hashes = np.array([abs(hash(s)) % p for s in shingles], dtype=np.int64)
        hashes = (np.outer(a, shingle_hashes) + b[:, np.newaxis]) % p
        return np.min(hashes, axis=1)

    pairs = []
    countries = set(source1_norm[NORM_COL_COUNTRY_NORM].unique())

    for country in countries:
        s1_sub = source1_norm[source1_norm[NORM_COL_COUNTRY_NORM] == country].reset_index(drop=True)
        cand_sub = candidates_norm[candidates_norm[NORM_COL_COUNTRY_NORM] == country].reset_index(drop=True)

        if len(s1_sub) == 0 or len(cand_sub) == 0:
            continue

        cand_buckets = [defaultdict(list) for _ in range(num_bands)]
        for idx, row in cand_sub.iterrows():
            sig = _compute_minhash(row[NORM_COL_NAME_EXPANDED])
            for band_idx in range(num_bands):
                start = band_idx * rows_per_band
                end = start + rows_per_band
                band_key = tuple(sig[start:end])
                cand_buckets[band_idx][band_key].append(idx)

        for idx, row in s1_sub.iterrows():
            s1_id = row[NORM_COL_ENTITY_ID]
            sig = _compute_minhash(row[NORM_COL_NAME_EXPANDED])
            cand_matches: Counter[int] = Counter()

            for band_idx in range(num_bands):
                start = band_idx * rows_per_band
                end = start + rows_per_band
                band_key = tuple(sig[start:end])
                for cand_idx in cand_buckets[band_idx].get(band_key, []):
                    cand_matches[cand_idx] += 1

            for cand_idx, band_count in cand_matches.most_common(top_k):
                cid = cand_sub.at[cand_idx, NORM_COL_ENTITY_ID]
                est_jaccard = round(band_count / num_bands, 4)
                pairs.append((s1_id, cid, est_jaccard))

    if not pairs:
        return pd.DataFrame(columns=[CAND_COL_SOURCE1_ENTITY_ID, CAND_COL_CANDIDATE_ENTITY_ID, "raw_score"])

    df_pairs = pd.DataFrame(pairs, columns=[CAND_COL_SOURCE1_ENTITY_ID, CAND_COL_CANDIDATE_ENTITY_ID, "raw_score"])
    return df_pairs.drop_duplicates(subset=[CAND_COL_SOURCE1_ENTITY_ID, CAND_COL_CANDIDATE_ENTITY_ID])


# ---------------------------------------------------------------------------
# MERGE STRATEGY OUTPUTS & ORCHESTRATION
# ---------------------------------------------------------------------------

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
    """
    pair_map = defaultdict(lambda: {"flags": [], "scores": []})

    for strategy_name, df_strat in strategy_dfs.items():
        if df_strat is None or len(df_strat) == 0:
            continue
        for idx, row in df_strat.iterrows():
            s1 = row[CAND_COL_SOURCE1_ENTITY_ID]
            cand = row[CAND_COL_CANDIDATE_ENTITY_ID]
            score = float(row["raw_score"]) if "raw_score" in row else 1.0
            pair_map[(s1, cand)]["flags"].append(strategy_name)
            pair_map[(s1, cand)]["scores"].append(score)

    records = []
    for (s1, cand), data in pair_map.items():
        flags_str = ",".join(sorted(set(data["flags"])))
        best_score = max(data["scores"])
        records.append({
            CAND_COL_SOURCE1_ENTITY_ID: s1,
            CAND_COL_CANDIDATE_ENTITY_ID: cand,
            CAND_COL_BLOCKING_STRATEGY_FLAGS: flags_str,
            CAND_COL_BLOCKING_SCORE: round(best_score, 4),
        })

    if not records:
        return pd.DataFrame(columns=list(CANDIDATE_PAIR_COLUMNS))

    df_out = pd.DataFrame(records)
    return df_out[list(CANDIDATE_PAIR_COLUMNS)]


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
    source3_norm : pd.DataFrame | None, optional
        Normalised records from source3, if available.

    Returns
    -------
    pd.DataFrame
        Columns: source1_entity_id, candidate_entity_id,
                 blocking_strategy_flags, blocking_score.
    """
    if source3_norm is not None:
        candidates_norm = pd.concat([source2_norm, source3_norm], ignore_index=True)
    else:
        candidates_norm = source2_norm.copy()

    strategy_dfs = {}

    # 1. Name token blocking with address conjunction & guardrails
    strategy_dfs["name_token"] = name_token_block(source1_norm, candidates_norm)

    # 2. Sorted Neighbourhood Method
    strategy_dfs["snm"] = snm_block(source1_norm, candidates_norm, window_size=SNM_WINDOW_SIZE)

    # 3. Address token blocking with landmark fallback
    strategy_dfs["addr_token"] = addr_token_block(source1_norm, candidates_norm)

    # 4 & 5 vs 6. TF-IDF + kNN vs MinHash / LSH Gating
    cand_size = len(candidates_norm)
    if cand_size < CANDIDATE_SIZE_THRESHOLD:
        strategy_dfs["char_tfidf"] = tfidf_knn_block(
            source1_norm, candidates_norm, ngram_range=TFIDF_NGRAM_RANGE_CHAR, analyzer="char"
        )
        strategy_dfs["word_tfidf"] = tfidf_knn_block(
            source1_norm, candidates_norm, ngram_range=TFIDF_NGRAM_RANGE_WORD, analyzer="word"
        )
    else:
        strategy_dfs["minhash"] = minhash_lsh_block(
            source1_norm, candidates_norm, num_perm=64, num_bands=16
        )

    # Merge all generated candidate pairs
    return merge_strategy_outputs(strategy_dfs)


# ---------------------------------------------------------------------------
# EVALUATION METRICS
# ---------------------------------------------------------------------------

def evaluate_blocking(
    candidate_pairs: pd.DataFrame,
    ground_truth: pd.DataFrame,
    total_s1_count: int | None = None,
    total_cand_count: int | None = None,
) -> dict[str, float]:
    """
    Compute blocking recall and reduction ratio.

    Parameters
    ----------
    candidate_pairs : pd.DataFrame
        Schema 3 candidate pairs DataFrame.
    ground_truth : pd.DataFrame
        Ground truth DataFrame.
    total_s1_count : int, optional
        Total query entity count in universe.
    total_cand_count : int, optional
        Total candidate entity count in universe.

    Returns
    -------
    dict[str, float]
        Keys: "recall", "reduction_ratio", "true_pairs_found",
              "total_true_pairs", "candidate_pairs_count".
    """
    true_pairs = set()
    for idx, row in ground_truth.iterrows():
        s1 = str(row[GT_COL_SOURCE1_ENTITY_ID]).strip()
        mids = str(row[GT_COL_MATCHED_ENTITY_IDS]).strip()
        if not mids:
            continue
        delimiter = "," if "," in mids else "|"
        for mid in mids.split(delimiter):
            mid_clean = mid.strip()
            if mid_clean:
                true_pairs.add((s1, mid_clean))

    total_true_pairs = len(true_pairs)
    if total_true_pairs == 0:
        return {
            "recall": 1.0,
            "reduction_ratio": 1.0,
            "true_pairs_found": 0,
            "total_true_pairs": 0,
            "candidate_pairs_count": len(candidate_pairs),
        }

    cand_pairs_set = set(
        zip(candidate_pairs[CAND_COL_SOURCE1_ENTITY_ID], candidate_pairs[CAND_COL_CANDIDATE_ENTITY_ID])
    )

    found_true_pairs = len(true_pairs.intersection(cand_pairs_set))
    recall = found_true_pairs / total_true_pairs

    n_s1 = total_s1_count or ground_truth[GT_COL_SOURCE1_ENTITY_ID].nunique()
    n_cand = total_cand_count or 10_320_219
    total_universe = n_s1 * n_cand
    reduction_ratio = 1.0 - (len(cand_pairs_set) / total_universe) if total_universe > 0 else 1.0

    return {
        "recall": round(recall, 4),
        "reduction_ratio": round(reduction_ratio, 6),
        "true_pairs_found": found_true_pairs,
        "total_true_pairs": total_true_pairs,
        "candidate_pairs_count": len(cand_pairs_set),
    }
