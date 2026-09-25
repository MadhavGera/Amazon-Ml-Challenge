"""
config.py — Single source of truth for the business entity resolution pipeline.

USAGE
-----
All other modules must import constants from here — no hardcoded paths, column
names, or hyper-parameters anywhere downstream.

To move the project between machines, change only PROJECT_ROOT below.
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# ROOT — change ONLY this line when moving the project between machines
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# ─────────────────────────────────────────────────────────────────────────────
# DIRECTORY PATHS
# ─────────────────────────────────────────────────────────────────────────────
DATASET_DIR: Path = PROJECT_ROOT / "dataset"
TRAIN_DIR: Path = DATASET_DIR / "train"
TEST_DIR: Path = DATASET_DIR / "test"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
EXPERIMENTS_DIR: Path = PROJECT_ROOT / "experiments"
DOCS_DIR: Path = PROJECT_ROOT / "docs"

# ─────────────────────────────────────────────────────────────────────────────
# INPUT FILE NAMES  (actual files are placed by the user, not generated)
# ─────────────────────────────────────────────────────────────────────────────
# Training source files
TRAIN_SOURCE1_FILE: str = "train_source1.tsv"
TRAIN_SOURCE2_FILE: str = "train_source2.tsv"
TRAIN_SOURCE3_FILE: str = "train_source3.tsv"
TRAIN_GROUND_TRUTH_FILE: str = "train_ground_truth.tsv"

# Test source files
TEST_SOURCE1_FILE: str = "test_source1.tsv"
TEST_SOURCE2_FILE: str = "test_source2.tsv"
TEST_SOURCE3_FILE: str = "test_source3.tsv"

# Full paths (derived — never hardcoded)
TRAIN_SOURCE1_PATH: Path = TRAIN_DIR / TRAIN_SOURCE1_FILE
TRAIN_SOURCE2_PATH: Path = TRAIN_DIR / TRAIN_SOURCE2_FILE
TRAIN_SOURCE3_PATH: Path = TRAIN_DIR / TRAIN_SOURCE3_FILE
TRAIN_GROUND_TRUTH_PATH: Path = TRAIN_DIR / TRAIN_GROUND_TRUTH_FILE

TEST_SOURCE1_PATH: Path = TEST_DIR / TEST_SOURCE1_FILE
TEST_SOURCE2_PATH: Path = TEST_DIR / TEST_SOURCE2_FILE
TEST_SOURCE3_PATH: Path = TEST_DIR / TEST_SOURCE3_FILE

# Output paths
OUTPUT_MODEL_PATH: Path = OUTPUT_DIR / "model.txt"
OUTPUT_MATCHING_RESULTS_PATH: Path = OUTPUT_DIR / "matching_results.tsv"
OUTPUT_CANDIDATE_PAIRS_PATH: Path = OUTPUT_DIR / "candidate_pairs.tsv"

# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA 1 — Raw source file columns
# (entity_id, business_name, business_address, country)
# These must match the actual TSV headers exactly.
# ─────────────────────────────────────────────────────────────────────────────
RAW_COL_ENTITY_ID: str = "entity_id"
RAW_COL_BUSINESS_NAME: str = "business_name"
RAW_COL_BUSINESS_ADDRESS: str = "business_address"
RAW_COL_COUNTRY: str = "country"

# Ordered tuple — used for column-presence assertions in data_io.py
RAW_SOURCE_COLUMNS: tuple[str, ...] = (
    RAW_COL_ENTITY_ID,
    RAW_COL_BUSINESS_NAME,
    RAW_COL_BUSINESS_ADDRESS,
    RAW_COL_COUNTRY,
)

# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA 1 — Ground-truth file columns
# ─────────────────────────────────────────────────────────────────────────────
GT_COL_SOURCE1_ENTITY_ID: str = "source1_entity_id"
GT_COL_MATCHED_ENTITY_IDS: str = "matched_entity_ids"

GROUND_TRUTH_COLUMNS: tuple[str, ...] = (
    GT_COL_SOURCE1_ENTITY_ID,
    GT_COL_MATCHED_ENTITY_IDS,
)

# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA 2 — Normalized record (output of normalize.py → input to blocking.py)
#
# Column          dtype   Description
# --------------- ------- -------------------------------------------------------
# entity_id       str     Original entity identifier from the source file
# raw_name        str     Unmodified business_name from source
# name_expanded   str     Name after abbreviation expansion (e.g. "St" → "Street")
# name_core       str     Name after stopword/legal-suffix removal (e.g. "LLC")
# raw_address     str     Unmodified business_address from source
# address_expanded str    Address after abbreviation expansion
# house_number    str     Extracted leading numeric/alphanumeric house number token
# postal_prefix   str     Extracted postal/ZIP code prefix (first N chars)
# landmark        str     Extracted landmark token if present (e.g. mall, plaza)
# country_norm    str     ISO 3166-1 alpha-2 code inferred/validated from raw country
# ─────────────────────────────────────────────────────────────────────────────
NORM_COL_ENTITY_ID: str = "entity_id"
NORM_COL_RAW_NAME: str = "raw_name"
NORM_COL_NAME_EXPANDED: str = "name_expanded"
NORM_COL_NAME_CORE: str = "name_core"
NORM_COL_RAW_ADDRESS: str = "raw_address"
NORM_COL_ADDRESS_EXPANDED: str = "address_expanded"
NORM_COL_HOUSE_NUMBER: str = "house_number"
NORM_COL_POSTAL_PREFIX: str = "postal_prefix"
NORM_COL_LANDMARK: str = "landmark"
NORM_COL_COUNTRY_NORM: str = "country_norm"

NORMALIZED_COLUMNS: tuple[str, ...] = (
    NORM_COL_ENTITY_ID,
    NORM_COL_RAW_NAME,
    NORM_COL_NAME_EXPANDED,
    NORM_COL_NAME_CORE,
    NORM_COL_RAW_ADDRESS,
    NORM_COL_ADDRESS_EXPANDED,
    NORM_COL_HOUSE_NUMBER,
    NORM_COL_POSTAL_PREFIX,
    NORM_COL_LANDMARK,
    NORM_COL_COUNTRY_NORM,
)

# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA 3 — Candidate pairs (output of blocking.py → input to features.py)
#
# Column                    dtype   Description
# ------------------------- ------- -------------------------------------------
# source1_entity_id         str     Entity ID from source1
# candidate_entity_id       str     Entity ID from source2/source3 candidate
# blocking_strategy_flags   str     Comma-separated strategy names that nominated
#                                   this pair (e.g. "name_token,snm")
# blocking_score            float   Best similarity score across contributing
#                                   strategies (higher = more similar)
# ─────────────────────────────────────────────────────────────────────────────
CAND_COL_SOURCE1_ENTITY_ID: str = "source1_entity_id"
CAND_COL_CANDIDATE_ENTITY_ID: str = "candidate_entity_id"
CAND_COL_BLOCKING_STRATEGY_FLAGS: str = "blocking_strategy_flags"
CAND_COL_BLOCKING_SCORE: str = "blocking_score"

CANDIDATE_PAIR_COLUMNS: tuple[str, ...] = (
    CAND_COL_SOURCE1_ENTITY_ID,
    CAND_COL_CANDIDATE_ENTITY_ID,
    CAND_COL_BLOCKING_STRATEGY_FLAGS,
    CAND_COL_BLOCKING_SCORE,
)

# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA 4 — Feature table (output of features.py → input to train.py/predict.py)
#
# All feature columns listed up front so model code can be written against
# final column names before feature engineering is implemented.
# ─────────────────────────────────────────────────────────────────────────────
FEAT_COL_SOURCE1_ENTITY_ID: str = "source1_entity_id"
FEAT_COL_CANDIDATE_ENTITY_ID: str = "candidate_entity_id"

# Name similarity features
FEAT_COL_LEV_DIST_RAW: str = "lev_dist_raw"
FEAT_COL_JARO_WINKLER_RAW: str = "jaro_winkler_raw"
FEAT_COL_TOKEN_SORT_RATIO_NORM: str = "token_sort_ratio_norm"
FEAT_COL_TOKEN_SET_RATIO_NORM: str = "token_set_ratio_norm"
FEAT_COL_TFIDF_COSINE_CHAR: str = "tfidf_cosine_char"
FEAT_COL_TFIDF_COSINE_WORD: str = "tfidf_cosine_word"
FEAT_COL_JACCARD_TOKENS_FULL: str = "jaccard_tokens_full"
FEAT_COL_JACCARD_TOKENS_CORE: str = "jaccard_tokens_core"
FEAT_COL_FIRST_TOKEN_MATCH: str = "first_token_match"
FEAT_COL_NAME_LEN_DIFF_RATIO: str = "name_len_diff_ratio"

# Address similarity features
FEAT_COL_ADDR_LEV_DIST: str = "addr_lev_dist"
FEAT_COL_ADDR_JARO_WINKLER: str = "addr_jaro_winkler"
FEAT_COL_ADDR_TOKEN_SORT_RATIO: str = "addr_token_sort_ratio"
FEAT_COL_ADDR_TOKEN_SET_RATIO: str = "addr_token_set_ratio"
FEAT_COL_ADDR_JACCARD: str = "addr_jaccard"
FEAT_COL_HOUSE_NUMBER_MATCH: str = "house_number_match"
FEAT_COL_POSTAL_PREFIX_MATCH: str = "postal_prefix_match"
FEAT_COL_CITY_TOKEN_OVERLAP: str = "city_token_overlap"

# Structural / meta features
FEAT_COL_LANDMARK_FLAG: str = "landmark_flag"
FEAT_COL_SAME_COUNTRY_FLAG: str = "same_country_flag"
FEAT_COL_CANDIDATE_SCORE_GAP: str = "candidate_score_gap"

# Blocking provenance features (binary flags)
FEAT_COL_BLOCKED_BY_NAME_TOKEN: str = "blocked_by_name_token"
FEAT_COL_BLOCKED_BY_SNM: str = "blocked_by_snm"
FEAT_COL_BLOCKED_BY_CHAR_TFIDF: str = "blocked_by_char_tfidf"
FEAT_COL_BLOCKED_BY_WORD_TFIDF: str = "blocked_by_word_tfidf"
FEAT_COL_BLOCKED_BY_ADDR_TOKEN: str = "blocked_by_addr_token"
FEAT_COL_BLOCKED_BY_MINHASH: str = "blocked_by_minhash"

# Ordered list — used as `df[FEATURE_COLUMNS]` in train.py / predict.py
FEATURE_COLUMNS: tuple[str, ...] = (
    FEAT_COL_LEV_DIST_RAW,
    FEAT_COL_JARO_WINKLER_RAW,
    FEAT_COL_TOKEN_SORT_RATIO_NORM,
    FEAT_COL_TOKEN_SET_RATIO_NORM,
    FEAT_COL_TFIDF_COSINE_CHAR,
    FEAT_COL_TFIDF_COSINE_WORD,
    FEAT_COL_JACCARD_TOKENS_FULL,
    FEAT_COL_JACCARD_TOKENS_CORE,
    FEAT_COL_FIRST_TOKEN_MATCH,
    FEAT_COL_NAME_LEN_DIFF_RATIO,
    FEAT_COL_ADDR_LEV_DIST,
    FEAT_COL_ADDR_JARO_WINKLER,
    FEAT_COL_ADDR_TOKEN_SORT_RATIO,
    FEAT_COL_ADDR_TOKEN_SET_RATIO,
    FEAT_COL_ADDR_JACCARD,
    FEAT_COL_HOUSE_NUMBER_MATCH,
    FEAT_COL_POSTAL_PREFIX_MATCH,
    FEAT_COL_CITY_TOKEN_OVERLAP,
    FEAT_COL_LANDMARK_FLAG,
    FEAT_COL_SAME_COUNTRY_FLAG,
    FEAT_COL_CANDIDATE_SCORE_GAP,
    FEAT_COL_BLOCKED_BY_NAME_TOKEN,
    FEAT_COL_BLOCKED_BY_SNM,
    FEAT_COL_BLOCKED_BY_CHAR_TFIDF,
    FEAT_COL_BLOCKED_BY_WORD_TFIDF,
    FEAT_COL_BLOCKED_BY_ADDR_TOKEN,
    FEAT_COL_BLOCKED_BY_MINHASH,
)

# Full feature-table column order (IDs first, then features)
FEATURE_TABLE_COLUMNS: tuple[str, ...] = (
    FEAT_COL_SOURCE1_ENTITY_ID,
    FEAT_COL_CANDIDATE_ENTITY_ID,
    *FEATURE_COLUMNS,
)

# ─────────────────────────────────────────────────────────────────────────────
# SUBMISSION FORMAT SCHEMAS  (fixed by the challenge — do not modify)
# ─────────────────────────────────────────────────────────────────────────────
# matching_results.tsv  →  source1_entity_id | matched_entity_ids
# candidate_pairs.tsv   →  source1_entity_id | candidate_entity_ids
SUBMISSION_MATCHING_COLS: tuple[str, ...] = ("source1_entity_id", "matched_entity_ids")
SUBMISSION_CANDIDATE_COLS: tuple[str, ...] = ("source1_entity_id", "candidate_entity_ids")

# ─────────────────────────────────────────────────────────────────────────────
# BLOCKING TUNABLES
# ─────────────────────────────────────────────────────────────────────────────
# Gate for choosing blocking strategy:
#   candidate_set_size < CANDIDATE_SIZE_THRESHOLD  → TF-IDF + kNN
#   candidate_set_size >= CANDIDATE_SIZE_THRESHOLD → MinHash / LSH fallback
# (Logic implemented in Phase 3 — only the constant is frozen here.)
CANDIDATE_SIZE_THRESHOLD: int = 500_000

# Number of nearest-neighbour candidates to retrieve per source1 entity
# Tune in Phase 3
TOP_K_CANDIDATES: int = 10

# Sorted-Neighbourhood Method window size
# Tune in Phase 3
SNM_WINDOW_SIZE: int = 5

# Character n-gram range for TF-IDF vectoriser
# Tune in Phase 3
TFIDF_NGRAM_RANGE_CHAR: tuple[int, int] = (2, 4)

# Word n-gram range for TF-IDF vectoriser
# Tune in Phase 3
TFIDF_NGRAM_RANGE_WORD: tuple[int, int] = (1, 2)

# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION / POST-PROCESSING TUNABLES
# ─────────────────────────────────────────────────────────────────────────────
# Probability threshold above which a candidate pair is labelled a match.
# Set to None until Phase 5 (threshold search on validation set).
MATCH_THRESHOLD: float | None = None

# Minimum probability margin between top-1 and top-2 candidates required to
# emit a positive match (used in Phase 6 for confidence-gated predictions).
# Set to None until Phase 6.
MARGIN_THRESHOLD: float | None = None
