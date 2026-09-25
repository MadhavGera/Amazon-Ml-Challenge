# Data Schemas — Business Entity Resolution Pipeline

> **⚠ FROZEN as of Phase 0. Any schema change requires both teammates to agree
> and must be updated here AND in `src/config.py` simultaneously.**

---

## Schema 1 — Raw source file (`train_source*.tsv`, `test_source*.tsv`)

These columns are dictated by the challenge dataset. Do **not** rename them.

| Column | dtype | Description |
|---|---|---|
| `entity_id` | `str` | Unique identifier for a business record within its source file. |
| `business_name` | `str` | Raw, unmodified business name as provided by the data source. |
| `business_address` | `str` | Raw, unmodified business address string. May contain noise, abbreviations, or missing parts. |
| `country` | `str` | Country field as provided; may be a full name, code, or mixed. |

---

## Schema 1b — Ground-truth file (`train_ground_truth.tsv`)

| Column | dtype | Description |
|---|---|---|
| `source1_entity_id` | `str` | Entity ID from source1 (the "query" side). |
| `matched_entity_ids` | `str` | Pipe-separated (`\|`) or comma-separated list of matching entity IDs from source2/source3. |

---

## Schema 2 — Normalized record

**Producer:** `normalize.py`  
**Consumer:** `blocking.py`

This is the internal canonical representation of every record after cleaning. All
downstream stages operate on this schema — never on raw source columns directly.

| Column | dtype | Description |
|---|---|---|
| `entity_id` | `str` | Carried through unchanged from the raw source file. |
| `raw_name` | `str` | Verbatim copy of `business_name` — preserved for audit/error analysis. |
| `name_expanded` | `str` | Name after expanding common abbreviations (e.g. `"St."` → `"Street"`, `"Bros"` → `"Brothers"`). |
| `name_core` | `str` | Name after stripping legal suffixes (`LLC`, `Ltd`, `Inc`, `Pvt`) and high-frequency stopwords, leaving the meaningful name token(s). Used for high-precision blocking. |
| `raw_address` | `str` | Verbatim copy of `business_address`. |
| `address_expanded` | `str` | Address after abbreviation expansion and Unicode normalisation. |
| `house_number` | `str` | Leading numeric or alphanumeric house/plot number token extracted from the address. Empty string if not found. |
| `postal_prefix` | `str` | First 3–4 characters of the postal/ZIP code extracted from the address. Empty string if not found. |
| `landmark` | `str` | Landmark token if present in the address (e.g. mall name, plaza, sector). Empty string if not found. |
| `country_norm` | `str` | Normalised ISO 3166-1 alpha-2 country code inferred from the raw `country` field. Empty string if unresolvable. No country-specific logic may be hardcoded; use a lookup table. |

---

## Schema 3 — Candidate pairs

**Producer:** `blocking.py`  
**Consumer:** `features.py`

| Column | dtype | Description |
|---|---|---|
| `source1_entity_id` | `str` | Entity ID from source1 (the "query" entity). |
| `candidate_entity_id` | `str` | Entity ID from source2 or source3 that was nominated as a potential match. |
| `blocking_strategy_flags` | `str` | Comma-separated list of blocking strategy names that nominated this pair. Possible values: `name_token`, `snm`, `char_tfidf`, `word_tfidf`, `addr_token`, `minhash`. Example: `"name_token,snm"`. |
| `blocking_score` | `float` | Best raw similarity / proximity score across all contributing blocking strategies for this pair. Higher = more similar. Range depends on strategy (e.g. 0–1 for cosine; raw edit distance for SNM). |

---

## Schema 4 — Feature table

**Producer:** `features.py`  
**Consumer:** `train.py`, `predict.py`

All feature columns are listed here even before they are implemented, so both
teammates can write model/training code referencing final column names immediately.

### Key columns

| Column | dtype | Description |
|---|---|---|
| `source1_entity_id` | `str` | Query entity ID. |
| `candidate_entity_id` | `str` | Candidate entity ID. |

### Name similarity features

| Column | dtype | Description |
|---|---|---|
| `lev_dist_raw` | `float` | Normalised Levenshtein distance on `raw_name` (0 = identical). |
| `jaro_winkler_raw` | `float` | Jaro-Winkler similarity on `raw_name` (1 = identical). |
| `token_sort_ratio_norm` | `float` | RapidFuzz `token_sort_ratio` on `name_expanded`, normalised to [0, 1]. |
| `token_set_ratio_norm` | `float` | RapidFuzz `token_set_ratio` on `name_expanded`, normalised to [0, 1]. |
| `tfidf_cosine_char` | `float` | TF-IDF cosine similarity using character n-grams on `name_expanded`. |
| `tfidf_cosine_word` | `float` | TF-IDF cosine similarity using word n-grams on `name_expanded`. |
| `jaccard_tokens_full` | `float` | Jaccard similarity of token sets from `name_expanded`. |
| `jaccard_tokens_core` | `float` | Jaccard similarity of token sets from `name_core`. |
| `first_token_match` | `float` | Binary (0/1): first token of `name_core` matches between the pair. |
| `name_len_diff_ratio` | `float` | Absolute length difference divided by max length, on `name_expanded`. |

### Address similarity features

| Column | dtype | Description |
|---|---|---|
| `addr_lev_dist` | `float` | Normalised Levenshtein distance on `address_expanded`. |
| `addr_jaro_winkler` | `float` | Jaro-Winkler similarity on `address_expanded`. |
| `addr_token_sort_ratio` | `float` | RapidFuzz `token_sort_ratio` on `address_expanded`, normalised to [0, 1]. |
| `addr_token_set_ratio` | `float` | RapidFuzz `token_set_ratio` on `address_expanded`, normalised to [0, 1]. |
| `addr_jaccard` | `float` | Jaccard similarity of token sets from `address_expanded`. |
| `house_number_match` | `float` | Binary (0/1): `house_number` fields are non-empty and equal. |
| `postal_prefix_match` | `float` | Binary (0/1): `postal_prefix` fields are non-empty and equal. |
| `city_token_overlap` | `float` | Proportion of city-like tokens (heuristic) that overlap between addresses. |

### Structural / meta features

| Column | dtype | Description |
|---|---|---|
| `landmark_flag` | `float` | Binary (0/1): both records have a non-empty `landmark` token. |
| `same_country_flag` | `float` | Binary (0/1): `country_norm` is identical for both records. |
| `candidate_score_gap` | `float` | Gap between this candidate's `blocking_score` and the top-1 candidate's score for the same source1 entity. |

### Blocking provenance features (binary)

| Column | dtype | Description |
|---|---|---|
| `blocked_by_name_token` | `float` | 1 if the `name_token` strategy nominated this pair. |
| `blocked_by_snm` | `float` | 1 if the `snm` (Sorted-Neighbourhood Method) strategy nominated this pair. |
| `blocked_by_char_tfidf` | `float` | 1 if the `char_tfidf` strategy nominated this pair. |
| `blocked_by_word_tfidf` | `float` | 1 if the `word_tfidf` strategy nominated this pair. |
| `blocked_by_addr_token` | `float` | 1 if the `addr_token` strategy nominated this pair. |
| `blocked_by_minhash` | `float` | 1 if the `minhash` / LSH strategy nominated this pair. |

---

## Submission Format Schemas (challenge-fixed — do not modify)

### `matching_results.tsv`

| Column | dtype | Description |
|---|---|---|
| `source1_entity_id` | `str` | Query entity ID from source1. |
| `matched_entity_ids` | `str` | Pipe-separated list of matched entity IDs from source2/source3. |

### `candidate_pairs.tsv` (blocking submission)

| Column | dtype | Description |
|---|---|---|
| `source1_entity_id` | `str` | Query entity ID from source1. |
| `candidate_entity_ids` | `str` | Pipe-separated list of candidate entity IDs. |

---

*Machine-readable counterparts of all column names above live in `src/config.py`.*
