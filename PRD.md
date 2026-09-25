PRD + Phase-Wise Roadmap

# Business Entity Resolution — Amazon ML Challenge 2026


## 1. Problem & objective

Three independent, noisy sources (S1/S2/S3) describe overlapping sets of real-world businesses with no shared ID. S1 is the deduplicated reference set. For every S1 entity, predict all matching S2/S3 records — zero, one, or many — maximizing macro-averaged F0.5 per entity across the eval set.

`F0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)` — precision weighted 2× recall. Singletons score 1.0 for a correct empty prediction, 0.0 for any false merge.

## 2. Hard constraints

| Constraint | Detail |
| --- | --- |
| Format | TSV in/out, exact column names, one row per S1 test entity (empty allowed, missing row rejected) |
| ID hygiene | No self-matches to S1, no IDs outside the test set, no duplicates; every matched ID must also appear in `candidate_pairs.tsv` |
| Model license | MIT/Apache-2.0, ≤8B parameters |
| Fair play | No external lookups — registries, geocoding APIs, ER services. Disqualifying if detected |
| Country | Open string set. US/India in training; **France only at test time** — no hard-coded country logic anywhere |
| Submissions | 5/day leaderboard uploads, 15 total over 3 days — local validation must be trusted before spending one |
| Dataset size | Large — stays in a local `dataset/` folder, not committed/zipped; pipeline reads paths via config, not hardcoded |

## 3. Repo layout (for Antigravity)

```
business_entity_resolution/
├── dataset/                  # LOCAL ONLY — gitignored, large files live here
│   ├── train/  (train_source1/2/3.tsv, train_ground_truth.tsv)
│   └── test/   (test_source1/2/3.tsv)
├── src/
│   ├── config.py              # paths, column names, tunables — no hardcoded countries
│   ├── data_io.py              # load + normalize wrappers
│   ├── normalize.py            # name/address/country normalization
│   ├── blocking.py             # multi-strategy candidate generation + recall check
│   ├── features.py             # pairwise similarity features
│   ├── train.py                 # S1-level split, LightGBM, F0.5 threshold tuning
│   ├── predict.py               # test-set inference, dedup, output writers
│   ├── metrics.py               # exact macro F0.5 + diagnostics
│   ├── eda.py                   # Phase 1 exploration script
│   └── error_analysis.py        # Phase 7 false-positive/negative inspection
├── notebooks/                # exploratory only — logic graduates to src/
├── output/                    # matching_results.tsv, candidate_pairs.tsv, model.txt
├── experiments/                # one subfolder per submission — see §6
├── README.md
├── requirements.txt
└── Documentation_template.md   # filled in during Phase 8
```

`.gitignore` should exclude `dataset/` and `output/*.tsv` given the file sizes; keep `output/model.txt` and config artifacts if you want reproducibility without re-training.

## 4. Success metrics to track (not just leaderboard score)

| Metric | Why it matters |
| --- | --- |
| Candidate recall (per source, overall) | Hard ceiling on achievable F0.5 — measure before every model change |
| Candidate set size / reduction ratio | Too large slows feature/inference; too small kills recall |
| Local macro F0.5 (val split) | Primary optimization target, must correlate with leaderboard |
| Precision / recall (micro, diagnostic) | Tells you which lever to pull when F0.5 stalls |
| Singleton accuracy | Separately tracked — easy to silently regress while chasing recall |
| Cross-country val F0.5 (train-US→val-India and reverse) | Proxy for France generalization, which has zero training signal |

## 5. Phase-wise roadmap

Phase 0 — Environment & scaffolding \~2 hrs

- Set up repo structure above inside Antigravity; point `config.py` at the local `dataset/` folder
- Pin `requirements.txt` (pandas, numpy, scikit-learn, lightgbm, rapidfuzz, unidecode)
- Verify TSVs load correctly with `sep="\t"` and expected columns/row counts

Exit: all four training files and three test files load without errors; shapes and dtypes sanity-checked.

Phase 1 — EDA (do this before writing any model code)

- Per-source record counts, missingness in name/address, duplicate names/addresses
- Country distribution per source (confirm US/India only in training)
- Ground-truth match-count distribution per S1 (0 / 1 / many), overall singleton rate
- **Critical check:** does any S2/S3 record match more than one S1 entity? Decides whether the "one-S1-per-record" dedup rule is safe to use in Phase 6
- Sample and read \~30 true-positive pairs and \~15 near-miss non-matches by hand to catalog real noise patterns (not just the ones listed in the problem statement)
- **Mine the training corpus for common legal suffixes and address abbreviations** — output a frequency-ranked list (e.g. top-50 name tokens, top-30 address tokens) to drive the normalization rules in Phase 2 rather than relying on a hardcoded list

Exit: a short EDA summary written to `experiments/eda_notes.md` (numbers + 5-10 annotated true-positive pairs + the mined suffix/abbreviation frequency tables) that the blocking and feature design in Phases 2-3 are based on, not assumptions.

Phase 2 — Normalization pipeline

- **Keep both raw and normalized fields throughout the pipeline.** Raw strings feed edit-distance metrics (Levenshtein, Jaro-Winkler) where character fidelity matters; normalized strings feed token/TF-IDF metrics. Never discard the original.
- Name: lowercase → unicode transliteration (unidecode) → expand legal suffixes using the corpus-mined list from Phase 1 (e.g. Corp → Corporation, Pvt → Private, Ltd → Limited, & → and) into a `name_expanded` field, then strip those suffixes to produce a `name_core` field. **Expand first, then strip — stripping without expansion loses information needed for cross-source comparison.**
- Address: expand abbreviations using the corpus-mined list (Rd → Road, St → Street, Ave → Avenue, Nr → Near, etc.), extract house/street number, extract postal code prefix (first 3-5 digits), isolate landmark phrases (tokens after "Near", "Opp.", "Behind", etc.) into a `landmark` field
- Country: normalized lowercase string only — no branching logic keyed to specific country values anywhere in the codebase
- Unit-test normalization on the hand-picked Phase 1 examples before moving on

Exit: normalized fields visibly collapse the Phase 1 noise examples (e.g. "Pvt Ltd" vs "Private Limited" → same `name_core`) without merging the deliberate look-alike negatives.

Phase 3 — Blocking / candidate generation

Implement each strategy independently, measure its recall contribution, then take the union:

1. **Name-token key blocking** — sorted significant name tokens (drop stopwords + legal suffixes) + coarse address signal (city or postal prefix). Exact key match → candidate.
2. **Sorted Neighborhood Method (SNM)** — sort all records by normalized name, slide a fixed window (tune window size), pair everything within the window. Catches near-misses that exact-key blocking misses due to single-token differences.
3. **Character n-gram TF-IDF + approximate nearest neighbors (kNN)** — vectorize `name_core` + address as character n-grams (e.g. 2-4 grams), cosine top-k per S1 entity via `sklearn.neighbors.NearestNeighbors` or an ANN library. Best defense against typos and transliteration.
4. **Word n-gram TF-IDF kNN** — same as above but word-level tokens; captures transpositions and abbreviation differences that character n-grams handle less cleanly.
5. **Address-token blocking** — separate key on postal code prefix + street number; catches entities with very different names but identical addresses.
6. **MinHash / LSH fallback** — if dataset size makes TF-IDF kNN too slow (>500k pairs per strategy), replace kNN with MinHash signatures on word shingles for scalable approximate matching. Implement as a drop-in alternative to step 3/4, gated by a size check in `config.py`.

- Measure candidate recall *per strategy*, then for the union — know which strategy is pulling its weight
- Tune top-k, window size, and token-length thresholds against the recall/candidate-set-size tradeoff
- Target ≥97% candidate recall on the training set before proceeding

Exit: documented candidate recall ≥97% (or a deliberate, justified lower number) with per-strategy recall breakdown logged to `experiments/blocking_eval.md`, and candidate set size small enough for fast feature computation.

Phase 4 — Feature engineering

- **Name features** (computed on both raw and `name_core`/`name_expanded` fields separately — keep as distinct columns):
  - Levenshtein edit distance (raw)
  - Jaro-Winkler (raw — good for short strings and prefix differences)
  - Token sort ratio + token set ratio (normalized — handles word-order transpositions like "Acme Robotics" vs "Robotics Acme")
  - TF-IDF cosine — **character n-gram version** (catches typos/transliterations) and **word n-gram version** (catches abbreviations) as two separate features
  - Jaccard on token sets (full name + `name_core`)
  - First-token exact match, length difference ratio
- **Address features:** same Levenshtein/Jaro-Winkler/token-set/Jaccard family applied to the full normalized address, plus:
  - House/street number exact match flag
  - Postal code prefix match flag
  - City token overlap
  - **Landmark flag:** binary — is one (or both) address(es) landmark-based (non-numeric, contains landmark-phrase tokens)? When `True`, name-similarity features become the primary signal and address similarity is down-weighted — let the model learn this, but make the flag explicit so it can.
- **Structural / meta features:**
  - Same-country flag (let the model learn its weight — do not hard-filter on it)
  - **Candidate score gap:** difference between the blocking score (TF-IDF cosine or kNN distance) of the top candidate and the second-best candidate for the same S1 entity. A large gap signals a clear winner; a small gap signals ambiguity — include as a feature so the model can be more conservative in ambiguous cases.
  - Blocking strategy membership flags (which strategies nominated this pair)
- Sanity-check feature distributions separately for true matches vs. look-alike negatives

Exit: feature table built for the full candidate set with no nulls/NaNs, and visibly separates known positives from known hard negatives on at least 2-3 features.

Phase 5 — Model training & threshold tuning

- Split by S1 entity (never by pair) into train/val
- Train LightGBM on labeled pairs (positives from ground truth, negatives from remaining candidates)
- Sweep threshold directly against macro F0.5 on the validation split — not a fixed 0.5 cutoff
- Run the cross-country generalization check (train-US → val-India, reverse) to catch France-fragile features

Exit: a saved model + tuned threshold, with val macro F0.5, precision, recall, and cross-country F0.5 all logged.

Phase 6 — Entity-level decision logic

- Apply the F0.5-tuned threshold from Phase 5 to scored candidates — keep all candidates above the bar (do **not** force top-1; ground truth allows zero, one, or many matches per S1 entity)
- **Relative margin check (conservative ambiguity handling):** for each S1 entity, if the highest-scoring candidate is within a small margin (tune this margin on val) of the second-best, treat the top candidate as borderline and apply a stricter sub-threshold before accepting it. Protects precision when the model is uncertain.
- Enforce (if Phase 1 confirmed it) one-S1-per-record assignment: each S2/S3 id kept only for its highest-scoring S1
- **Singleton handling:** explicitly track S1 entities that clear no threshold — these predict an empty list, which scores 1.0 if they are true singletons. Do not force a match when nothing is confident enough.
- Confirm every S1 test entity gets exactly one output row, empty list where nothing clears the bar

Exit: `matching_results.tsv` and `candidate_pairs.tsv` generated on the validation split and pass a local reimplementation of `validate_submission.py`'s rules.

Phase 7 — Error analysis & iteration highest ROI phase

- Inspect false positives: what made the model think two different businesses matched?
- Inspect false negatives: typo severity, transliteration, DBA names, landmark-only addresses, blocking misses vs. model misses
- For each recurring failure pattern, decide: new blocking strategy, new feature, or accept as a known limitation
- Re-run Phases 3-6 with fixes; track F0.5 delta per change in `experiments/`

Exit: at least one documented iteration cycle with a measured F0.5 improvement and a written reason for it.

Phase 8 — Test inference & submission package

- Run the full pipeline on the real test set (France included)
- Run `utils/validate_submission.py` locally — fix everything before uploading
- Upload to the leaderboard (budget submissions per §6 below)
- Fill in `Documentation_template.md`: methodology, blocking strategy, model/features, results
- Assemble final zip: `output/` + `code/business_entity_resolution/` + methodology doc

Exit: a `SCORED` leaderboard status and a validated final zip ready to submit.

## 6. Submission budget plan (15 total, 5/day)

| Day | Planned use |
| --- | --- |
| 25 Sept | 1-2 submissions: confirm pipeline + format correctness end-to-end on real data (Phases 0-6 baseline) |
| 26 Sept | 2-3 submissions: after each meaningful Phase 7 iteration, not after every small tweak |
| 27 Sept | Reserve 1-2 for the final, best-validated version; submit early enough to catch upload issues before the deadline |

Log every submission in `experiments/` (config used, local val F0.5, leaderboard score once known) — this is also the source material for the methodology doc.

## 7. Risks

- Candidate recall silently caps the score — Phase 3 exit criteria exists specifically to catch this early.
- France generalization is untestable directly (no training examples) — the cross-country US/India check in Phase 5 is the closest proxy available.
- Submission scarcity means a bug caught only on the leaderboard is expensive — local `validate_submission.py` replication in Phase 6 must be trusted.
- Large local dataset — keep all file I/O path-configurable in `config.py` so nothing breaks when moved between machines.