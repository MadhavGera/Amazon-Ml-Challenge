PRD + Phase-Wise Roadmap

# Business Entity Resolution — Amazon ML Challenge 2026

Owner: Madhav · Build target: Antigravity (agentic IDE, local dataset) · Challenge window: 25–27 Sept 2026 IST

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

Exit: a short EDA summary (numbers + 5-10 example pairs) that the blocking and feature design in Phases 2-3 are based on, not assumptions.

Phase 2 — Normalization pipeline

- Name: lowercase, unicode/transliteration normalize, legal-suffix stripping into a "core name" field, keep original
- Address: abbreviation expansion, postal-code + house-number extraction, landmark-phrase isolation
- Country: normalized string only — no branching logic keyed to specific country values anywhere in the codebase
- Unit-test normalization on the hand-picked Phase 1 examples before moving on

Exit: normalized fields visibly collapse the Phase 1 noise examples (e.g. "Pvt Ltd" vs "Private Limited" → same core name) without merging the deliberate look-alike negatives.

Phase 3 — Blocking / candidate generation

- Implement each strategy independently first (name-token, address-token, postal, name TF-IDF kNN, address TF-IDF kNN)
- Measure candidate recall *per strategy*, then for the union — know which strategy is pulling its weight
- Tune top-k and token-length thresholds against the recall/candidate-set-size tradeoff
- Target ≥97% candidate recall on the training set before proceeding

Exit: documented candidate recall ≥97% (or a deliberate, justified lower number) with candidate set size small enough for fast feature computation.

Phase 4 — Feature engineering

- Name features: Levenshtein, Jaro-Winkler, token-sort/set ratio, Jaccard (full + core), length diff, first-token match
- Address features: same similarity family + postal-code match, house-number overlap, token overlap ratio
- Structural: same-country, plus any Phase-1-discovered signal (e.g. candidate rank/score gap from blocking)
- Sanity-check feature distributions separately for true matches vs. look-alike negatives

Exit: feature table built for the full candidate set with no nulls/NaNs, and visibly separates known positives from known hard negatives on at least 2-3 features.

Phase 5 — Model training & threshold tuning

- Split by S1 entity (never by pair) into train/val
- Train LightGBM on labeled pairs (positives from ground truth, negatives from remaining candidates)
- Sweep threshold directly against macro F0.5 on the validation split — not a fixed 0.5 cutoff
- Run the cross-country generalization check (train-US → val-India, reverse) to catch France-fragile features

Exit: a saved model + tuned threshold, with val macro F0.5, precision, recall, and cross-country F0.5 all logged.

Phase 6 — Entity-level decision logic

- Apply threshold to scored candidates
- Enforce (if Phase 1 confirmed it) one-S1-per-record assignment: each S2/S3 id kept only for its highest-scoring S1
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