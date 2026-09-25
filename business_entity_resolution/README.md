# Business Entity Resolution — ML Challenge

## Overview
A two-person team pipeline for resolving business entities across multiple
noisy data sources (source1/source2/source3).  The end product is a
submission TSV mapping every source1 entity to its matching entities in the
other sources.

## 8-Phase Roadmap

| Phase | Owner | Goal |
|---|---|---|
| **0 — Scaffolding** | Both | Repo structure, frozen schemas, config |
| **1 — Normalisation** | A | `normalize.py` — cleaning, field extraction |
| **2 — EDA** | B | `eda.py` — data profiling, inform blocking design |
| **3 — Blocking** | A | `blocking.py` — candidate generation, recall ≥ 95% |
| **4 — Features** | B | `features.py` — name + address + meta features |
| **5 — Modelling** | Both | `train.py`, `predict.py`, `metrics.py` |
| **6 — Threshold tuning** | B | Margin gate, MATCH_THRESHOLD search |
| **7 — Error analysis** | Both | `error_analysis.py`, hard-negative mining |
| **8 — Submission** | Both | Final run, validate with `utils/validate_submission.py` |

## Project Structure
```
business_entity_resolution/
├── dataset/
│   ├── train/     ← place train_source{1,2,3}.tsv + train_ground_truth.tsv
│   └── test/      ← place test_source{1,2,3}.tsv
├── src/           ← pipeline source code
├── notebooks/     ← exploratory notebooks
├── output/        ← generated TSVs and model.txt
├── experiments/   ← notes, blocking eval results
├── docs/          ← schemas.md and other design docs
├── utils/         ← validate_submission.py (provided separately)
└── ...
```

## Quickstart
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place raw TSV files in dataset/train/ and dataset/test/

# 3. Sanity-check data loading
python src/data_io.py
```

## Key Design Decisions
- **Single source of truth**: all paths, column names, and tunables live in
  `src/config.py`.  Change `PROJECT_ROOT` to move the project between machines.
- **Schema-first**: all column names are frozen in both `docs/schemas.md` and
  `src/config.py` so both teammates code against the same contracts independently.
- **No country hardcoding**: all country logic uses lookup tables, never
  `if country == "India":` style branches.
- **Blocking gate**: `CANDIDATE_SIZE_THRESHOLD` gates between TF-IDF+kNN and
  MinHash/LSH — defined in config, logic implemented in Phase 3.
