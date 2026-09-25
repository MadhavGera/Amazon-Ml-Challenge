# Blocking Strategy Evaluation Log

Updated each time a strategy is added or tuned in Phase 3.

| Run | Strategy Config | Recall | Reduction Ratio | Pairs Generated | Notes |
|-----|----------------|--------|-----------------|-----------------|-------|
| — | Baseline (no blocking) | 1.000 | 0.000 | n² | Reference |

---

## Strategy Notes

### name_token
-

### snm (window = ?)
-

### char_tfidf (ngram = ?)
-

### word_tfidf (ngram = ?)
-

### addr_token
-

### minhash (threshold = ?)
-

---

## Target
- Blocking recall ≥ **0.95** (i.e. ≤ 5% true pairs missed)
- Reduction ratio ≥ **0.99** (i.e. discard ≥ 99% of the n² universe)
