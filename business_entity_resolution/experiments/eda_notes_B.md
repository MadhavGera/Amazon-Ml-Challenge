# EDA Notes — Person B (Phase B1)

**Date:** 2026-09-26
**Sources analysed:** train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv

---

## 0. Dataset Sizes

| File | Row count |
|---|---|
| train_source1.tsv | 2,206,821 |
| train_source2.tsv | 5,034,616 |
| train_source3.tsv | 5,285,603 |
| train_ground_truth.tsv | 2,206,821 |

---

## 1. Match-Count Distribution per S1 Entity

| # Matches | # S1 Entities | % of total |
|---|---|---|
| 0 (singletons) | 123,247 | 5.58% |
| 1 | 119,157 | 5.40% |
| 2 | 375,212 | 17.00% |
| 3+ | 1,589,205 | 72.01% |

**Full histogram (exact counts):**

| # Matches | # S1 Entities | % |
|---|---|---|
| 0 | 123,247 | 5.58% |
| 1 | 119,157 | 5.40% |
| 2 | 375,212 | 17.00% |
| 3 | 530,841 | 24.05% |
| 4 | 484,115 | 21.94% |
| 5 | 321,957 | 14.59% |
| 6 | 164,868 | 7.47% |
| 7 | 63,968 | 2.90% |
| 8 | 18,680 | 0.85% |
| 9 | 4,205 | 0.19% |
| 10 | 534 | 0.02% |
| 11 | 37 | 0.00% |

Total S1 entities scored: **2,206,821**

---

## 2. Singleton Rate

- **True singletons (0 ground-truth matches):** 123,247
- **Singleton rate:** 5.58% of all S1 entities

Implication: predicting "empty" for every S1 entity would yield a macro F0.5 of
0.0558 (the singleton baseline). Any model must beat this floor.

---

## 3. Matches per Non-Singleton S1 Entity

- Non-singleton S1 entities: 2,083,574
- **Average matches:** 3.6660
- **Median matches:** 4.0

---

## 4. Match Breakdown by Source

Total matched entity ids across all ground-truth rows: **7,638,365**

| Source prefix | Count | Fraction |
|---|---|---|
| S2- (source 2) | 3,693,619 | 48.36% |
| S3- (source 3) | 3,944,746 | 51.64% |


---

## 5. Critical Check — Multi-Match (gates Phase B4 dedup rule)

**Question:** Does any single S2 or S3 entity_id appear in the matched_entity_ids
of MORE THAN ONE S1 entity?

- S2/S3 ids appearing under 2+ S1 entities: **0**

_None found._

**Verdict:** **SAFE.** No S2 or S3 entity appears under more than one S1 entity across the entire ground-truth file. The one-S1-per-record dedup rule (proposed for Phase B4) is valid and will not silently discard any true match.

---

## 6. Data Integrity

- All S1 ids in ground truth appear in source1. **OK.**
- All source1 S1 ids appear in ground truth. **OK.**
- All matched entity ids resolve to source2 or source3. **OK.**

---

## Key Takeaways for Later Phases

1. **Singleton baseline:** 5.58% of S1 entities are true singletons.
   Predicting empty for all gives F0.5 = 0.0558; any model must exceed this.
2. **Non-singleton matches:** median = 4 match(es) per non-singleton.
   Ground truth is mostly low-cardinality (1–2 matches), so precision is critical (F0.5 weights it 2x).
3. **Source mix:** 48.4% S2, 51.6% S3.
   Both sources contribute; blocking must cover both.
4. **Dedup rule safety:** CONFIRMED SAFE —
   see Section 5 verdict above.
