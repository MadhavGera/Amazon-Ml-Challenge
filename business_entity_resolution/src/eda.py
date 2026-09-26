"""
eda.py - Ground-truth-focused EDA for the business entity resolution pipeline.
Person B track (Phase B1).

This script computes and prints:
  1. Match-count distribution per S1 entity (full histogram)
  2. Overall singleton rate
  3. Average and median matches for non-singleton S1 entities
  4. Match breakdown by source prefix (S2- vs S3-)
  5. Multi-match check: does any S2/S3 id appear under more than one S1 entity?
     (Critical for Phase B4 one-S1-per-record dedup rule)
  6. Data-integrity check: orphaned S1 ids and unresolvable matched ids

Results are also written to experiments/eda_notes_B.md.

Run from the project root:
    python src/eda.py

Original scaffold functions (field_completeness, name_length_distribution, etc.)
are kept as NotImplementedError stubs for Person A / Phase 2 EDA.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

# Ensure src/ is on the path regardless of invocation directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    EXPERIMENTS_DIR,
    GT_COL_MATCHED_ENTITY_IDS,
    GT_COL_SOURCE1_ENTITY_ID,
    RAW_COL_ENTITY_ID,
    TRAIN_GROUND_TRUTH_PATH,
    TRAIN_SOURCE1_PATH,
    TRAIN_SOURCE2_PATH,
    TRAIN_SOURCE3_PATH,
)
from data_io import load_ground_truth, load_source


# ---------------------------------------------------------------------------
# LEGACY SCAFFOLD STUBS (Person A / Phase 2 - do not remove)
# ---------------------------------------------------------------------------

def field_completeness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame reporting, per column: total rows, non-empty count,
    empty count, and % complete.

    Notes
    -----
    Implemented in Phase 1.
    """
    raise NotImplementedError("field_completeness: implement in Phase 1")


def name_length_distribution(df: pd.DataFrame) -> pd.Series:
    """
    Return a Series of value counts of token count per record in raw_name.

    Notes
    -----
    Implemented in Phase 1.
    """
    raise NotImplementedError("name_length_distribution: implement in Phase 1")


def top_name_tokens(df: pd.DataFrame, top_n: int = 50) -> pd.Series:
    """
    Return the top_n most frequent tokens across all raw_name values.

    Notes
    -----
    Implemented in Phase 1.
    """
    raise NotImplementedError("top_name_tokens: implement in Phase 1")


def country_distribution(df: pd.DataFrame) -> pd.Series:
    """
    Return value counts of the raw country field.

    Notes
    -----
    Implemented in Phase 1.
    """
    raise NotImplementedError("country_distribution: implement in Phase 1")


def cross_source_token_overlap(
    source1_norm: pd.DataFrame,
    candidates_norm: pd.DataFrame,
) -> float:
    """
    Estimate the fraction of source1 entities that share at least one
    significant name token with at least one candidate entity.

    Notes
    -----
    Implemented in Phase 2.
    """
    raise NotImplementedError("cross_source_token_overlap: implement in Phase 2")


def ground_truth_match_ratio(ground_truth: pd.DataFrame) -> pd.Series:
    """
    Return a Series of value counts of the number of matched entities per
    source1_entity_id in the ground-truth file.

    Notes
    -----
    Implemented in Phase 2.
    """
    raise NotImplementedError("ground_truth_match_ratio: implement in Phase 2")


# ---------------------------------------------------------------------------
# GROUND-TRUTH EDA (Person B - Phase B1)
# ---------------------------------------------------------------------------

def _parse_matched_ids(cell: str) -> list:
    """
    Parse a comma- or pipe-separated matched_entity_ids cell into a list.
    Returns empty list for empty/whitespace cells.
    """
    cell = cell.strip()
    if not cell:
        return []
    delimiter = "," if "," in cell else "|"
    return [tok.strip() for tok in cell.split(delimiter) if tok.strip()]


def run_ground_truth_eda(
    gt_df:  pd.DataFrame,
    s1_df:  pd.DataFrame,
    s2_df:  pd.DataFrame,
    s3_df:  pd.DataFrame,
) -> dict:
    """
    Run all ground-truth-focused EDA and return a results dict.

    Parameters
    ----------
    gt_df : ground-truth DataFrame (load_ground_truth)
    s1_df : source1 DataFrame    (load_source)
    s2_df : source2 DataFrame    (load_source)
    s3_df : source3 DataFrame    (load_source)

    Returns
    -------
    dict with all computed statistics (used by the reporter and note-writer).
    """
    # Build id sets for cross-file integrity checks
    s1_ids = set(s1_df[RAW_COL_ENTITY_ID].str.strip())
    s2_ids = set(s2_df[RAW_COL_ENTITY_ID].str.strip())
    s3_ids = set(s3_df[RAW_COL_ENTITY_ID].str.strip())

    # Parse matched IDs into lists
    gt_df = gt_df.copy()
    gt_df["_matched_list"] = gt_df[GT_COL_MATCHED_ENTITY_IDS].apply(_parse_matched_ids)
    gt_df["_match_count"]  = gt_df["_matched_list"].apply(len)

    total_s1 = len(gt_df)

    # ── 1. Match-count histogram ──────────────────────────────────────────
    count_col = gt_df["_match_count"]
    histogram = {}
    histogram[0] = int((count_col == 0).sum())
    histogram[1] = int((count_col == 1).sum())
    histogram[2] = int((count_col == 2).sum())
    histogram["3+"] = int((count_col >= 3).sum())
    # Full raw histogram (exact counts)
    full_hist = dict(Counter(count_col.tolist()))
    full_hist = dict(sorted(full_hist.items()))

    # ── 2. Singleton rate ─────────────────────────────────────────────────
    singleton_count = histogram[0]
    singleton_rate  = singleton_count / total_s1 if total_s1 > 0 else 0.0

    # ── 3. Average and median for non-singletons ──────────────────────────
    non_singleton_counts = count_col[count_col > 0]
    if len(non_singleton_counts) > 0:
        avg_matches    = float(non_singleton_counts.mean())
        median_matches = float(non_singleton_counts.median())
    else:
        avg_matches    = 0.0
        median_matches = 0.0

    # ── 4. Source breakdown ───────────────────────────────────────────────
    all_matched_ids = [
        mid
        for row in gt_df["_matched_list"]
        for mid in row
    ]
    total_matched = len(all_matched_ids)
    s2_count = sum(1 for mid in all_matched_ids if mid.startswith("S2-"))
    s3_count = sum(1 for mid in all_matched_ids if mid.startswith("S3-"))
    other_count = total_matched - s2_count - s3_count
    s2_frac = s2_count / total_matched if total_matched > 0 else 0.0
    s3_frac = s3_count / total_matched if total_matched > 0 else 0.0

    # ── 5. Multi-match check ──────────────────────────────────────────────
    # Does any S2/S3 id appear under MORE THAN ONE S1 entity?
    matched_id_to_s1s = defaultdict(list)
    for _, row in gt_df.iterrows():
        s1_id = row[GT_COL_SOURCE1_ENTITY_ID]
        for mid in row["_matched_list"]:
            matched_id_to_s1s[mid].append(s1_id)

    multi_matched = {
        mid: s1_list
        for mid, s1_list in matched_id_to_s1s.items()
        if len(s1_list) > 1
    }
    multi_match_count = len(multi_matched)
    multi_match_examples = list(multi_matched.items())[:5]  # up to 5 examples

    # Verdict: is one-S2/S3-per-S1 dedup rule safe?
    dedup_safe = multi_match_count == 0

    # ── 6. Data integrity ─────────────────────────────────────────────────
    gt_s1_ids = set(gt_df[GT_COL_SOURCE1_ENTITY_ID].str.strip())

    # S1 ids in ground truth but not in source1
    orphan_gt_s1_ids = gt_s1_ids - s1_ids
    # S1 ids in source1 but not in ground truth
    missing_gt_s1_ids = s1_ids - gt_s1_ids

    # Matched ids that don't appear in source2 or source3
    unresolvable_matched = set()
    for mid in set(all_matched_ids):
        if not (mid in s2_ids or mid in s3_ids):
            unresolvable_matched.add(mid)

    return {
        "total_s1":           total_s1,
        "histogram":          histogram,
        "full_hist":          full_hist,
        "singleton_count":    singleton_count,
        "singleton_rate":     singleton_rate,
        "avg_matches":        avg_matches,
        "median_matches":     median_matches,
        "total_matched_ids":  total_matched,
        "s2_count":           s2_count,
        "s3_count":           s3_count,
        "other_count":        other_count,
        "s2_frac":            s2_frac,
        "s3_frac":            s3_frac,
        "multi_match_count":  multi_match_count,
        "multi_matched":      multi_matched,
        "multi_match_examples": multi_match_examples,
        "dedup_safe":         dedup_safe,
        "orphan_gt_s1_ids":   orphan_gt_s1_ids,
        "missing_gt_s1_ids":  missing_gt_s1_ids,
        "unresolvable_matched": unresolvable_matched,
        "s1_row_count":       len(s1_df),
        "s2_row_count":       len(s2_df),
        "s3_row_count":       len(s3_df),
        "gt_row_count":       len(gt_df),
    }


def print_eda_results(r: dict) -> None:
    """Print all EDA results to stdout in a readable format."""
    SEP = "=" * 65
    sep = "-" * 65

    print(f"\n{SEP}")
    print("  Ground-Truth EDA  (Person B - Phase B1)")
    print(SEP)

    print(f"\n  Dataset sizes:")
    print(f"    Source 1 records : {r['s1_row_count']:,}")
    print(f"    Source 2 records : {r['s2_row_count']:,}")
    print(f"    Source 3 records : {r['s3_row_count']:,}")
    print(f"    Ground-truth rows: {r['gt_row_count']:,}")

    # 1. Match-count histogram
    print(f"\n{sep}")
    print("  1. Match-count distribution per S1 entity")
    print(sep)
    total = r["total_s1"]
    print(f"  Total S1 entities in ground truth : {total:,}")
    print()
    h = r["histogram"]
    print(f"  {'Matches':<10} {'Count':>8}  {'%':>7}")
    print(f"  {'-'*30}")
    for key in [0, 1, 2, "3+"]:
        cnt = h[key]
        pct = cnt / total * 100 if total > 0 else 0.0
        print(f"  {str(key):<10} {cnt:>8,}  {pct:>6.2f}%")

    print(f"\n  Full histogram (exact counts):")
    print(f"  {'#Matches':<10} {'#S1 entities':>14}")
    print(f"  {'-'*28}")
    for k, v in r["full_hist"].items():
        print(f"  {k:<10} {v:>14,}")

    # 2. Singleton rate
    print(f"\n{sep}")
    print("  2. Singleton rate")
    print(sep)
    print(f"  True singletons (0 matches) : {r['singleton_count']:,}")
    print(f"  Singleton rate              : {r['singleton_rate']*100:.2f}%")

    # 3. Average / median for non-singletons
    print(f"\n{sep}")
    print("  3. Matches per non-singleton S1 entity")
    print(sep)
    non_single = total - r["singleton_count"]
    print(f"  Non-singleton S1 entities : {non_single:,}")
    print(f"  Average matches           : {r['avg_matches']:.4f}")
    print(f"  Median  matches           : {r['median_matches']:.1f}")

    # 4. Source breakdown
    print(f"\n{sep}")
    print("  4. Match breakdown by source prefix")
    print(sep)
    print(f"  Total matched entity ids : {r['total_matched_ids']:,}")
    print(f"  S2- prefixed             : {r['s2_count']:,}  ({r['s2_frac']*100:.2f}%)")
    print(f"  S3- prefixed             : {r['s3_count']:,}  ({r['s3_frac']*100:.2f}%)")
    if r["other_count"] > 0:
        print(f"  Other prefix (anomaly)   : {r['other_count']:,}")

    # 5. Multi-match check
    print(f"\n{sep}")
    print("  5. CRITICAL CHECK — Multi-match (one-S1-per-record dedup rule)")
    print(sep)
    print(f"  S2/S3 ids appearing under 2+ S1 entities: {r['multi_match_count']:,}")
    if r["multi_match_examples"]:
        print(f"\n  Examples (up to 5):")
        for mid, s1_list in r["multi_match_examples"]:
            print(f"    {mid}  ->  S1 entities: {s1_list}")
    else:
        print("  (none found)")

    verdict = (
        "SAFE — no S2/S3 id appears under more than one S1 entity. "
        "One-S1-per-record dedup rule is valid."
        if r["dedup_safe"]
        else f"NOT SAFE — {r['multi_match_count']:,} S2/S3 id(s) are matched to "
             "more than one S1 entity. The dedup rule would silently drop valid matches; "
             "review Phase B4 design."
    )
    print(f"\n  VERDICT: {verdict}")

    # 6. Data integrity
    print(f"\n{sep}")
    print("  6. Data integrity checks")
    print(sep)
    orph_count = len(r["orphan_gt_s1_ids"])
    miss_count = len(r["missing_gt_s1_ids"])
    unres_count = len(r["unresolvable_matched"])
    print(f"  S1 ids in ground truth NOT in source1 : {orph_count:,}", end="")
    if orph_count > 0:
        sample = list(r["orphan_gt_s1_ids"])[:3]
        print(f"  (sample: {sample})")
    else:
        print()
    print(f"  S1 ids in source1 NOT in ground truth : {miss_count:,}", end="")
    if miss_count > 0 and miss_count <= 5:
        print(f"  ({list(r['missing_gt_s1_ids'])})")
    elif miss_count > 5:
        print(f"  (sample: {list(r['missing_gt_s1_ids'])[:5]})")
    else:
        print()
    print(f"  Matched ids not in source2 or source3 : {unres_count:,}", end="")
    if unres_count > 0:
        sample = list(r["unresolvable_matched"])[:3]
        print(f"  (sample: {sample})")
    else:
        print()

    print(f"\n{SEP}\n")


def write_eda_notes(r: dict, output_path: Path) -> None:
    """
    Write EDA findings to experiments/eda_notes_B.md.
    Overwrites existing placeholder content.
    """
    from datetime import date

    dedup_verdict_detail = (
        f"**SAFE.** No S2 or S3 entity appears under more than one S1 entity "
        f"across the entire ground-truth file. The one-S1-per-record dedup rule "
        f"(proposed for Phase B4) is valid and will not silently discard any true match."
        if r["dedup_safe"]
        else f"**NOT SAFE.** {r['multi_match_count']} S2/S3 entities appear under "
             f"more than one S1 entity. Enforcing the one-S1-per-record rule in Phase B4 "
             f"would silently drop valid matches. The assignment strategy must account for "
             f"shared S2/S3 records or accept the loss."
    )

    examples_md = ""
    if r["multi_match_examples"]:
        lines = ["| Shared S2/S3 ID | S1 Entities |", "|---|---|"]
        for mid, s1_list in r["multi_match_examples"]:
            lines.append(f"| `{mid}` | {', '.join(s1_list)} |")
        examples_md = "\n".join(lines)
    else:
        examples_md = "_None found._"

    integrity_lines = []
    if len(r["orphan_gt_s1_ids"]) == 0:
        integrity_lines.append("- All S1 ids in ground truth appear in source1. **OK.**")
    else:
        sample = list(r["orphan_gt_s1_ids"])[:5]
        integrity_lines.append(
            f"- **{len(r['orphan_gt_s1_ids'])} S1 ids** in ground truth are absent from "
            f"source1. Sample: {sample}"
        )
    if len(r["missing_gt_s1_ids"]) == 0:
        integrity_lines.append("- All source1 S1 ids appear in ground truth. **OK.**")
    else:
        integrity_lines.append(
            f"- **{len(r['missing_gt_s1_ids'])} source1 ids** have no ground-truth row "
            f"(no coverage entry)."
        )
    if len(r["unresolvable_matched"]) == 0:
        integrity_lines.append("- All matched entity ids resolve to source2 or source3. **OK.**")
    else:
        sample = list(r["unresolvable_matched"])[:5]
        integrity_lines.append(
            f"- **{len(r['unresolvable_matched'])} matched ids** cannot be found in source2 "
            f"or source3. Sample: {sample}"
        )

    full_hist_rows = "\n".join(
        f"| {k} | {v:,} | {v/r['total_s1']*100:.2f}% |"
        for k, v in r["full_hist"].items()
    )

    content = f"""# EDA Notes — Person B (Phase B1)

**Date:** {date.today().isoformat()}
**Sources analysed:** train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv

---

## 0. Dataset Sizes

| File | Row count |
|---|---|
| train_source1.tsv | {r['s1_row_count']:,} |
| train_source2.tsv | {r['s2_row_count']:,} |
| train_source3.tsv | {r['s3_row_count']:,} |
| train_ground_truth.tsv | {r['gt_row_count']:,} |

---

## 1. Match-Count Distribution per S1 Entity

| # Matches | # S1 Entities | % of total |
|---|---|---|
| 0 (singletons) | {r['histogram'][0]:,} | {r['histogram'][0]/r['total_s1']*100:.2f}% |
| 1 | {r['histogram'][1]:,} | {r['histogram'][1]/r['total_s1']*100:.2f}% |
| 2 | {r['histogram'][2]:,} | {r['histogram'][2]/r['total_s1']*100:.2f}% |
| 3+ | {r['histogram']['3+']:,} | {r['histogram']['3+']/r['total_s1']*100:.2f}% |

**Full histogram (exact counts):**

| # Matches | # S1 Entities | % |
|---|---|---|
{full_hist_rows}

Total S1 entities scored: **{r['total_s1']:,}**

---

## 2. Singleton Rate

- **True singletons (0 ground-truth matches):** {r['singleton_count']:,}
- **Singleton rate:** {r['singleton_rate']*100:.2f}% of all S1 entities

Implication: predicting "empty" for every S1 entity would yield a macro F0.5 of
{r['singleton_rate']:.4f} (the singleton baseline). Any model must beat this floor.

---

## 3. Matches per Non-Singleton S1 Entity

- Non-singleton S1 entities: {r['total_s1'] - r['singleton_count']:,}
- **Average matches:** {r['avg_matches']:.4f}
- **Median matches:** {r['median_matches']:.1f}

---

## 4. Match Breakdown by Source

Total matched entity ids across all ground-truth rows: **{r['total_matched_ids']:,}**

| Source prefix | Count | Fraction |
|---|---|---|
| S2- (source 2) | {r['s2_count']:,} | {r['s2_frac']*100:.2f}% |
| S3- (source 3) | {r['s3_count']:,} | {r['s3_frac']*100:.2f}% |
{"| Other (anomaly) | " + str(r['other_count']) + " | " + f"{r['other_count']/r['total_matched_ids']*100:.2f}%" + " |" if r['other_count'] > 0 else ""}

---

## 5. Critical Check — Multi-Match (gates Phase B4 dedup rule)

**Question:** Does any single S2 or S3 entity_id appear in the matched_entity_ids
of MORE THAN ONE S1 entity?

- S2/S3 ids appearing under 2+ S1 entities: **{r['multi_match_count']:,}**

{examples_md}

**Verdict:** {dedup_verdict_detail}

---

## 6. Data Integrity

{chr(10).join(integrity_lines)}

---

## Key Takeaways for Later Phases

1. **Singleton baseline:** {r['singleton_rate']*100:.2f}% of S1 entities are true singletons.
   Predicting empty for all gives F0.5 = {r['singleton_rate']:.4f}; any model must exceed this.
2. **Non-singleton matches:** median = {r['median_matches']:.0f} match(es) per non-singleton.
   Ground truth is mostly low-cardinality (1–2 matches), so precision is critical (F0.5 weights it 2x).
3. **Source mix:** {r['s2_frac']*100:.1f}% S2, {r['s3_frac']*100:.1f}% S3.
   Both sources contribute; blocking must cover both.
4. **Dedup rule safety:** {"CONFIRMED SAFE" if r['dedup_safe'] else "NOT SAFE — revisit Phase B4"} —
   see Section 5 verdict above.
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    print(f"  EDA notes written to: {output_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading data files...")
    missing = []
    for label, path in [
        ("train_source1",       TRAIN_SOURCE1_PATH),
        ("train_source2",       TRAIN_SOURCE2_PATH),
        ("train_source3",       TRAIN_SOURCE3_PATH),
        ("train_ground_truth",  TRAIN_GROUND_TRUTH_PATH),
    ]:
        if not path.exists():
            missing.append((label, path))

    if missing:
        print("\n[ERROR] The following dataset files are missing:")
        for label, path in missing:
            print(f"  {label}: {path}")
        print(
            "\nPlace the training TSV files in the dataset/train/ directory "
            "and re-run this script.\n"
        )
        sys.exit(1)

    s1_df  = load_source(TRAIN_SOURCE1_PATH)
    s2_df  = load_source(TRAIN_SOURCE2_PATH)
    s3_df  = load_source(TRAIN_SOURCE3_PATH)
    gt_df  = load_ground_truth(TRAIN_GROUND_TRUTH_PATH)

    print("Computing EDA statistics...")
    results = run_ground_truth_eda(gt_df, s1_df, s2_df, s3_df)

    print_eda_results(results)

    notes_path = EXPERIMENTS_DIR / "eda_notes_B.md"
    write_eda_notes(results, notes_path)
