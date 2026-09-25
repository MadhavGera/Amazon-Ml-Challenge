"""
data_io.py — Data loading utilities for the business entity resolution pipeline.

Phase 0: Loader functions only.
Normalization, feature engineering, and model logic are implemented in later phases.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Ensure src/ is on the path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    GROUND_TRUTH_COLUMNS,
    GT_COL_MATCHED_ENTITY_IDS,
    GT_COL_SOURCE1_ENTITY_ID,
    RAW_COL_BUSINESS_ADDRESS,
    RAW_COL_BUSINESS_NAME,
    RAW_COL_COUNTRY,
    RAW_COL_ENTITY_ID,
    RAW_SOURCE_COLUMNS,
    TEST_SOURCE1_PATH,
    TEST_SOURCE2_PATH,
    TEST_SOURCE3_PATH,
    TRAIN_GROUND_TRUTH_PATH,
    TRAIN_SOURCE1_PATH,
    TRAIN_SOURCE2_PATH,
    TRAIN_SOURCE3_PATH,
)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_source(path: str | Path) -> pd.DataFrame:
    """
    Load a source TSV file (train or test) and return it as a DataFrame.

    Parameters
    ----------
    path : str | Path
        Absolute or relative path to the TSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: entity_id, business_name, business_address, country.
        All columns are read as strings (dtype=str) to preserve leading zeros and
        avoid silent type coercion.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at *path*.
    ValueError
        If the file is missing one or more of the expected columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Source file not found: {path}\n"
            f"  → Place the raw TSV there and re-run."
        )

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,   # treat empty cells as "" not NaN
    )

    _assert_columns(df, expected=RAW_SOURCE_COLUMNS, filepath=path)
    return df


def load_ground_truth(path: str | Path) -> pd.DataFrame:
    """
    Load the training ground-truth TSV file.

    Parameters
    ----------
    path : str | Path
        Path to ``train_ground_truth.tsv``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: source1_entity_id, matched_entity_ids.
        Both columns are read as strings.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If expected columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Ground-truth file not found: {path}\n"
            f"  → Place the raw TSV there and re-run."
        )

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    _assert_columns(df, expected=GROUND_TRUTH_COLUMNS, filepath=path)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _assert_columns(
    df: pd.DataFrame,
    expected: tuple[str, ...],
    filepath: Path,
) -> None:
    """Raise a descriptive ValueError if any expected column is absent."""
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(
            f"File {filepath} is missing expected columns: {missing}\n"
            f"  Found columns: {list(df.columns)}"
        )


def _report(label: str, df: pd.DataFrame) -> None:
    """Print a compact summary of a loaded DataFrame."""
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    print(f"  Shape   : {df.shape}")
    print(f"  Columns : {list(df.columns)}")
    print(f"  Dtypes  :")
    for col, dtype in df.dtypes.items():
        print(f"    {col:<30} {dtype}")


# ─────────────────────────────────────────────────────────────────────────────
# SANITY-CHECK MAIN BLOCK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    FILES: dict[str, tuple] = {
        "train_source1": (TRAIN_SOURCE1_PATH, load_source),
        "train_source2": (TRAIN_SOURCE2_PATH, load_source),
        "train_source3": (TRAIN_SOURCE3_PATH, load_source),
        "train_ground_truth": (TRAIN_GROUND_TRUTH_PATH, load_ground_truth),
        "test_source1": (TEST_SOURCE1_PATH, load_source),
        "test_source2": (TEST_SOURCE2_PATH, load_source),
        "test_source3": (TEST_SOURCE3_PATH, load_source),
    }

    missing: list[str] = []
    errors: list[str] = []

    for label, (path, loader) in FILES.items():
        try:
            df = loader(path)
            _report(label, df)
        except FileNotFoundError as exc:
            missing.append(str(path))
            print(f"\n[MISSING] {label}: {path}")
        except Exception as exc:
            errors.append(f"{label}: {exc}")
            print(f"\n[ERROR]   {label}: {exc}")

    print(f"\n{'=' * 60}")
    if not missing and not errors:
        print("  [OK] All 7 files loaded successfully.")
    else:
        if missing:
            print(f"  [FAIL] Missing files ({len(missing)}):")
            for p in missing:
                print(f"      {p}")
        if errors:
            print(f"  [FAIL] Load errors ({len(errors)}):")
            for e in errors:
                print(f"      {e}")
    print(f"{'=' * 60}\n")
