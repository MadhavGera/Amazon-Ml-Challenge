"""
normalize.py — Text normalisation and field extraction.

Phase 1 implementation target.

This module consumes raw source DataFrames (Schema 1) and produces the
Normalized record schema (Schema 2).  No model logic or feature computation
lives here.

Planned steps
-------------
1. Unicode normalisation (NFC → NFKC, unidecode for transliteration).
2. Lowercase and whitespace collapse.
3. Legal-suffix stripping (LLC, Ltd, Inc, Pvt, GmbH, …) → name_core.
4. Abbreviation expansion table (St → Street, Bros → Brothers, …) → name_expanded, address_expanded.
5. House-number extraction (leading numeric/alphanumeric token) → house_number.
6. Postal-code extraction (regex per pattern, first N chars) → postal_prefix.
7. Landmark detection (keyword lookup in address tokens) → landmark.
8. Country normalisation via ISO-3166 lookup table (no hardcoded country logic) → country_norm.

All country-specific logic MUST use a lookup table / mapping — never if/elif
chains keyed on country names.
"""

from __future__ import annotations

import pandas as pd


def normalize_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise a raw source DataFrame into the Normalized record schema.

    Parameters
    ----------
    df : pd.DataFrame
        Raw source DataFrame with columns defined in config.RAW_SOURCE_COLUMNS.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns defined in config.NORMALIZED_COLUMNS, in order.

    Notes
    -----
    Implemented in Phase 1.
    """
    raise NotImplementedError("normalize_records: implement in Phase 1")


def expand_abbreviations(text: str, lookup: dict[str, str]) -> str:
    """
    Replace abbreviations in *text* using *lookup* (token-level substitution).

    Parameters
    ----------
    text : str
        Input string (name or address).
    lookup : dict[str, str]
        Mapping from abbreviation → expanded form.

    Returns
    -------
    str
        Text with abbreviations replaced.

    Notes
    -----
    Implemented in Phase 1.
    """
    raise NotImplementedError("expand_abbreviations: implement in Phase 1")


def strip_legal_suffixes(name: str) -> str:
    """
    Remove common legal entity suffixes from *name* to produce name_core.

    Parameters
    ----------
    name : str
        Lowercase, whitespace-collapsed business name.

    Returns
    -------
    str
        Name with legal suffixes stripped.

    Notes
    -----
    Implemented in Phase 1.
    """
    raise NotImplementedError("strip_legal_suffixes: implement in Phase 1")


def extract_house_number(address: str) -> str:
    """
    Extract the leading house/plot number token from *address*.

    Returns empty string if no numeric token is found.

    Notes
    -----
    Implemented in Phase 1.
    """
    raise NotImplementedError("extract_house_number: implement in Phase 1")


def extract_postal_prefix(address: str, prefix_len: int = 3) -> str:
    """
    Extract and return the first *prefix_len* characters of the postal/ZIP code
    found in *address*.  Returns empty string if no postal code is detected.

    Notes
    -----
    Implemented in Phase 1.
    """
    raise NotImplementedError("extract_postal_prefix: implement in Phase 1")


def extract_landmark(address: str, landmark_keywords: set[str]) -> str:
    """
    Return the first landmark token found in *address* that appears in
    *landmark_keywords*, or empty string if none.

    Notes
    -----
    Implemented in Phase 1.
    """
    raise NotImplementedError("extract_landmark: implement in Phase 1")


def normalize_country(raw_country: str, iso_lookup: dict[str, str]) -> str:
    """
    Map *raw_country* to an ISO 3166-1 alpha-2 code via *iso_lookup*.
    Returns empty string if unresolvable.

    Parameters
    ----------
    raw_country : str
        The `country` field value from a raw source record.
    iso_lookup : dict[str, str]
        Mapping from any known country representation → ISO alpha-2 code.
        No hardcoded logic — the lookup table is the only country logic here.

    Notes
    -----
    Implemented in Phase 1.
    """
    raise NotImplementedError("normalize_country: implement in Phase 1")
