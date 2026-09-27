"""
normalize.py — Text normalisation and field extraction.

Phase A2 implementation (Person A Track).

This module consumes raw source DataFrames (Schema 1) and produces the
Normalized record schema (Schema 2). No model logic or feature computation
lives here.

Normalization Pipeline Overview
-------------------------------
1. Unicode & Script Normalisation:
   - unidecode for phonetic transliteration of accented and non-Latin scripts (Devanagari, Gujarati).
   - Case-insensitive literal "null", "none", "n/a" handling as missing-value sentinels.
2. Noise & Prefix Stripping:
   - Strips leading noise punctuation (">>", "--", "...", "@", "#").
   - Strips web domain suffixes (".com", "www.").
   - Normalizes leet/OCR digit substitutions in words ("5olutions" -> "solutions", "f0undry" -> "foundry").
3. Alias Injection Handling:
   - Detects "a/k/a", "aka", "d/b/a", "dba" and isolates the canonical business name.
4. Name Normalisation:
   - Abbreviation expansion using corpus-mined top legal/corporate suffixes -> name_expanded.
   - Suffix and high-frequency domain stopword stripping -> name_core.
5. Address Normalisation:
   - Address abbreviation expansion (Rd -> Road, St -> Street, Ste -> Suite) -> address_expanded.
   - House / plot number extraction (handles US fractions/ranges and Indian "H.no", "Shop No", "Ag-", "Q-") -> house_number.
   - Postal prefix extraction (first 3-5 digits of ZIP/PIN if present) -> postal_prefix.
   - Landmark detection (phrases after Near, Opp, C/o, or matching commercial plaza/center keywords) -> landmark.
6. Country Normalisation:
   - Lookup-table-based mapping to ISO 3166-1 alpha-2 codes without any hardcoded branching logic.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd
from unidecode import unidecode

# Ensure src/ is on the path regardless of invocation directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    NORM_COL_ADDRESS_EXPANDED,
    NORM_COL_COUNTRY_NORM,
    NORM_COL_ENTITY_ID,
    NORM_COL_HOUSE_NUMBER,
    NORM_COL_LANDMARK,
    NORM_COL_NAME_CORE,
    NORM_COL_NAME_EXPANDED,
    NORM_COL_POSTAL_PREFIX,
    NORM_COL_RAW_ADDRESS,
    NORM_COL_RAW_NAME,
    NORMALIZED_COLUMNS,
    RAW_COL_BUSINESS_ADDRESS,
    RAW_COL_BUSINESS_NAME,
    RAW_COL_COUNTRY,
    RAW_COL_ENTITY_ID,
)


# ---------------------------------------------------------------------------
# LOOKUP TABLES & CONSTANTS (Mined from Phase A1 12.5M-Record Corpus)
# ---------------------------------------------------------------------------

LEGAL_SUFFIX_EXPANSIONS: dict[str, str] = {
    "ltd": "limited",
    "pvt": "private",
    "inc": "incorporated",
    "corp": "corporation",
    "co": "company",
    "llc": "llc",
    "llp": "llp",
    "pllc": "pllc",
    "lp": "lp",
    "gmbh": "gmbh",
    "sa": "sa",
    "bv": "bv",
    "plc": "plc",
    "bros": "brothers",
    "assoc": "associates",
    "intl": "international",
    "tech": "technologies",
    "mfg": "manufacturing",
    "mgmt": "management",
    "dept": "department",
    "univ": "university",
    "med": "medicine",
    "hosp": "hospital",
    "&": "and",
}

LEGAL_SUFFIXES_TO_STRIP: set[str] = {
    "llc", "l.l.c.", "l.l.c", "inc", "incorporated", "ltd", "limited",
    "pvt", "private", "corp", "corporation", "co", "company", "llp",
    "pllc", "lp", "gmbh", "sa", "bv", "plc", "partners", "services",
    "group", "holdings", "associates", "solutions", "enterprises",
    "industries", "ventures", "center", "care"
}

ADDRESS_ABBREVIATIONS: dict[str, str] = {
    "st": "street",
    "str": "street",
    "rd": "road",
    "dr": "drive",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "ln": "lane",
    "ct": "court",
    "cir": "circle",
    "pkwy": "parkway",
    "hwy": "highway",
    "ste": "suite",
    "apt": "apartment",
    "fl": "floor",
    "flr": "floor",
    "bldg": "building",
    "blk": "block",
    "pl": "plaza",
    "plz": "plaza",
    "ctr": "center",
    "cntr": "center",
    "sec": "sector",
    "sq": "square",
    "ter": "terrace",
    "tpke": "turnpike",
    "opp": "opposite",
    "nr": "near",
    "no": "number",
    "twp": "township",
    "fwy": "freeway",
    "rt": "route",
    "rte": "route",
}

ISO_COUNTRY_LOOKUP: dict[str, str] = {
    "us": "us",
    "usa": "us",
    "united states": "us",
    "united states of america": "us",
    "u.s.": "us",
    "u.s.a.": "us",
    "in": "in",
    "ind": "in",
    "india": "in",
    "fr": "fr",
    "fra": "fr",
    "france": "fr",
    "de": "de",
    "deu": "de",
    "germany": "de",
    "uk": "gb",
    "gb": "gb",
    "gbr": "gb",
    "united kingdom": "gb",
    "great britain": "gb",
    "ca": "ca",
    "can": "ca",
    "canada": "ca",
    "au": "au",
    "aus": "au",
    "australia": "au",
}

DEFAULT_LANDMARK_KEYWORDS: set[str] = {
    "plaza", "centre", "center", "mall", "tower", "complex", "bourse",
    "market", "mkt", "cinema", "hospital", "campus", "park", "colony", "enclave"
}


# ---------------------------------------------------------------------------
# TEXT NORMALISATION HELPERS
# ---------------------------------------------------------------------------

def _clean_text_basic(text: str) -> str:
    """
    Apply basic sanitisation: unidecode transliteration, noise prefix stripping,
    web domain removal, and missing-value sentinel ('null', 'none', 'n/a') handling.
    """
    if not text or not isinstance(text, str):
        return ""
    t = unidecode(text).strip()
    if not t or t.lower() in {"null", "none", "n/a", "nan"}:
        return ""
    # Strip leading noise punctuation (>>, --, ..., @, #, etc.)
    t = re.sub(r'^[\s\>\-\.\@\#\:\~\*]+', '', t)
    # Strip .com and www. prefixes/suffixes
    t = re.sub(r'\.com\b', '', t, flags=re.IGNORECASE)
    t = re.sub(r'\bwww\.', '', t, flags=re.IGNORECASE)
    # Remove literal null/none/n/a tokens
    t = re.sub(r'\b(?:null|none|n/a)\b', ' ', t, flags=re.IGNORECASE)
    # Lowercase and collapse whitespace
    t = t.lower()
    t = re.sub(r'[\s_]+', ' ', t).strip()
    return t


def _handle_alias_injection(name: str) -> str:
    """
    Detect and isolate canonical name from synthetic alias injection patterns.
    
    Rationale & Strategy:
    Real records frequently contain patterns such as 'Lyrahalo a/k/a Hack Beacon Royalties Inc'
    or 'Nylaquoquo aka Tirupati Marketing Limited', where a synthetic decoy prefix is joined
    to the real business name via 'a/k/a' or 'aka'. Retaining the prefix corrupts the first
    token (destroying first-token blocking) and penalizes string similarity metrics.
    When an alias marker ('a/k/a', 'aka', 'd/b/a', 'dba') is detected, we extract the right-hand
    (canonical) segment for downstream expansion and core name extraction.
    """
    pattern = r'\b(?:a/?k/?a|aka|d/?b/?a|dba|f/?k/?a)\b'
    parts = re.split(pattern, name, flags=re.IGNORECASE)
    if len(parts) > 1:
        right = parts[1].strip()
        left = parts[0].strip()
        if right:
            return right
        elif left:
            return left
    return name


def _normalize_leet_word(word: str) -> str:
    """
    Normalize character/digit OCR substitutions inside mixed alphanumeric tokens.
    E.g., '5olutions' -> 'solutions', 'f0undry' -> 'foundry', 'aur0ra' -> 'aurora', '6lobe' -> 'globe'.
    Purely numeric tokens (e.g. house numbers '1824') are left untouched.
    """
    has_letters = any(c.isalpha() for c in word)
    has_digits = any(c.isdigit() for c in word)
    if not (has_letters and has_digits):
        return word

    w = word
    w = re.sub(r'([a-zA-Z])0', r'\g<1>o', w)
    w = re.sub(r'0([a-zA-Z])', r'o\g<1>', w)
    w = re.sub(r'([a-zA-Z])5', r'\g<1>s', w)
    w = re.sub(r'5([a-zA-Z])', r's\g<1>', w)
    w = re.sub(r'([a-zA-Z])6', r'\g<1>g', w)
    w = re.sub(r'6([a-zA-Z])', r'g\g<1>', w)
    w = re.sub(r'([a-zA-Z])1([a-zA-Z])', r'\g<1>l\g<2>', w)
    return w


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
    """
    if not text:
        return ""
    words = re.findall(r'[a-zA-Z0-9\&\/\.\-]+|[^\w\s]', text)
    res = []
    for w in words:
        clean_w = w.strip('.').lower()
        if clean_w in lookup:
            res.append(lookup[clean_w])
        elif w == "&" and "&" in lookup:
            res.append(lookup["&"])
        else:
            res.append(w)
    out = " ".join(res)
    out = re.sub(r'\s+([,\.\/\-])', r'\1', out)
    out = re.sub(r'([,\.\/\-])\s+', r'\1 ', out)
    out = re.sub(r'[\s]+', ' ', out).strip()
    return out


def strip_legal_suffixes(name: str) -> str:
    """
    Remove common legal entity suffixes and high-frequency non-discriminating
    stopwords from *name* to produce name_core.

    Parameters
    ----------
    name : str
        Lowercase, whitespace-collapsed business name.

    Returns
    -------
    str
        Name with legal suffixes and salutations stripped.
    """
    if not name:
        return ""
    # Strip leading salutations / prefixes (M/s, Mr, The)
    n = re.sub(r'^(?:m/s|mr|mrs|ms|the)\b\.?\s*', '', name, flags=re.IGNORECASE)

    # Tokenize and strip legal suffixes from the trailing position
    tokens = re.findall(r'\b[a-zA-Z0-9]+\b', n.lower())
    while tokens and tokens[-1] in LEGAL_SUFFIXES_TO_STRIP:
        tokens.pop()
    while tokens and tokens[0] in {"the", "and", "or"}:
        tokens.pop(0)

    # Filter any remaining isolated legal designator tokens
    filtered = [t for t in tokens if t not in {"llc", "ltd", "inc", "pvt", "corp"}]
    if filtered:
        return " ".join(filtered)
    return " ".join(tokens)


def normalize_name(raw_name: str) -> tuple[str, str]:
    """
    Normalize raw business name into (name_expanded, name_core).

    Parameters
    ----------
    raw_name : str
        Raw business name from source record.

    Returns
    -------
    tuple[str, str]
        (name_expanded, name_core)
    """
    if not raw_name or not isinstance(raw_name, str):
        return "", ""

    # 1. Alias handling (isolate canonical name if alias injected)
    name_clean = _handle_alias_injection(raw_name)

    # 2. Basic sanitization, unidecode, lowercase
    name_clean = _clean_text_basic(name_clean)
    if not name_clean:
        return "", ""

    # 3. Leet OCR token normalization
    words = [_normalize_leet_word(w) for w in name_clean.split()]
    name_clean = " ".join(words)

    # 4. Expand legal abbreviations
    name_expanded = expand_abbreviations(name_clean, LEGAL_SUFFIX_EXPANSIONS)
    name_expanded = re.sub(r'[^a-zA-Z0-9\s]', ' ', name_expanded)
    name_expanded = re.sub(r'\s+', ' ', name_expanded).strip()

    # 5. Extract core name
    name_core = strip_legal_suffixes(name_expanded)
    return name_expanded, name_core


def extract_house_number(address: str) -> str:
    """
    Extract the leading house/plot/shop/door number token from *address*.
    Handles both US street numbers and Indian plot/shop/house identifiers
    (e.g., 'H.no 104/A', 'Door No 2/528 B', 'Shop No. M-2693/94', 'Ag-572', 'Q-903').

    Returns empty string if no numeric token is found.
    """
    if not address or not isinstance(address, str):
        return ""

    addr = unidecode(address).strip().lower()
    if not addr or addr in {"null", "none", "n/a"}:
        return ""
    addr = re.sub(r'^[\s\>\-\.\@\#\:\~\*]+', '', addr)

    # 1. Explicit Indian / unit prefix markers: H.no, House no, Plot no, Shop no, Door no, Flat no
    m_pref = re.search(
        r'\b(?:h\.?\s*no|house\s*no|plot\s*no|shop\s*no|door\s*no|d\.?\s*no|flat\s*no|unit\s*no|no)\b\.?\s*([0-9a-zA-Z\/\-\#]+(?:\s*[0-9]\/[0-9])?(?:\s+[a-zA-Z])?)\b',
        addr,
        flags=re.IGNORECASE
    )
    if m_pref:
        val = m_pref.group(1).strip().lstrip('#').strip()
        if val.isdigit():
            val = str(int(val))
        return val

    # 2. Specialized alpha-numeric identifiers at start of address (e.g. Q-903, Ag-572, Gw 7012, 100-A)
    m_alpha = re.match(
        r'^\s*([a-zA-Z]{1,3}(?:-[0-9]+|\s+[0-9]+)(?:-[0-9]+|\/[0-9a-zA-Z]+)?)\b',
        addr
    )
    if m_alpha:
        return m_alpha.group(1).strip()

    # 3. Standard leading number / fraction / range: e.g. "212", "00212", "1007 1/2", "212-214", "104/a", "#1824", "##25", "100-a"
    m_lead = re.match(
        r'^\s*(?:#|##)?\s*([0-9]+(?:-[a-zA-Z]+|\/[0-9a-zA-Z]+|\s+[0-9]\/[0-9])?(?:-[0-9]+(?:\s+[0-9]\/[0-9])?)?(?:[a-zA-Z])?)\b',
        addr
    )
    if m_lead:
        val = m_lead.group(1).strip()
        if re.match(r'^0+[0-9]+$', val):
            val = str(int(val))
        return val

    return ""


def extract_postal_prefix(address: str, prefix_len: int = 3) -> str:
    """
    Extract and return the first *prefix_len* characters of the postal/ZIP code
    found in *address*. Returns empty string if no postal code is detected.
    """
    if not address or not isinstance(address, str):
        return ""
    addr = unidecode(address).strip().lower()
    if not addr or addr in {"null", "none", "n/a"}:
        return ""

    # Strip leading street numbers to prevent false-positive 5-digit house number matching
    search_text = re.sub(
        r'^\s*(?:#|##)?[0-9a-zA-Z\/\-\s]+?\b(?:street|st|road|rd|avenue|ave|drive|dr|court|ct|lane|ln|blvd|way|pkwy|hwy|highway|colony|nagar|bagh|extn)\b',
        '',
        addr
    )
    m = re.findall(r'\b([0-9]{5,6})(?:-[0-9]{4})?\b', search_text)
    if m:
        code = m[-1]
        return code[:prefix_len]
    return ""


def extract_landmark(address: str, landmark_keywords: set[str] | None = None) -> str:
    """
    Return the first landmark token or relative landmark phrase found in *address*,
    or empty string if none.
    """
    if not address or not isinstance(address, str):
        return ""
    addr = unidecode(address).strip().lower()
    if not addr or addr in {"null", "none", "n/a"}:
        return ""

    # 1. Relative landmark markers: Near, Opp, Opposite, Behind, Adjacent, C/o, Next to
    m_rel = re.search(
        r'\b(?:near|opp\.?|opposite|behind|adjacent|c\/o|care\s+of|next\s+to|beside)\s+([a-zA-Z0-9\s\-]+?)(?:,|$|\.|\b(?:road|rd|st|street|ave|avenue|lane|fl|floor|dr|drive|highway|hwy|dist|district)\b)',
        addr,
        flags=re.IGNORECASE
    )
    if m_rel:
        return m_rel.group(1).strip()

    # 2. Known commercial landmark / campus keywords
    keywords = landmark_keywords or DEFAULT_LANDMARK_KEYWORDS
    for part in addr.split(','):
        part_clean = part.strip()
        for kw in keywords:
            if re.search(rf'\b{kw}\b', part_clean):
                return part_clean
    return ""


def normalize_address(raw_address: str) -> tuple[str, str, str, str]:
    """
    Normalize raw address into (address_expanded, house_number, postal_prefix, landmark).

    Parameters
    ----------
    raw_address : str
        Raw business address from source record.

    Returns
    -------
    tuple[str, str, str, str]
        (address_expanded, house_number, postal_prefix, landmark)
    """
    if not raw_address or not isinstance(raw_address, str):
        return "", "", "", ""
    addr_clean = _clean_text_basic(raw_address)
    if not addr_clean:
        return "", "", "", ""

    house_num = extract_house_number(raw_address)
    postal_pref = extract_postal_prefix(raw_address)
    landmark = extract_landmark(raw_address)

    addr_exp = expand_abbreviations(addr_clean, ADDRESS_ABBREVIATIONS)
    addr_exp = re.sub(r'[^a-zA-Z0-9\s]', ' ', addr_exp)
    addr_exp = re.sub(r'\s+', ' ', addr_exp).strip()

    return addr_exp, house_num, postal_pref, landmark


def normalize_country(raw_country: str, iso_lookup: dict[str, str] | None = None) -> str:
    """
    Map *raw_country* to an ISO 3166-1 alpha-2 code via *iso_lookup*.
    Returns empty string if unresolvable.

    Parameters
    ----------
    raw_country : str
        The `country` field value from a raw source record.
    iso_lookup : dict[str, str], optional
        Mapping from country representation → ISO alpha-2 code.
        No hardcoded logic — the lookup table is the only country logic here.

    Returns
    -------
    str
        Normalized 2-letter ISO country code or cleaned string.
    """
    if not raw_country or not isinstance(raw_country, str):
        return ""
    lookup = iso_lookup if iso_lookup is not None else ISO_COUNTRY_LOOKUP
    clean = unidecode(raw_country).strip().lower()
    if not clean or clean in {"null", "none", "n/a"}:
        return ""
    return lookup.get(clean, clean)


# ---------------------------------------------------------------------------
# MAIN NORMALISATION PIPELINE
# ---------------------------------------------------------------------------

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
        DataFrame with columns defined in config.NORMALIZED_COLUMNS, in order:
        [entity_id, raw_name, name_expanded, name_core, raw_address,
         address_expanded, house_number, postal_prefix, landmark, country_norm]
    """
    id_col = RAW_COL_ENTITY_ID if RAW_COL_ENTITY_ID in df.columns else df.columns[0]
    name_col = RAW_COL_BUSINESS_NAME if RAW_COL_BUSINESS_NAME in df.columns else df.columns[1]
    addr_col = RAW_COL_BUSINESS_ADDRESS if RAW_COL_BUSINESS_ADDRESS in df.columns else df.columns[2]
    country_col = RAW_COL_COUNTRY if RAW_COL_COUNTRY in df.columns else df.columns[3]

    entity_ids = df[id_col].astype(str).tolist()
    raw_names = df[name_col].fillna("").astype(str).tolist()
    raw_addresses = df[addr_col].fillna("").astype(str).tolist()
    raw_countries = df[country_col].fillna("").astype(str).tolist()

    norm_names = [normalize_name(n) for n in raw_names]
    name_expanded = [n[0] for n in norm_names]
    name_core = [n[1] for n in norm_names]

    norm_addrs = [normalize_address(a) for a in raw_addresses]
    address_expanded = [a[0] for a in norm_addrs]
    house_numbers = [a[1] for a in norm_addrs]
    postal_prefixes = [a[2] for a in norm_addrs]
    landmarks = [a[3] for a in norm_addrs]

    country_norm = [normalize_country(c) for c in raw_countries]

    res = pd.DataFrame({
        NORM_COL_ENTITY_ID: entity_ids,
        NORM_COL_RAW_NAME: raw_names,
        NORM_COL_NAME_EXPANDED: name_expanded,
        NORM_COL_NAME_CORE: name_core,
        NORM_COL_RAW_ADDRESS: raw_addresses,
        NORM_COL_ADDRESS_EXPANDED: address_expanded,
        NORM_COL_HOUSE_NUMBER: house_numbers,
        NORM_COL_POSTAL_PREFIX: postal_prefixes,
        NORM_COL_LANDMARK: landmarks,
        NORM_COL_COUNTRY_NORM: country_norm,
    })

    return res[list(NORMALIZED_COLUMNS)]
