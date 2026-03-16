#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Table Sampling, Column Profiling, and Description Generation

Implements CONTENT_VIEW_EXTENSION.md sections 2.5 (main table selection),
3.1 (low-cost table sampling), 3.2 (column type identification), and
3.3 (column description text generation).
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..constants import TAU_NUMERIC, TAU_DATETIME, TEXT_AVG_CHAR_LEN, ID_UNIQUE_RATIO

# ---------- Constants ----------

TABULAR_EXTS = {".csv", ".tsv", ".parquet", ".xlsx", ".xls"}

BLACKLIST_PATTERNS = [
    re.compile(r"^sample_submission", re.IGNORECASE),
    re.compile(r"_submission", re.IGNORECASE),
    re.compile(r"_dictionary", re.IGNORECASE),
    re.compile(r"_codebook", re.IGNORECASE),
    re.compile(r"^README", re.IGNORECASE),
    re.compile(r"^LICENSE", re.IGNORECASE),
]

# ID column pattern
_ID_PATTERN = re.compile(r"(?i)^(unnamed|index|id)$|_id$|_idx$")


# ---------- Data classes ----------

@dataclass
class ColStats:
    """Column-level profiling statistics."""
    name: str
    dtype: str           # "numeric" | "datetime" | "categorical" | "text"
    missing_pct: float
    n_unique: int
    unique_pct: float
    stats: Dict[str, Any] = field(default_factory=dict)


# ---------- File I/O ----------

def read_by_ext(path, nrows=None):
    """Read a tabular file by dispatching on its extension.

    Supports csv, tsv, parquet, xlsx, and xls formats with automatic
    encoding fallback for csv/tsv.

    Args:
        path: Path to the tabular file.
        nrows: Maximum number of rows to read.

    Returns:
        DataFrame or None on failure.
    """
    path = Path(path)
    ext = path.suffix.lower()

    try:
        if ext == ".csv":
            return pd.read_csv(path, nrows=nrows, encoding="utf-8",
                               on_bad_lines="skip", engine="python")
        elif ext == ".tsv":
            return pd.read_csv(path, sep="\t", nrows=nrows, encoding="utf-8",
                               on_bad_lines="skip", engine="python")
        elif ext == ".parquet":
            df = pd.read_parquet(path, engine="fastparquet")
            if nrows is not None:
                df = df.head(nrows)
            return df
        elif ext in (".xlsx", ".xls"):
            return pd.read_excel(path, nrows=nrows)
        else:
            return None
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError, OSError) as first_err:
        if ext in (".csv", ".tsv"):
            sep = "\t" if ext == ".tsv" else ","
            # Try latin-1 with C engine (handles NUL bytes better)
            try:
                return pd.read_csv(path, sep=sep, nrows=nrows, encoding="latin-1",
                                   on_bad_lines="skip", engine="c")
            except (UnicodeDecodeError, pd.errors.ParserError, ValueError, OSError):
                pass
        return None


def select_main_table(dataset_dir) -> Tuple[Optional[str], int, Optional[str]]:
    """Select the largest tabular file from a dataset directory.

    Scans for tabular files, filters against blacklist patterns, and
    returns the largest file by size.

    Args:
        dataset_dir: Path to the dataset directory.

    Returns:
        (path, file_size, extension) or (None, 0, None) if no table found.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        return None, 0, None

    tabular_files = []
    for f in dataset_dir.rglob("*"):
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        if ext not in TABULAR_EXTS:
            continue
        if any(pat.search(f.stem) for pat in BLACKLIST_PATTERNS):
            continue
        tabular_files.append((f, f.stat().st_size, ext))

    if not tabular_files:
        return None, 0, None

    tabular_files.sort(key=lambda x: x[1], reverse=True)
    best = tabular_files[0]
    return str(best[0]), best[1], best[2]


# ---------- Table sampling ----------

def sample_table(path, max_rows=1024, max_cols=60):
    """Read and sample a table: drop useless columns, keep top informative ones.

    Implements the low-cost table sampling from CONTENT_VIEW_EXTENSION.md section 3.1:
    1. Read up to ``max_rows`` rows.
    2. Drop all-null, constant, and high-uniqueness ID columns.
    3. Rank remaining columns by info_score = unique_ratio * notna_ratio,
       keep top ``max_cols``.

    Args:
        path: Path to the tabular file.
        max_rows: Maximum rows to read.
        max_cols: Maximum columns to keep.

    Returns:
        Sampled DataFrame or None on failure.
    """
    df = read_by_ext(path, nrows=max_rows)
    if df is None or df.empty or len(df.columns) == 0:
        return None

    # Drop all-null columns
    df = df.dropna(axis=1, how="all")

    # Drop columns with unhashable types (e.g. lists, dicts)
    hashable_cols = []
    for c in df.columns:
        try:
            df[c].nunique()
            hashable_cols.append(c)
        except TypeError:
            pass
    df = df[hashable_cols]

    if df.empty or len(df.columns) == 0:
        return None

    # Drop constant columns (nunique <= 1)
    nuniques = df.nunique()
    df = df.loc[:, nuniques > 1]

    # Drop ID-like columns: name matches pattern and unique ratio > 95%
    cols_to_drop = []
    for c in df.columns:
        if _ID_PATTERN.search(str(c)):
            unique_ratio = df[c].nunique() / max(len(df), 1)
            if unique_ratio > ID_UNIQUE_RATIO:
                cols_to_drop.append(c)
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    if df.empty or len(df.columns) == 0:
        return None

    # Column budget: rank by info_score = unique_ratio * notna_ratio
    if len(df.columns) > max_cols:
        n = max(len(df), 1)
        scores = {}
        for c in df.columns:
            nunique_ratio = df[c].nunique() / n
            notna_ratio = df[c].notna().mean()
            scores[c] = nunique_ratio * notna_ratio
        top_cols = sorted(scores, key=scores.get, reverse=True)[:max_cols]
        df = df[top_cols]

    return df


# ---------- Column profiling ----------

def profile_column(series, col_name):
    """Profile a single column and determine its type.

    Priority: numeric > datetime > text > categorical.

    Args:
        series: pandas Series (single column data).
        col_name: Column name string.

    Returns:
        ColStats instance.
    """
    missing_pct = float(series.isna().mean() * 100)
    non_null = series.dropna()
    n = len(non_null)

    if n == 0:
        return ColStats(name=col_name, dtype="categorical",
                        missing_pct=missing_pct, n_unique=0, unique_pct=0.0)

    n_unique = int(non_null.nunique())
    unique_pct = float(n_unique / max(n, 1) * 100)

    # 1. Try numeric
    numeric_converted = pd.to_numeric(non_null, errors="coerce")
    numeric_ratio = numeric_converted.notna().mean()
    if numeric_ratio >= TAU_NUMERIC:
        vals = numeric_converted.dropna()
        stats = {
            "min": float(vals.min()),
            "max": float(vals.max()),
            "median": float(vals.median()),
            "mean": float(vals.mean()),
            "std": float(vals.std()) if len(vals) > 1 else 0.0,
        }
        return ColStats(name=col_name, dtype="numeric", missing_pct=missing_pct,
                        n_unique=n_unique, unique_pct=unique_pct, stats=stats)

    # 2. Try datetime
    str_vals = non_null.astype(str)
    try:
        dt_converted = pd.to_datetime(str_vals, errors="coerce")
        dt_ratio = dt_converted.notna().mean()
    except (ValueError, TypeError, OverflowError):
        dt_ratio = 0.0

    if dt_ratio >= TAU_DATETIME:
        dt_valid = dt_converted.dropna()
        try:
            span = (dt_valid.max() - dt_valid.min()).days if len(dt_valid) > 1 else 0
        except (OverflowError, pd.errors.OutOfBoundsDatetime, Exception):
            span = 0
        try:
            earliest_str = str(dt_valid.min().date())
            latest_str = str(dt_valid.max().date())
        except (OverflowError, Exception):
            earliest_str = "unknown"
            latest_str = "unknown"
        stats = {
            "earliest": earliest_str,
            "latest": latest_str,
            "span_days": int(span),
        }
        return ColStats(name=col_name, dtype="datetime", missing_pct=missing_pct,
                        n_unique=n_unique, unique_pct=unique_pct, stats=stats)

    # 3. Distinguish text vs categorical by avg string length
    str_lens = non_null.astype(str).str.len()
    avg_len = float(str_lens.mean())

    if avg_len >= TEXT_AVG_CHAR_LEN:
        sample_text = str(non_null.iloc[0])[:200] if len(non_null) > 0 else ""
        stats = {
            "avg_len": avg_len,
            "max_len": float(str_lens.max()),
            "sample_text": sample_text,
        }
        return ColStats(name=col_name, dtype="text", missing_pct=missing_pct,
                        n_unique=n_unique, unique_pct=unique_pct, stats=stats)

    # 4. Categorical
    vc = non_null.value_counts().head(5)
    top_values = [(str(k), int(v)) for k, v in vc.items()]
    stats = {"top_values": top_values}
    return ColStats(name=col_name, dtype="categorical", missing_pct=missing_pct,
                    n_unique=n_unique, unique_pct=unique_pct, stats=stats)


def profile_table(df):
    """Profile all columns in a DataFrame.

    Args:
        df: DataFrame to profile.

    Returns:
        List of ColStats, one per column.
    """
    return [profile_column(df[col], str(col)) for col in df.columns]


# ---------- Description generation ----------

def col_to_description(cs):
    """Generate an English text description from ColStats using type-specific templates.

    Templates follow CONTENT_VIEW_EXTENSION.md section 3.3.

    Args:
        cs: ColStats instance.

    Returns:
        English description string.
    """
    name = cs.name
    s = cs.stats

    if cs.dtype == "numeric":
        return (
            f'Column "{name}": numeric, range [{s["min"]}, {s["max"]}], '
            f'median={s["median"]}, mean={s["mean"]:.2f}, std={s["std"]:.2f}, '
            f'{cs.missing_pct:.1f}% missing.'
        )
    elif cs.dtype == "categorical":
        top_vals = s.get("top_values", [])
        top_str = ", ".join(f"{v} ({c})" for v, c in top_vals[:3])
        return (
            f'Column "{name}": categorical with {cs.n_unique} unique values '
            f'({cs.unique_pct:.1f}% unique). Top values: {top_str}. '
            f'{cs.missing_pct:.1f}% missing.'
        )
    elif cs.dtype == "datetime":
        return (
            f'Column "{name}": datetime from {s["earliest"]} to {s["latest"]}, '
            f'spanning {s["span_days"]} days. {cs.missing_pct:.1f}% missing.'
        )
    elif cs.dtype == "text":
        sample = s.get("sample_text", "")[:100]
        return (
            f'Column "{name}": free text, avg length {s["avg_len"]:.0f} chars. '
            f'Sample: "{sample}". {cs.missing_pct:.1f}% missing.'
        )
    return f'Column "{name}": unknown type. {cs.missing_pct:.1f}% missing.'
