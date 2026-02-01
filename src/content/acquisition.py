#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Acquisition Module

Extracts and enhances the core acquisition functions from NB00
(00_tabular_data_download.ipynb) for reuse in scale expansion scripts.

Provides:
  - Filter presets for 10K / 50K / 100K targets
  - Candidate filtering
  - Kaggle API search with caching
  - Slug-to-ref matching
  - Dataset download with retries
  - Non-tabular backfill (Tier-1 + Tier-2)
"""

import csv
import json
import re
import shutil
import subprocess
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .sampling import select_main_table

# Raise CSV field size limit for DatasetVersions.csv
csv.field_size_limit(sys.maxsize)

# ---------------------------------------------------------------------------
# Tag sets
# ---------------------------------------------------------------------------

TABULAR_TAGS = {
    "tabular", "classification", "regression",
    "exploratory data analysis", "data analytics",
}

BROAD_TAGS = TABULAR_TAGS | {
    "data visualization", "feature engineering", "statistics",
    "finance", "healthcare", "economics", "survey", "census",
    "social science", "business", "education", "sports",
    "geospatial analysis", "time series analysis",
}

# ---------------------------------------------------------------------------
# Filter presets
# ---------------------------------------------------------------------------

FILTER_PRESETS: Dict[int, Dict[str, Any]] = {
    10000: {
        "tags": TABULAR_TAGS,
        "size_min": 100 * 1024,           # 100 KB
        "size_max": 100 * 1024 * 1024,    # 100 MB
        "min_views": 0,
    },
    50000: {
        "tags": BROAD_TAGS,
        "size_min": 50 * 1024,            # 50 KB
        "size_max": 500 * 1024 * 1024,    # 500 MB
        "min_views": 0,
    },
    100000: {
        "tags": None,                     # any tags
        "size_min": 10 * 1024,            # 10 KB
        "size_max": 1024 * 1024 * 1024,   # 1 GB
        "min_views": 0,
    },
}

SEARCH_MARGIN = 1.2


def get_filter_preset(target: int) -> Dict[str, Any]:
    """Return the filter preset for the given target size.

    Selects the preset whose key is the smallest value >= target.
    Falls back to the largest preset if target exceeds all keys.
    """
    sorted_keys = sorted(FILTER_PRESETS.keys())
    for k in sorted_keys:
        if target <= k:
            return FILTER_PRESETS[k]
    return FILTER_PRESETS[sorted_keys[-1]]


# ---------------------------------------------------------------------------
# Candidate filtering
# ---------------------------------------------------------------------------

def _has_tag_match(tags_str, tag_set):
    """Check whether a comma-separated Tags string contains any tag in *tag_set*."""
    if pd.isna(tags_str) or not isinstance(tags_str, str):
        return False
    tags = {t.strip().lower() for t in tags_str.split(",")}
    return bool(tags & tag_set)


def filter_candidates(
    meta: pd.DataFrame,
    tags: Optional[set] = None,
    size_min: int = 100 * 1024,
    size_max: int = 100 * 1024 * 1024,
    min_views: int = 0,
) -> pd.DataFrame:
    """Apply tag, size, and view filters to the metadata DataFrame.

    Args:
        meta: DataFrame with columns Tags, TotalCompressedBytes, TotalViews.
        tags: Set of tag strings to match (None = no tag filter).
        size_min: Minimum compressed bytes.
        size_max: Maximum compressed bytes.
        min_views: Minimum TotalViews (0 = no filter).

    Returns:
        Filtered copy of *meta*.
    """
    if tags is not None:
        mask_tag = meta["Tags"].apply(lambda t: _has_tag_match(t, tags))
    else:
        mask_tag = pd.Series(True, index=meta.index)

    mask_size = (
        meta["TotalCompressedBytes"].notna()
        & (meta["TotalCompressedBytes"] >= size_min)
        & (meta["TotalCompressedBytes"] <= size_max)
    )

    if min_views > 0:
        mask_views = meta["TotalViews"] >= min_views
    else:
        mask_views = pd.Series(True, index=meta.index)

    combined = mask_tag & mask_size & mask_views
    print(f"[acquisition] Filter: tag={combined.sum() if tags else 'any'}, "
          f"size={mask_size.sum()}, views={mask_views.sum()}, combined={combined.sum()}")
    return meta[combined].copy()


# ---------------------------------------------------------------------------
# Kaggle API helpers
# ---------------------------------------------------------------------------

def check_kaggle_api() -> bool:
    """Test whether the ``kaggle`` CLI is available and authenticated."""
    try:
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return False
    except (FileNotFoundError, Exception):
        return False

    try:
        result = subprocess.run(
            ["kaggle", "datasets", "list", "-s", "titanic", "--csv",
             "--max-size", "1048576"],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0
    except Exception:
        return False


def search_kaggle_slug(
    slug: str,
    cache_dir: Path,
    max_size_mb: int = 100,
) -> List[Dict[str, Any]]:
    """Search Kaggle API for datasets matching *slug*, with file-based caching.

    Args:
        slug: Dataset slug to search for.
        cache_dir: Directory for per-slug JSON cache files.
        max_size_mb: Maximum dataset size filter (MB).

    Returns:
        List of result dicts from the Kaggle API.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{slug}.json"

    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    try:
        max_size_bytes = max_size_mb * 1024 * 1024
        cmd = [
            "kaggle", "datasets", "list",
            "-s", slug,
            "--csv",
            "--max-size", str(max_size_bytes),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        time.sleep(0.5)  # rate limit

        if result.returncode != 0 or not result.stdout.strip():
            results: List[Dict[str, Any]] = []
        else:
            csv_text = result.stdout.strip()
            if csv_text.startswith("ref") or csv_text.startswith('"ref'):
                df = pd.read_csv(StringIO(csv_text))
                results = df.to_dict("records")
            else:
                results = []
    except Exception as e:
        print(f"  [acquisition] API error for '{slug}': {e}")
        results = []

    with open(cache_file, "w") as f:
        json.dump(results, f)

    return results


def match_slug_to_ref(
    slug: str,
    title: str,
    api_results: List[Dict[str, Any]],
) -> Tuple[Optional[str], str]:
    """Match a local slug to a Kaggle API ref.

    Priority: exact suffix match on ref, then fuzzy title match.

    Args:
        slug: Local dataset slug.
        title: Dataset title string.
        api_results: Results from :func:`search_kaggle_slug`.

    Returns:
        ``(ref, confidence)`` where confidence is ``"exact"`` | ``"fuzzy"`` | ``"none"``.
    """
    if not api_results:
        return None, "none"

    slug_lower = slug.lower().strip()

    # 1. Exact suffix match
    for item in api_results:
        ref = str(item.get("ref", "")).lower().strip()
        if ref.endswith(f"/{slug_lower}"):
            return item["ref"], "exact"

    # 2. Fuzzy title match
    if title and isinstance(title, str):
        title_lower = title.lower().strip()
        for item in api_results:
            api_title = str(item.get("title", "")).lower().strip()
            if (title_lower == api_title
                    or title_lower in api_title
                    or api_title in title_lower):
                return item["ref"], "fuzzy"

    return None, "none"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_dataset(
    ref: str,
    dataset_id: int,
    output_dir: Path,
    timeout: int = 300,
    max_retries: int = 2,
) -> bool:
    """Download a Kaggle dataset using the CLI with retry logic.

    Args:
        ref: Kaggle dataset ref string (e.g. ``"user/dataset-name"``).
        dataset_id: Dataset ID (for logging).
        output_dir: Target directory for downloaded files.
        timeout: Download timeout in seconds.
        max_retries: Number of retry attempts.

    Returns:
        True if download succeeded, False otherwise.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ref or ref == "":
        return False

    for attempt in range(max_retries + 1):
        if attempt > 0:
            wait = 2 ** attempt
            print(f"  [acquisition] Retry {attempt}/{max_retries} for {ref} after {wait}s")
            time.sleep(wait)
        try:
            cmd = [
                "kaggle", "datasets", "download",
                "-d", ref,
                "-p", str(output_dir),
                "--unzip",
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
            )
            if result.returncode == 0:
                return True
            print(f"  [acquisition] Download failed for {ref}: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            print(f"  [acquisition] Download timeout for {ref}")
        except Exception as e:
            print(f"  [acquisition] Download error for {ref}: {e}")

    return False


# ---------------------------------------------------------------------------
# Backfill non-tabular datasets
# ---------------------------------------------------------------------------

def backfill_non_tabular(
    d_content: pd.DataFrame,
    main_tables: pd.DataFrame,
    candidates: pd.DataFrame,
    slug_to_ref: pd.DataFrame,
    index_map: pd.DataFrame,
    tab_raw_dir: Path,
    cache_dir: Path,
    target: int,
    kaggle_available: bool = True,
    max_size_mb: int = 100,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Replace non-tabular datasets with backfill candidates.

    Identifies datasets where ``main_table_path == ""``, removes them, then
    fills the gap from Tier-1 (already matched but unselected) and Tier-2
    (unsearched candidates).

    Args:
        d_content: Current D_content DataFrame.
        main_tables: Current main_tables DataFrame.
        candidates: Full candidate pool.
        slug_to_ref: Slug-to-ref mapping DataFrame.
        index_map: Global doc_idx mapping.
        tab_raw_dir: Root download directory.
        cache_dir: API cache directory.
        target: Target D_content size.
        kaggle_available: Whether Kaggle API is available.
        max_size_mb: Max size for API search.

    Returns:
        Updated ``(d_content, main_tables, slug_to_ref)`` tuple.
    """
    non_tab_mask = main_tables["main_table_path"] == ""
    n_non_tabular = non_tab_mask.sum()
    if n_non_tabular == 0:
        print("[acquisition] All datasets are tabular, no backfill needed.")
        return d_content, main_tables, slug_to_ref

    print(f"[acquisition] Non-tabular datasets: {n_non_tabular}")
    non_tab_ids = set(main_tables.loc[non_tab_mask, "DatasetId"].astype(int).values)

    # Remove non-tabular entries
    d_content = d_content[~d_content["Id"].astype(int).isin(non_tab_ids)].reset_index(drop=True)
    main_tables = main_tables[~main_tables["DatasetId"].astype(int).isin(non_tab_ids)].reset_index(drop=True)
    need = target - len(d_content)
    print(f"[acquisition] After removal: {len(d_content)}, need backfill: {need}")

    if need <= 0:
        return d_content, main_tables, slug_to_ref

    dc_ids = set(d_content["Id"].astype(int).values)
    imap_lookup = dict(zip(index_map["Id"].astype(int), index_map["doc_idx"].astype(int)))

    backfill_dc = []
    backfill_mt = []

    # --- Tier-1: Already matched but not selected ---
    tier1_pool = slug_to_ref[
        (slug_to_ref["confidence"] != "none")
        & (~slug_to_ref["Id"].astype(int).isin(dc_ids))
        & (~slug_to_ref["Id"].astype(int).isin(non_tab_ids))
    ].copy()
    tier1_pool = tier1_pool.merge(
        candidates[["Id", "TotalDownloads", "TotalViews"]],
        on="Id", how="left",
    )
    tier1_pool = tier1_pool.sort_values("TotalDownloads", ascending=False).reset_index(drop=True)
    print(f"[acquisition] Tier-1 pool: {len(tier1_pool)}")

    n_t1_ok = 0
    for _, brow in tier1_pool.iterrows():
        if need <= 0:
            break
        ds_id = int(brow["Id"])
        ref = brow["ref"]
        slug = brow["Slug"]
        ds_dir = tab_raw_dir / str(ds_id)

        # Download if needed
        if not (ds_dir.exists() and any(ds_dir.iterdir())):
            if not kaggle_available:
                continue
            ok = download_dataset(ref, ds_id, ds_dir)
            if not ok:
                continue

        # Verify tabular
        path, fsize, ext = select_main_table(ds_dir)
        if path is None:
            if ds_dir.exists():
                shutil.rmtree(ds_dir)
            continue

        doc_idx = imap_lookup.get(ds_id)
        if doc_idx is None:
            continue

        backfill_dc.append({
            "doc_idx": doc_idx, "Id": ds_id, "Slug": slug,
            "ref": ref,
            "TotalDownloads": brow.get("TotalDownloads", 0),
            "TotalViews": brow.get("TotalViews", 0),
        })
        backfill_mt.append({
            "DatasetId": ds_id, "doc_idx": doc_idx,
            "main_table_path": path, "file_size": fsize, "extension": ext,
        })
        need -= 1
        n_t1_ok += 1

    print(f"[acquisition] Tier-1 backfill: {n_t1_ok}")

    # --- Tier-2: Unsearched candidates ---
    n_t2_ok = 0
    tier2_new_rows = []

    if need > 0 and kaggle_available:
        print(f"[acquisition] Tier-1 insufficient, starting Tier-2 (need {need})...")
        already_ids = dc_ids | {r["Id"] for r in backfill_dc} | non_tab_ids
        already_slugs = set(slug_to_ref["Slug"].values)
        tier2_pool = candidates[
            (~candidates["Id"].isin(already_ids))
            & (~candidates["Slug"].isin(already_slugs))
        ].nlargest(need * 5, "TotalDownloads")

        for _, crow in tier2_pool.iterrows():
            if need <= 0:
                break
            slug = crow["Slug"]
            title = crow["Title"]
            ds_id = int(crow["Id"])

            results = search_kaggle_slug(slug, cache_dir, max_size_mb)
            ref, confidence = match_slug_to_ref(slug, title, results)
            tier2_new_rows.append({
                "Id": ds_id, "Slug": slug, "Title": title,
                "ref": ref if ref else "", "confidence": confidence,
            })

            if confidence == "none" or not ref:
                continue

            ds_dir = tab_raw_dir / str(ds_id)
            if not (ds_dir.exists() and any(ds_dir.iterdir())):
                ok = download_dataset(ref, ds_id, ds_dir)
                if not ok:
                    continue

            path, fsize, ext = select_main_table(ds_dir)
            if path is None:
                if ds_dir.exists():
                    shutil.rmtree(ds_dir)
                continue

            doc_idx = imap_lookup.get(ds_id)
            if doc_idx is None:
                continue

            backfill_dc.append({
                "doc_idx": doc_idx, "Id": ds_id, "Slug": slug,
                "ref": ref,
                "TotalDownloads": crow.get("TotalDownloads", 0),
                "TotalViews": crow.get("TotalViews", 0),
            })
            backfill_mt.append({
                "DatasetId": ds_id, "doc_idx": doc_idx,
                "main_table_path": path, "file_size": fsize, "extension": ext,
            })
            need -= 1
            n_t2_ok += 1

        print(f"[acquisition] Tier-2 backfill: {n_t2_ok}")

    # Merge backfill results
    if backfill_dc:
        d_content = pd.concat(
            [d_content, pd.DataFrame(backfill_dc)], ignore_index=True,
        )
        main_tables = pd.concat(
            [main_tables, pd.DataFrame(backfill_mt)], ignore_index=True,
        )

    # Update slug_to_ref with Tier-2 additions
    if tier2_new_rows:
        slug_to_ref = pd.concat(
            [slug_to_ref, pd.DataFrame(tier2_new_rows)], ignore_index=True,
        )

    # Clean up non-tabular directories
    n_cleaned = 0
    for nid in non_tab_ids:
        ndir = tab_raw_dir / str(nid)
        if ndir.exists():
            shutil.rmtree(ndir)
            n_cleaned += 1
    if n_cleaned:
        print(f"[acquisition] Cleaned {n_cleaned} non-tabular directories")

    print(f"[acquisition] Backfill complete: d_content={len(d_content)}, "
          f"main_tables={len(main_tables)}")
    return d_content, main_tables, slug_to_ref


# ---------------------------------------------------------------------------
# Integrity check
# ---------------------------------------------------------------------------

def check_integrity(
    d_content: pd.DataFrame,
    main_tables: pd.DataFrame,
    index_map: pd.DataFrame,
    tab_raw_dir: Path,
    target: int,
) -> List[str]:
    """Run post-acquisition integrity checks.

    Returns a list of error strings (empty = all checks passed).
    """
    errors = []

    # 1. Row counts
    if len(d_content) != target:
        errors.append(f"d_content rows {len(d_content)} != target {target}")
    if len(main_tables) != target:
        errors.append(f"main_tables rows {len(main_tables)} != target {target}")

    # 2. ID consistency
    dc_ids = set(d_content["Id"].astype(int).values)
    mt_ids = set(main_tables["DatasetId"].astype(int).values)
    if dc_ids != mt_ids:
        diff = len(dc_ids.symmetric_difference(mt_ids))
        errors.append(f"ID mismatch between d_content and main_tables: {diff} IDs differ")

    # 3. No duplicates
    if d_content["Id"].duplicated().any():
        errors.append(f"d_content has {d_content['Id'].duplicated().sum()} duplicate IDs")
    if main_tables["DatasetId"].duplicated().any():
        errors.append(f"main_tables has {main_tables['DatasetId'].duplicated().sum()} duplicate IDs")

    # 4. All main_table_path non-empty
    n_empty = (main_tables["main_table_path"] == "").sum()
    if n_empty > 0:
        errors.append(f"{n_empty} datasets still have no main table")

    # 5. Directories exist
    n_missing = 0
    for did in dc_ids:
        ddir = tab_raw_dir / str(did)
        if not ddir.exists() or not any(ddir.iterdir()):
            n_missing += 1
    if n_missing > 0:
        errors.append(f"{n_missing} dataset directories missing or empty")

    # 6. doc_idx consistency
    imap_lookup = dict(zip(index_map["Id"].astype(int), index_map["doc_idx"].astype(int)))
    n_mismatch = 0
    for _, row in d_content.iterrows():
        did = int(row["Id"])
        expected = imap_lookup.get(did)
        if expected is not None and int(row["doc_idx"]) != expected:
            n_mismatch += 1
    if n_mismatch > 0:
        errors.append(f"{n_mismatch} doc_idx values don't match index_map")

    return errors
