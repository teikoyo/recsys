#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Expand Content Coverage to 10K / 50K / 100K Datasets

Incremental data acquisition script that extends D_content from 1,000 to
larger scales.  Reuses existing downloads and API cache, supports
checkpoint/resume, and applies progressively relaxed filters.

Usage:
    python scripts/expand_content_coverage.py --target 10000
    python scripts/expand_content_coverage.py --target 50000
    python scripts/expand_content_coverage.py --target 100000

Outputs (for target N):
    tmp/content/scale_{N}/d_content.parquet
    tmp/content/scale_{N}/main_tables.parquet
    tmp/content/scale_{N}/slug_to_ref.csv
    data/tabular_raw/{Id}/  (shared download directory)
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

from src.content.acquisition import (
    SEARCH_MARGIN,
    backfill_non_tabular,
    check_integrity,
    check_kaggle_api,
    download_dataset,
    filter_candidates,
    get_filter_preset,
    match_slug_to_ref,
    search_kaggle_slug,
)
from src.content.sampling import select_main_table

# Raise CSV field size limit
csv.field_size_limit(sys.maxsize)


def parse_args():
    p = argparse.ArgumentParser(
        description="Expand content coverage to 10K/50K/100K datasets",
    )
    p.add_argument(
        "--target", type=int, required=True,
        help="Target number of content datasets (e.g. 10000, 50000, 100000)",
    )
    p.add_argument(
        "--checkpoint-every", type=int, default=100,
        help="Save checkpoint every N datasets",
    )
    p.add_argument(
        "--download-timeout", type=int, default=300,
        help="Download timeout in seconds",
    )
    p.add_argument(
        "--max-retries", type=int, default=2,
        help="Download retry count",
    )
    return p.parse_args()


def load_metadata(data_dir: Path, raw_dir: Path):
    """Load and merge metadata_merged.csv with DatasetVersions.csv."""
    print("[expand] Loading metadata...")
    meta = pd.read_csv(data_dir / "metadata_merged.csv", engine="python")
    print(f"  metadata_merged: {meta.shape}")

    dv_cols = ["Id", "DatasetId", "VersionNumber", "TotalCompressedBytes"]
    dv = pd.read_csv(raw_dir / "DatasetVersions.csv", usecols=dv_cols, engine="python")
    dv = dv.dropna(subset=["DatasetId", "VersionNumber"])
    idx_latest = dv.groupby("DatasetId")["VersionNumber"].idxmax()
    dv_latest = dv.loc[idx_latest, ["DatasetId", "TotalCompressedBytes"]].copy()
    dv_latest.rename(columns={"DatasetId": "Id"}, inplace=True)
    del dv

    meta = meta.merge(dv_latest, on="Id", how="left")
    print(f"  After join: {meta.shape}, "
          f"TotalCompressedBytes notna: {meta['TotalCompressedBytes'].notna().sum()}")
    return meta


def main():
    args = parse_args()
    target = args.target

    DATA_DIR = ROOT / "data"
    RAW_DIR = DATA_DIR / "raw_data"
    TMP_DIR = ROOT / "tmp"
    CONTENT_DIR = TMP_DIR / "content"
    TAB_RAW_DIR = DATA_DIR / "tabular_raw"
    CACHE_DIR = CONTENT_DIR / "api_cache"
    SCALE_DIR = CONTENT_DIR / f"scale_{target}"

    for d in [CONTENT_DIR, CACHE_DIR, TAB_RAW_DIR, SCALE_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 0. Check for existing outputs (resume support)
    # ------------------------------------------------------------------
    dc_path = SCALE_DIR / "d_content.parquet"
    mt_path = SCALE_DIR / "main_tables.parquet"
    sr_path = SCALE_DIR / "slug_to_ref.csv"

    if dc_path.exists() and mt_path.exists():
        existing_dc = pd.read_parquet(dc_path, engine="fastparquet")
        existing_mt = pd.read_parquet(mt_path, engine="fastparquet")
        if len(existing_dc) >= target and len(existing_mt) >= target:
            print(f"[expand] Target {target} already met: "
                  f"d_content={len(existing_dc)}, main_tables={len(existing_mt)}")
            print("[expand] Skipping acquisition. Delete outputs to re-run.")
            return

    # ------------------------------------------------------------------
    # 1. Load metadata
    # ------------------------------------------------------------------
    meta = load_metadata(DATA_DIR, RAW_DIR)

    # Load index_map
    index_map = pd.read_parquet(TMP_DIR / "index_map.parquet", engine="fastparquet")
    imap_lookup = dict(zip(index_map["Id"].astype(int), index_map["doc_idx"].astype(int)))

    # ------------------------------------------------------------------
    # 2. Filter candidates using target-appropriate preset
    # ------------------------------------------------------------------
    preset = get_filter_preset(target)
    print(f"\n[expand] Target: {target}, filter preset: {preset}")

    candidates = filter_candidates(
        meta,
        tags=preset["tags"],
        size_min=preset["size_min"],
        size_max=preset["size_max"],
        min_views=preset["min_views"],
    )
    print(f"[expand] Candidate pool: {len(candidates)}")

    if len(candidates) < target:
        print(f"[expand] WARNING: Candidate pool ({len(candidates)}) < target ({target})")

    # ------------------------------------------------------------------
    # 3. Load existing slug_to_ref (global + scale-specific)
    # ------------------------------------------------------------------
    global_sr_path = CONTENT_DIR / "slug_to_ref.csv"
    slug_to_ref = pd.DataFrame(columns=["Id", "Slug", "Title", "ref", "confidence"])

    if sr_path.exists():
        slug_to_ref = pd.read_csv(sr_path)
        print(f"[expand] Loaded scale slug_to_ref: {len(slug_to_ref)}")
    elif global_sr_path.exists():
        slug_to_ref = pd.read_csv(global_sr_path)
        print(f"[expand] Loaded global slug_to_ref: {len(slug_to_ref)}")

    # ------------------------------------------------------------------
    # 4. Check Kaggle API
    # ------------------------------------------------------------------
    kaggle_available = check_kaggle_api()
    if kaggle_available:
        print("[expand] Kaggle API available")
    else:
        print("[expand] WARNING: Kaggle API not available")

    # ------------------------------------------------------------------
    # 5. Determine search pool and run API matching
    # ------------------------------------------------------------------
    n_search = int(target * SEARCH_MARGIN)
    search_pool = candidates.nlargest(n_search, "TotalDownloads")
    print(f"\n[expand] Search pool: {len(search_pool)} (target*{SEARCH_MARGIN}={n_search})")

    already_searched = set(slug_to_ref["Slug"].values) if len(slug_to_ref) > 0 else set()
    to_search = search_pool[~search_pool["Slug"].isin(already_searched)]
    print(f"[expand] Already searched: {len(already_searched)}, remaining: {len(to_search)}")

    if kaggle_available and len(to_search) > 0:
        # Determine max_size_mb from preset
        max_size_mb = preset["size_max"] // (1024 * 1024)

        new_rows = []
        t0 = time.time()
        for i, (_, row) in enumerate(to_search.iterrows()):
            slug = row["Slug"]
            title = row["Title"]
            ds_id = row["Id"]

            results = search_kaggle_slug(slug, CACHE_DIR, max_size_mb)
            ref, confidence = match_slug_to_ref(slug, title, results)

            new_rows.append({
                "Id": ds_id, "Slug": slug, "Title": title,
                "ref": ref if ref else "", "confidence": confidence,
            })

            # Checkpoint
            if (i + 1) % args.checkpoint_every == 0:
                incr_df = pd.DataFrame(new_rows)
                slug_to_ref = pd.concat([slug_to_ref, incr_df], ignore_index=True)
                slug_to_ref.to_csv(sr_path, index=False)
                new_rows = []
                matched = (slug_to_ref["confidence"] != "none").sum()
                elapsed = time.time() - t0
                print(f"  [expand] Searched {i+1}/{len(to_search)} "
                      f"(matched: {matched}, elapsed: {elapsed:.0f}s)")

        # Save remaining
        if new_rows:
            slug_to_ref = pd.concat(
                [slug_to_ref, pd.DataFrame(new_rows)], ignore_index=True,
            )
        slug_to_ref.to_csv(sr_path, index=False)
        print(f"[expand] API search complete: {len(slug_to_ref)} total rows")

    # ------------------------------------------------------------------
    # 6. Build d_content from matched results
    # ------------------------------------------------------------------
    matched = slug_to_ref[slug_to_ref["confidence"] != "none"].copy()
    matched["Id"] = matched["Id"].astype(int)
    matched = matched.merge(
        candidates[["Id", "TotalDownloads", "TotalViews"]],
        on="Id", how="left",
    )
    # Drop rows without download info (not in candidates)
    matched = matched.dropna(subset=["TotalDownloads"])

    d_content = matched.nlargest(target, "TotalDownloads").copy()
    d_content["doc_idx"] = d_content["Id"].map(imap_lookup)
    d_content = d_content.dropna(subset=["doc_idx"])
    d_content["doc_idx"] = d_content["doc_idx"].astype(int)
    d_content = d_content[
        ["doc_idx", "Id", "Slug", "ref", "TotalDownloads", "TotalViews"]
    ].reset_index(drop=True)

    print(f"\n[expand] d_content candidates: {len(d_content)}")

    # ------------------------------------------------------------------
    # 7. Batch download (incremental)
    # ------------------------------------------------------------------
    if kaggle_available and (d_content["ref"] != "").any():
        to_download = d_content[d_content["ref"] != ""].copy()
        n_downloaded = 0
        n_skipped = 0
        n_failed = 0
        failed_list = []

        t0 = time.time()
        for i, (_, row) in enumerate(to_download.iterrows()):
            ds_id = int(row["Id"])
            ref = row["ref"]
            ds_dir = TAB_RAW_DIR / str(ds_id)

            # Skip if already downloaded
            if ds_dir.exists() and any(ds_dir.iterdir()):
                n_skipped += 1
                continue

            success = download_dataset(
                ref, ds_id, ds_dir,
                timeout=args.download_timeout,
                max_retries=args.max_retries,
            )

            if success:
                n_downloaded += 1
            else:
                n_failed += 1
                failed_list.append({"Id": ds_id, "ref": ref})

            if (i + 1) % args.checkpoint_every == 0:
                elapsed = time.time() - t0
                print(f"  [expand] Download progress: {i+1}/{len(to_download)} "
                      f"(new={n_downloaded}, skip={n_skipped}, fail={n_failed}, "
                      f"elapsed={elapsed:.0f}s)")

        print(f"\n[expand] Download complete: "
              f"{n_downloaded} new, {n_skipped} skipped, {n_failed} failed")

    # ------------------------------------------------------------------
    # 8. Main table selection
    # ------------------------------------------------------------------
    print("\n[expand] Selecting main tables...")
    table_rows = []
    for i, (_, row) in enumerate(d_content.iterrows()):
        ds_id = int(row["Id"])
        doc_idx = int(row["doc_idx"])
        ds_dir = TAB_RAW_DIR / str(ds_id)

        path, fsize, ext = select_main_table(ds_dir)
        table_rows.append({
            "DatasetId": ds_id,
            "doc_idx": doc_idx,
            "main_table_path": path if path else "",
            "file_size": fsize,
            "extension": ext if ext else "",
        })

        if (i + 1) % 1000 == 0:
            print(f"  [expand] Main table selection: {i+1}/{len(d_content)}")

    main_tables = pd.DataFrame(table_rows)
    n_found = (main_tables["main_table_path"] != "").sum()
    print(f"[expand] Tables found: {n_found}/{len(main_tables)} "
          f"({n_found/max(len(main_tables),1)*100:.1f}%)")

    # ------------------------------------------------------------------
    # 9. Backfill non-tabular datasets
    # ------------------------------------------------------------------
    d_content, main_tables, slug_to_ref = backfill_non_tabular(
        d_content=d_content,
        main_tables=main_tables,
        candidates=candidates,
        slug_to_ref=slug_to_ref,
        index_map=index_map,
        tab_raw_dir=TAB_RAW_DIR,
        cache_dir=CACHE_DIR,
        target=target,
        kaggle_available=kaggle_available,
        max_size_mb=preset["size_max"] // (1024 * 1024),
    )

    # ------------------------------------------------------------------
    # 10. Integrity check
    # ------------------------------------------------------------------
    print("\n[expand] Running integrity checks...")
    # Use min of target and actual count for the check
    actual_target = min(target, len(d_content))
    errors = check_integrity(d_content, main_tables, index_map, TAB_RAW_DIR, actual_target)

    if errors:
        print(f"[expand] {len(errors)} issues found:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("[expand] All checks passed!")

    # ------------------------------------------------------------------
    # 11. Save outputs
    # ------------------------------------------------------------------
    d_content.to_parquet(dc_path, engine="fastparquet")
    main_tables.to_parquet(mt_path, engine="fastparquet")
    slug_to_ref.to_csv(sr_path, index=False)

    print(f"\n[expand] Saved to {SCALE_DIR}:")
    print(f"  d_content.parquet:   {len(d_content)} rows")
    print(f"  main_tables.parquet: {len(main_tables)} rows")
    print(f"  slug_to_ref.csv:     {len(slug_to_ref)} rows")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"EXPANSION COMPLETE: target={target}")
    print(f"{'=' * 60}")
    print(f"  D_content size:      {len(d_content)}")
    print(f"  Tables found:        {(main_tables['main_table_path'] != '').sum()}")
    print(f"  API matches:         {(slug_to_ref['confidence'] != 'none').sum()}")
    if len(d_content) > 0:
        print(f"  Downloads range:     "
              f"[{d_content['TotalDownloads'].min():,.0f}, "
              f"{d_content['TotalDownloads'].max():,.0f}]")


if __name__ == "__main__":
    main()
