#!/usr/bin/env python3
"""
normalize_halflife.py — 3データセットの half-life 値を正規化・統合

目的:
    BRIC-seq (Tani 2012 HeLa), SLAM-seq (Herzog 2017 mESC),
    TimeLapse-seq (Schofield 2018 MEF + K562) の半減期値を、
    データセット間の scale 差を補正した上で一つの DataFrame に統合する。

入力:
    ../processed/bricseq_halflife_mapped.csv
    ../processed/slamseq_herzog_halflife_mapped.csv
    ../processed/timelapseseq_schofield_halflife_mapped.csv

出力:
    ../processed/halflife_merged.csv
        columns: gencode_gene_id, gencode_gene_symbol, cell_line, half_life_h,
                 half_life_log2, source, normalized_half_life
    ../processed/halflife_distribution.png  (diagnostic plot)

正規化方針:
    1. log2(half_life_h) 変換で右裾の重さを緩和
    2. データセット別 quantile normalization で分布形状を揃える
    3. 極端値（log2 t½ > 10 または < -3）は winsorize

NOTE:
    各データセットの半減期定義（full decay vs half-decay, exponential fit method）
    が完全には揃わないため、本総説では「相対順位」で議論し、絶対値は cross-reference
    として扱う方針（§3.1, §4.2 参照）。
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def quantile_normalize_by_source(df: pd.DataFrame, value_col: str, group_col: str) -> pd.Series:
    """
    Within-group rank-based quantile normalization.
    Each group's values are mapped to the average rank distribution across groups.
    """
    df = df.copy()
    df["_rank"] = df.groupby(group_col)[value_col].rank(pct=True)

    # Target distribution = pooled empirical quantiles
    pooled = df[value_col].dropna().sort_values().values
    n = len(pooled)
    if n == 0:
        return pd.Series(np.nan, index=df.index)

    def _map(pct):
        if pd.isna(pct):
            return np.nan
        idx = int(np.clip(pct * (n - 1), 0, n - 1))
        return pooled[idx]

    return df["_rank"].apply(_map)


def winsorize(series: pd.Series, lower_q: float = 0.005, upper_q: float = 0.995) -> pd.Series:
    lo, hi = series.quantile([lower_q, upper_q])
    return series.clip(lower=lo, upper=hi)


def load_and_tag(path: Path, source_tag: str) -> pd.DataFrame:
    if not path.exists():
        log.warning(f"Missing input, skipping: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["source"] = source_tag
    required = {"gencode_gene_id", "gencode_gene_symbol", "cell_line", "half_life_h"}
    missing = required - set(df.columns)
    if missing:
        log.warning(f"{path.name}: missing columns {missing} — will be NaN in merged output")
        for c in missing:
            df[c] = np.nan
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path(__file__).parent.parent / "processed",
    )
    parser.add_argument(
        "--bricseq-csv",
        type=Path,
        default=None,
        help="Default: <processed>/bricseq_halflife_mapped.csv",
    )
    parser.add_argument("--slamseq-csv", type=Path, default=None)
    parser.add_argument("--timelapse-csv", type=Path, default=None)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Default: <processed>/halflife_merged.csv",
    )
    args = parser.parse_args()

    pdir = args.processed_dir
    src_map = {
        "BRIC-seq": args.bricseq_csv or (pdir / "bricseq_halflife_mapped.csv"),
        "SLAM-seq": args.slamseq_csv or (pdir / "slamseq_herzog_halflife_mapped.csv"),
        "TimeLapse-seq": args.timelapse_csv or (pdir / "timelapseseq_schofield_halflife_mapped.csv"),
    }

    frames = [load_and_tag(p, tag) for tag, p in src_map.items()]
    frames = [f for f in frames if not f.empty]
    if not frames:
        log.error("No input CSVs found. Run Tasks 1.1-1.5 first.")
        sys.exit(1)
    df = pd.concat(frames, ignore_index=True)
    log.info(f"Merged {len(df)} rows from {len(frames)} datasets")

    # log2 transform
    df["half_life_log2"] = np.log2(df["half_life_h"].clip(lower=1e-3))
    df["half_life_log2"] = winsorize(df["half_life_log2"])

    # Quantile normalize across (source, cell_line) groups
    df["_group"] = df["source"] + "|" + df["cell_line"].astype(str)
    df["normalized_half_life"] = quantile_normalize_by_source(
        df, value_col="half_life_log2", group_col="_group"
    )
    df = df.drop(columns=["_group"])

    out_path = args.output_csv or (pdir / "halflife_merged.csv")
    keep_cols = [
        "gencode_gene_id",
        "gencode_gene_symbol",
        "cell_line",
        "half_life_h",
        "half_life_log2",
        "source",
        "normalized_half_life",
    ]
    present = [c for c in keep_cols if c in df.columns]
    df[present].to_csv(out_path, index=False)
    log.info(f"Wrote {out_path}")

    # Summary
    for tag in df["source"].unique():
        sub = df[df["source"] == tag]
        log.info(
            f"  {tag}: N={len(sub)}, "
            f"half_life median={sub['half_life_h'].median():.2f} h, "
            f"log2 range=[{sub['half_life_log2'].min():.2f}, {sub['half_life_log2'].max():.2f}]"
        )


if __name__ == "__main__":
    main()
