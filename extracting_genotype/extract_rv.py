#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract the genotype matrix for ONE BLOCK from a VCF, using SNP positions
listed in one Parquet file (with columns: CHR, BP, Block).

- Designed to run one block per process (e.g., with nohup & parallel).
- Output (per block):
  ./genotype/{block}.parquet     (samples x loci, values in {0,1,2,3})

Typical UKB exome pVCF path looks like:
  "... final release/{block}_v1.vcf.gz"
so we use a --vcf-template with "{block}" placeholder by default.

Example:
  python extracting_rv.py \
    --block-parquet ./block/ukb23157_c4_b25.parquet \
    --block ukb23157_c4_b25 \
    --vcf-template "/mnt/project/Bulk/Exome sequences/Population level exome OQFE variants, pVCF format - final release/{block}_v1.vcf.gz" \
    --threads 12 \
    --out-folder ./rv_gt \
    --contig_prefix chr \
    --max-aaf 0.01 \
    --exclude-filter MONOALLELIC

Parallel (examples):
  nohup python3 extracting_rv.py --block-parquet ./block/ukb23157_c4_b25.parquet --block ukb23157_c4_b25 > log.c4_b25 2>&1 &
  nohup python3 extracting_rv.py --block-parquet ./block/ukb23157_c7_b24.parquet --block ukb23157_c7_b24 > log.c7_b24 2>&1 &
"""

import os
import re
import argparse
import textwrap
import pandas as pd
from cyvcf2 import VCF


def infer_chrom_from_path(path: str) -> int | None:
    """
    Infer chromosome number from names like 'ukb23157_c4_b25.parquet'
    (capture the number after '_c').
    """
    m = re.search(r"_c(\d+)_", os.path.basename(path))
    return int(m.group(1)) if m else None


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract per-block genotypes from a VCF for given SNP positions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Notes
        -----
        - Input Parquet must contain 'BP' (base position). If 'CHR' exists, rows
          will be filtered to the inferred chromosome automatically.
        - Locus names are formatted as '{contig_prefix}{chrom}:{pos}' (e.g., 'chr4:12345').
        - gt_types from cyvcf2 are in {0,1,2,3}, where 3 usually denotes missing.

        Examples
        --------
        1) example for block ukb23157_c4_b25：
            python extracting_rv.py \
                --block-parquet ./block/ukb23157_c4_b25.parquet \
                --block ukb23157_c4_b25 \
                --vcf-template "/mnt/project/Bulk/Exome sequences/Population level exome OQFE variants, pVCF format - final release/{block}_v1.vcf.gz" \
                --threads 12 \
                --out-folder ./rv_gt \
                --contig_prefix chr \
                --max-aaf 0.01 \
                --exclude-filter MONOALLELIC
                               
        2) Parallel (examples):
            nohup python3 extracting_rv.py --block-parquet ./block/ukb23157_c4_b25.parquet --block ukb23157_c4_b25 > log.c4_b25 2>&1 &
            nohup python3 extracting_rv.py --block-parquet ./block/ukb23157_c7_b24.parquet --block ukb23157_c7_b24 > log.c7_b24 2>&1 &                                                                  
        """)
    )
    
    p.add_argument(
        "--block-parquet", required=True,
        help="Parquet file for one block (e.g., ./block/ukb23157_c4_b25.parquet)."
    )
    p.add_argument(
        "--block", required=True,
        help="Block key (e.g., ukb23157_c4_b25)."
    )
    p.add_argument(
        "--vcf-template",
        default="/mnt/project/Bulk/Exome sequences/Population level exome OQFE variants, pVCF format - final release/{block}_v1.vcf.gz",
        help="VCF path template with '{block}' placeholder."
    )
    p.add_argument("--threads", type=int, default=12, help="cyvcf2 VCF reader threads.")
    p.add_argument("--out-folder", default="./genotype", help="Output dir for genotype matrices.")
    p.add_argument("--contig_prefix", default="chr", help="VCF contig prefix: 'chr' or ''.")
    # Optional filters (match your original logic)
    p.add_argument("--max-aaf", type=float, default=None,
                   help="Keep variants with aaf < MAX_AAF (default: None = no MAF filter).")
    p.add_argument("--exclude-filter", default=None,
                   help="Exclude variants whose FILTER equals this string (e.g., MONOALLELIC).")
    return p.parse_args()


def _first_alt_aaf(aaf):
    """Return a float allele frequency for the first ALT if possible."""
    if aaf is None:
        return None
    try:
        # array-like (numpy/list)
        if hasattr(aaf, "__len__") and not isinstance(aaf, (str, bytes)):
            return float(aaf[0]) if len(aaf) > 0 else None
        return float(aaf)  # scalar
    except Exception:
        return None


def main():
    args = parse_args()

    # --- Read block parquet ---
    if not os.path.exists(args.block_parquet):
        raise FileNotFoundError(f"Input Parquet not found: {args.block_parquet}")
    df = pd.read_parquet(args.block_parquet, engine="pyarrow")

    # --- Infer chromosome from filename, then optionally filter CHR column ---
    chrom = infer_chrom_from_path(args.block_parquet)
    if chrom is None:
        if "CHR" not in df.columns:
            raise ValueError("Cannot infer chromosome: CHR column missing and filename has no '_cN_'.")
        uniq = (
            df["CHR"].astype(str).str.extract(r"(\d+)", expand=False).dropna().astype(int).unique()
        )
        if uniq.size != 1:
            raise ValueError(f"Cannot infer a single chromosome from CHR column: found {uniq}.")
        chrom = int(uniq[0])

    print(f"[INFO] Block: {args.block}", flush=True)
    print(f"[INFO] Chromosome inferred: {chrom}", flush=True)

    # --- Clean/prepare positions ---
    if "BP" not in df.columns:
        raise ValueError("Input Parquet must contain column 'BP'.")

    if "CHR" in df.columns:
        df = df.copy()
        df["CHR"] = df["CHR"].astype(str).str.extract(r"(\d+)", expand=False).astype("Int64")
        df = df[df["CHR"] == chrom]

    df = df.dropna(subset=["BP"]).copy()
    df["BP"] = df["BP"].astype(int)
    df = df.drop_duplicates(subset=["BP"])

    if df.empty:
        print(f"[WARN] No positions found for CHR{chrom} in {args.block_parquet}. Nothing to do.", flush=True)
        return

    positions = df["BP"].to_numpy()
    positions.sort()
    pos_set = set(positions)

    # --- Build VCF path and region ---
    vcf_path = args.vcf_template.format(block=args.block)
    if not os.path.exists(vcf_path):
        raise FileNotFoundError(f"VCF not found: {vcf_path}")

    region = f"{args.contig_prefix}{chrom}:{int(positions[0])}-{int(positions[-1])}"
    print(f"[INFO] VCF    : {vcf_path}", flush=True)
    print(f"[INFO] Region : {region}", flush=True)

    vcf = VCF(vcf_path, threads=args.threads, gts012=True, strict_gt=True)
    samples = vcf.samples

    # --- Scan region & collect ---
    gt_cols = []  # list[pd.DataFrame] — each is one locus column

    for rec in vcf(region):
        p = rec.POS
        if p not in pos_set:
            continue

        # Optional filters
        if args.exclude_filter is not None and rec.FILTER == args.exclude_filter:
            continue
        if args.max_aaf is not None:
            aaf = _first_alt_aaf(rec.aaf)
            if aaf is not None and aaf >= args.max_aaf:
                continue

        locus = f"{args.contig_prefix}{chrom}:{p}"
        gt_cols.append(pd.DataFrame({locus: rec.gt_types}, index=samples))

    if not gt_cols:
        print(f"[INFO] No positions matched (after filters) in region for block {args.block}. Skip output.", flush=True)
        return

    df_final = pd.concat(gt_cols, axis=1)  # samples x loci

    os.makedirs(args.out_folder, exist_ok=True)
    out_gt_path = os.path.join(args.out_folder, f"{args.block}.parquet")

    df_final.to_parquet(out_gt_path, engine="pyarrow", compression="snappy")

    print(
        f"[OK] Wrote genotype: {out_gt_path}\n"
        f"[OK] Shape: {df_final.shape} (rows=samples, cols=loci)",
        flush=True,
    )


if __name__ == "__main__":
    main()
