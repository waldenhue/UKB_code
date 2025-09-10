#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract the genotype matrix for a single chromosome from a VCF,
using SNP positions listed in one Parquet file (CHR, BP).
- Run one chromosome per process to parallelize with nohup.
- Outputs:
  1) ./comparison/chr{chrom}.parquet  (two columns: locus, ID)
  2) ./genotype/chr{chrom}.parquet    (samples x loci, values in {0,1,2,3})
"""

import os
import re
import argparse
import textwrap
import pandas as pd
from cyvcf2 import VCF


def infer_chrom_from_path(parquet_path: str) -> int | None:
    """Infer chromosome number from filename like 'chr1.parquet'."""
    m = re.search(r"chr(\d+)", os.path.basename(parquet_path))
    return int(m.group(1)) if m else None

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract per-chromosome genotypes from VCF for given SNP positions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples
        --------
        1) example for chromosome 1（chr1）：
           python3 extract_chr.py \\
             --chr-parquet ./chr/chr1.parquet \\
             --chrom 1 \\
             --vcf-template "/mnt/project/lift_over_all/ukb22418_c{chrom}_b0_v2_GRCh38_full_analysis_set_plus_decoy_hla.vcf.gz" \\
             --threads 12 \\
             --out-comp ./comparison \\
             --out-gt ./genotype \\
             --contig-prefix chr

        2) Run multiple chromosomes in parallel using nohup (example)：
           nohup python3 extract_chr.py --chr-parquet ./chr/chr1.parquet --chrom 1 > log.chr1 2>&1 &
           nohup python3 extract_chr.py --chr-parquet ./chr/chr2.parquet --chrom 2 > log.chr2 2>&1 &
        """)
    )
    p.add_argument(
        "--chr-parquet",
        required=True,
        help="Input Parquet file containing columns CHR and BP (e.g., ./chr/chr1.parquet).",
    )
    p.add_argument(
        "--chrom",
        type=int,
        default=None,
        help="Chromosome number (e.g., 1). If omitted, will try to infer from file name or CHR column.",
    )
    p.add_argument(
        "--vcf-template",
        default="/mnt/project/lift_over_all/ukb22418_c{chrom}_b0_v2_GRCh38_full_analysis_set_plus_decoy_hla.vcf.gz",
        help="VCF path template with {chrom} placeholder.",
    )
    p.add_argument("--threads", type=int, default=12, help="Number of VCF reader threads (cyvcf2).")
    p.add_argument("--out-comp", default="./comparison", help="Output directory for comparison table.")
    p.add_argument("--out-gt", default="./genotype", help="Output directory for genotype matrix.")
    p.add_argument("--contig-prefix", default="chr", help="VCF contig prefix: 'chr' or ''.")
    return p.parse_args()




def main():
    args = parse_args()

    # --- Read Parquet ---
    if not os.path.exists(args.chr_parquet):
        raise FileNotFoundError(f"Input Parquet not found: {args.chr_parquet}")
    df = pd.read_parquet(args.chr_parquet, engine="pyarrow")

    # --- Determine chromosome ---
    chrom = args.chrom or infer_chrom_from_path(args.chr_parquet)
    if chrom is None:
        # Fallback: try to infer from CHR column
        if "CHR" not in df.columns:
            raise ValueError("Cannot infer chromosome: CHR column missing and filename has no 'chrN'.")
        tmp = (
            df["CHR"]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .dropna()
            .astype(int)
            .unique()
        )
        if tmp.size != 1:
            raise ValueError(f"Cannot infer a single chromosome from CHR column: found {tmp}.")
        chrom = int(tmp[0])

    print(f"[INFO] Chromosome: {chrom}", flush=True)

    # --- Clean & keep positions for this chromosome only ---
    if "BP" not in df.columns:
        raise ValueError("Input Parquet must contain column 'BP'.")

    # If CHR column exists, filter by chrom; otherwise assume file is per-chromosome already.
    if "CHR" in df.columns:
        df = df.copy()
        df["CHR"] = (
            df["CHR"].astype(str).str.extract(r"(\d+)", expand=False).astype("Int64")
        )
        df = df[df["CHR"] == chrom]

    df = df.dropna(subset=["BP"]).copy()
    df["BP"] = df["BP"].astype(int)
    df = df.drop_duplicates(subset=["BP"])

    if df.empty:
        print(f"[WARN] No positions found for CHR{chrom} in {args.chr_parquet}. Nothing to do.", flush=True)
        return

    positions = df["BP"].to_numpy()
    positions.sort()
    pos_set = set(positions)

    # --- Prepare VCF & region ---
    vcf_path = args.vcf_template.format(chrom=chrom)
    if not os.path.exists(vcf_path):
        raise FileNotFoundError(f"VCF not found: {vcf_path}")

    region = f"{args.contig_prefix}{chrom}:{int(positions[0])}-{int(positions[-1])}"
    print(f"[INFO] VCF: {vcf_path}", flush=True)
    print(f"[INFO] Region: {region}", flush=True)

    vcf = VCF(vcf_path, threads=args.threads, gts012=True, strict_gt=True)
    samples = vcf.samples

    # --- Collect matched variants ---
    gt_cols = []     # list of DataFrames, each is a single-locus column
    comp_rows = []   # list of tuples (locus, rsID)

    for rec in vcf(region):
        p = rec.POS
        if p in pos_set:
            locus = f"{args.contig_prefix}{chrom}:{p}"
            gt_cols.append(pd.DataFrame({locus: rec.gt_types}, index=samples))
            comp_rows.append((locus, rec.ID))

    if not gt_cols:
        print(f"[INFO] No positions matched within region for CHR{chrom}. Skip outputs.", flush=True)
        return

    df_final = pd.concat(gt_cols, axis=1)
    df_comp = pd.DataFrame(comp_rows, columns=["locus", "ID"])

    os.makedirs(args.out_comp, exist_ok=True)
    os.makedirs(args.out_gt, exist_ok=True)

    out_comp_path = os.path.join(args.out_comp, f"chr{chrom}.parquet")
    out_gt_path = os.path.join(args.out_gt, f"chr{chrom}.parquet")

    df_comp.to_parquet(out_comp_path, engine="pyarrow", compression="snappy")
    df_final.to_parquet(out_gt_path, engine="pyarrow", compression="snappy")

    print(
        f"[OK] Wrote:\n"
        f"  comparison: {out_comp_path}\n"
        f"  genotype  : {out_gt_path}\n"
        f"  shapes    : genotype={df_final.shape} (rows=samples, cols=loci)",
        flush=True,
    )


if __name__ == "__main__":
    main()
