#!/bin/bash

# Set script to exit on error
set -e

# Define variables (optional, for better readability)
REGENIE="./regenie_v2.2.4.gz_x86_64_Linux_mkl"
BED_FILE="/mnt/project/merged_data"
PHENO_FILE="asthma_train0.pheno"
PRED_FILE="merged_data_pred.list"
SNPLIST="/mnt/project/qc_for_array/all_filtered.snplist"
OUT_PREFIX="gwas"

# Run REGENIE Step 2
$REGENIE --step 2 \
    --bed $BED_FILE \
    --phenoFile $PHENO_FILE \
    --bsize 200 \
    --pThresh 0.01 \
    --firth --approx \
	--bt \
    --test additive \
    --pred $PRED_FILE \
    --gz \
    --extract $SNPLIST \
    --covarFile $PHENO_FILE \
    --minMAC 3 \
    --phenoColList asthma \
    --covarColList sex,age,pc1,pc2,pc3,pc4,pc5,pc6,pc7,pc8,pc9,pc10 \
    --htp $OUT_PREFIX \
    --out $OUT_PREFIX


