#!/bin/bash

# Define the path to helper files
path_to_500kwes_helper_files="/mnt/project/Bulk/Exome sequences/Population level exome OQFE variants, PLINK format - final release/helper_files/"

# Loop over chromosomes 1 to 22
for i in {1..22}; do
    ./regenie_v2.2.4.gz_x86_64_Linux_mkl \
    --step 2 \
    --pred colon_train0_pred.list \
    --bgen "/mnt/project/Bulk/Exome sequences/Population level exome OQFE variants, BGEN format - final release/ukb23159_c${i}_b0_v1.bgen" \
    --ref-first \
    --sample "/mnt/project/Bulk/Exome sequences/Population level exome OQFE variants, BGEN format - final release/ukb23159_c${i}_b0_v1.sample" \
    --phenoFile colon_train0.phe \
    --covarFile colon_train0.phe \
    --phenoCol C18 \
    --covarColList age,sex,pc{1:10} \
    --set-list "${path_to_500kwes_helper_files}/ukb23158_500k_OQFE.sets.txt.gz" \
    --anno-file "${path_to_500kwes_helper_files}/ukb23158_500k_OQFE.annotations.txt.gz" \
    --mask-def masks.txt \
    --nauto 23 \
    --aaf-bins 0.01 \
    --bsize 200 \
    --bt \
    --out colon_burden_train0_chr${i}
done

