#!/bin/bash
bfile_path="/mnt/project/merged_data"
clump_input="/opt/notebooks/tg/gwas.txt"

plink \
	--bfile $bfile_path \
	--clump $clump_input \
	--clump-p1 1.96e-8 \
	--clump-p2 1e-3 \
	--clump-r2 0.1 \
	--clump-kb 250 \
	--out "tg.clumped"
