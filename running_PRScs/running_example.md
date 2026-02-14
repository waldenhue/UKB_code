# PRS pipeline example: standing_height (st)

This document demonstrates a complete PRS workflow using **standing_height (st)** as an example, including:

- GWAS summary formatting
- PRScs beta adjustment
- Posterior effect merging
- PRS calculation with PLINK

---

## 1) Prepare GWAS summary for PRScs

PRScs expects GWAS summary formatted like:

```text
SNP	A1	A2	BETA	SE
rs3115850	T	C	-0.00857501	0.004251
rs79373928	G	T	0.0112116	0.011633
rs57181708	G	A	0.0049968	0.004752
rs4422948	G	A	0.00389985	0.003367
rs4970383	A	C	0.00109734	0.003312
rs4970382	C	T	-0.000265803	0.002908
rs950122	C	G	0.00214148	0.003588
rs13303222	A	G	-0.00265445	0.003722
rs6657440	C	T	0.000920135	0.002919
```

- **A1** is the effect allele  
- **BETA** is the effect size of **A1**

However, REGENIE summary looks like:

```text
Name    Chr     Pos     Ref     Alt     Trait   Cohort  Model   Effect  LCI_Effect      UCI_Effect      Pval    AAF     Num_Cases       Cases_Ref Cases_Het        Cases_Alt       Num_Controls    Controls_Ref    Controls_Het    Controls_Alt    Info
rs3115850       1       825767  C       T       standing_height train0  ADD-WGR-LR      -0.00857501     -0.0169061      -0.000243959    0.0436582 0.139957 93532   68905   23073   1554    NA      NA      NA      NA      REGENIE_SE=0.004251;MAC=26181.000000
```

So we extract the needed fields (`SNP`, `Chr`, `Pos`, `Ref`, `Alt`, `BETA`, `SE`) like:

```Bash
zcat gwas_summary.regenie.gz | \
awk 'BEGIN{OFS="\t"}
NR==1 {
    print "SNP","Chr","Pos","Ref","Alt","BETA","SE"
    next
}
{
    se="NA"
    if (match($NF, /REGENIE_SE=([^;]+)/, a)) {
        se=a[1]
    }
    print $1,$2,$3,$4,$5,$9,se
}' > gwas_for_prscs.txt
```

Then rename columns to PRScs header (`SNP`, `A1`, `A2`, `BETA`, `SE`) in Python:


```python
import pandas as pd

df_gwas = pd.read_csv("gwas_for_prscs_st.txt", sep="\t")

# PRScs requires: SNP, A1, A2, BETA, SE
# Here Alt corresponds to A1 (effect allele), and Ref corresponds to A2.
df_gwas_formatted = df_gwas[["SNP", "Alt", "Ref", "BETA", "SE"]].rename(
    columns={"Alt": "A1", "Ref": "A2"}
)

df_gwas_formatted.to_csv(
    "/data/comics-hudh/prscs/gwas_formatted_st.tsv",
    sep="\t",
    index=None
)
```

Upload to `./prscs_tt/ `:

```bash
put ./tmp/gwas_formatted_st.tsv ./prscs_tt/gwas_formatted/
```

Match the SNPs:

```bash
awk '
BEGIN{
  FS=OFS="\t"
  comp["A"]="T"; comp["T"]="A"; comp["C"]="G"; comp["G"]="C"
  flip=0
}
function is_amb(a,b){
  return ( (a=="A"&&b=="T")||(a=="T"&&b=="A")||(a=="C"&&b=="G")||(a=="G"&&b=="C") )
}

NR==FNR{
  if(FNR==1) next
  a1[$2]=$4; a2[$2]=$5
  next
}

FNR==1{print; next}

{
  snp=$1; A1=$2; A2=$3; beta=$4; se=$5
  if(!(snp in a1)) next

  r1=a1[snp]; r2=a2[snp]

  # drop ambiguous SNPs
  if(is_amb(r1,r2)) next

  # same strand, same order
  if(A1==r1 && A2==r2){
    print snp, A1, A2, beta, se

  # same strand, flipped
  } else if(A1==r2 && A2==r1){
    flip++
    print snp, r1, r2, -beta, se

  } else {
    # try complement
    c1=comp[A1]; c2=comp[A2]

    if(c1==r1 && c2==r2){
      print snp, r1, r2, beta, se

    } else if(c1==r2 && c2==r1){
      flip++
      print snp, r1, r2, -beta, se
    }
  }
}

END{
  print "Flipped SNPs:", flip > "/dev/stderr"
}
' ldblk_ukbb_eur/snpinfo_ukbb_hm3 gwas_formatted_st.tsv \
> gwas_formatted_st.aligned.tsv
```

## 2) Beta adjustment with PRScs

Run `prscs_adjust_beta.lsf` in `./prscs_tt/`.

> [!WARNING]
> Rare variant IDs are NOT included in the provided LD matrix (`ldblk_ukbb_eur`):
> https://www.dropbox.com/s/t9opx2ty6ucrpib/ldblk_ukbb_eur.tar.gz?dl=0  
> Therefore, no beta adjustment for rare variants (RV) was performed.

> [!NOTE]
> `merged_data` is the PLINK file generated from chr[1-22] array data with MAF > 0.5 filtration.  
> Only the `.bim` file is used here.

Example LSF script:

```Bash
#!/bin/bash
#BSUB -J prscs[1-22]
#BSUB -q ser
#BSUB -n 10
#BSUB -R "span[ptile=10]"
#BSUB -e ./err_dir/prscs_%I.err
#BSUB -o ./out_dir/prscs_%I.out

cd $LS_SUBCWD

CHROM=$LSB_JOBINDEX

REF_DIR="./ldblk_ukbb_eur"
BIM_PREFIX="./merged_data"
SST_FILE="./prscs_tt/gwas_formatted_st.aligned.tsv"
N_GWAS=97843  # N = 97843 for st
OUT_DIR="./prscs_tt/adjust_beta_results/"

mkdir -p "$OUT_DIR"

python ./PRScs/PRScs.py \
    --ref_dir="$REF_DIR" \
    --bim_prefix="$BIM_PREFIX" \
    --sst_file="$SST_FILE" \
    --n_gwas="$N_GWAS" \
    --chrom="$CHROM" \
    --out_dir="$OUT_DIR" \
    2>&1 | tee "./log_dir/prscs_chr${CHROM}.log"
```

Generated files are named like:

- `prscs_tt_pst_eff_a1_b0.5_phiauto_chr[1-22].txt`
  
Meaning:

- `pst_eff`: posterior effect size
- `a1`: effect allele (A1)
- `phiauto`: phi chosen automatically from data

Example output format (chr, rsID, pos, A1, A2, pst_eff):

```Plain text
1	rs4970383	838555	A	C	5.095717e-05
1	rs3748592	880238	A	G	-2.268997e-04
1	rs3748597	888659	T	C	1.908535e-04
1	rs2341354	918573	A	G	3.658879e-05
1	rs35940137	940203	A	G	-4.040266e-06
1	rs3934834	1005806	T	C	-5.051741e-06
1	rs9442372	1018704	A	G	1.584368e-04
1	rs9651273	1031540	A	G	5.675811e-05
1	rs9442373	1062638	C	A	-5.674397e-05
1	rs9660710	1099342	A	C	4.063960e-05
```

Compare effect sizes before/after adjustment (example SNPs):

```Bash
awk 'NR==1 || $1 ~ /^(rs4970383|rs3748592|rs3748597|rs2341354|rs35940137)$/' \
    gwas_formatted_st.tsv
```

```Plain text
SNP	A1	A2	BETA	SE
rs4970383	A	C	0.00109734	0.003312
rs3748592	A	G	0.000734458	0.006373
rs3748597	T	C	0.000899093	0.006375
rs2341354	A	G	0.00283681	0.002906
rs35940137	A	G	-0.000170754	0.005941
```

## 3) Merge all `pst_eff` files

```Bash
cat ./prscs_tt/adjust_beta_results/prscs_tt_pst_eff_a1_b0.5_phiauto_chr*.txt \
  > ./prscs_tt/pst_eff_merged_st.txt
```

## 4) Calculate PRS with PLINK --`score`

Use `calculate_prs_with_cs.sh`:

>  `2 4 6` refers to: **variant ID col, allele col, score col**

```Bash
#!/bin/sh
position_dir="/extract_with_position/st"

run_plink="plink --bfile merged_data \
  --score pst_eff_merged_st.txt 2 4 6 sum \
  --out prs_st"

dx run swiss-army-knife \
  -iin="${position_dir}/merged_data.bed" \
  -iin="${position_dir}/merged_data.bim" \
  -iin="${position_dir}/merged_data.fam" \
  -iin="${position_dir}/pst_eff_merged_st.txt" \
  -icmd="${run_plink}" \
  --tag="prscs" \
  --instance-type "mem1_ssd1_v2_x36" \
  --destination="${position_dir}" \
  --brief --yes
  ```

Resulting `prs_st.profile` example:

```Plain text
      FID       IID  PHENO    CNT   CNT2 SCORESUM
  1000018   1000018    178 233748  54735 0.0583714
  1000051   1000051    151 236044  55266 -0.639069
  1000066   1000066    180 235462  55075 0.0439511
  1000084   1000084    163 235964  54762 -0.730994
  1000107   1000107    161 235898  55048 -0.202573
  1000135   1000135    177 234244  54714 -0.162166
  1000161   1000161  174.5 235920  54398 -0.288482
  1000172   1000172    175 235308  54947 -0.988155
  1000183   1000183    174 231818  54388 -0.720451
  1000199   1000199    185 235760  55132 -0.0344205
  1000233   1000233  173.5 234412  54641 0.233964
  1000249   1000249    177 235214  54858 -0.173351
```

---

## References

- PRScs: https://github.com/getian107/PRScs  
- PLINK 1.9: https://www.cog-genomics.org/plink/  
- REGENIE: https://github.com/rgcgithub/regenie  