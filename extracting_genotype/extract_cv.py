import pandas as pd
from cyvcf2 import VCF
import sys

df = pd.read_parquet('./chr/chr1.parquet',engine='pyarrow')


df['CHR'] = df['CHR'].astype(int)
df['BP'] = df['BP'].astype(int)


for chrom in list(set(df.CHR.tolist())):
    comparison_table = []
    concat_list = []
    df_c = df[df.CHR== chrom]
    snps_list = df_c.BP.tolist()
    snps_list.sort()
    vcf_tmp = VCF('/mnt/project/lift_over_all/ukb22418_c'+str(chrom)+'_b0_v2_GRCh38_full_analysis_set_plus_decoy_hla.vcf.gz',threads =12,gts012=True,strict_gt=True)
    vcf_tmp_id = vcf_tmp.samples
    for i in vcf_tmp('chr'+str(chrom)+':'+str(snps_list[0])+'-'+str(snps_list[-1])):
        pos = i.POS
        if pos in snps_list:
            df_current = pd.DataFrame({'chr'+str(chrom)+':'+str(pos):i.gt_types},index = vcf_tmp_id)
            comparison_table.append(['chr'+str(chrom)+':'+str(pos),i.ID])
            concat_list.append(df_current)
    df_final = pd.concat(concat_list,axis = 1)
    df_comparison = pd.DataFrame(comparison_table)
    df_comparison.to_parquet('./comparison/chr'+str(chrom)+'.parquet',engine='pyarrow',compression='snappy')
    df_final.to_parquet('./genotype/chr'+str(chrom)+'.parquet',engine='pyarrow',compression='snappy')
