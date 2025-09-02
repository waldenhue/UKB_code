import pandas as pd
from cyvcf2 import VCF
import sys

df = pd.read_parquet('chr/chr1.parquet',engine='pyarrow')

df['BP'] = df['BP'].astype(int)



for block in list(set(df.Block.tolist())):
    concat_list = []
    df_c = df[df.Block == block]
    chrom = df_c.CHR.tolist()[0]
    snps_list = df_c.BP.tolist()
    snps_list.sort()
    vcf_tmp = VCF('/mnt/project/Bulk/Exome sequences/Population level exome OQFE variants, pVCF format - final release/'+ block + '_v1.vcf.gz',threads =12,gts012=True,strict_gt=True)
    samples_index = vcf_tmp.samples
    for i in vcf_tmp('chr'+str(chrom)+':'+str(snps_list[0])+'-'+str(snps_list[-1])):
        pos = i.POS
        if ((pos in snps_list)&(i.FILTER !='MONOALLELIC')&(i.aaf < 0.01)):
            df_tmp = pd.DataFrame({'chr'+str(chrom)+':'+str(pos):i.gt_types},index=samples_index)
            concat_list.append(df_tmp)
    if len(concat_list)>0:        
        df_final = pd.concat(concat_list,axis = 1)
        df_final.to_parquet('./tg_rv/df_'+str(block) +'.parquet',engine = 'pyarrow', compression = 'snappy')

