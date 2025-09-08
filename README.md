# UKB_code: analysis pipeline for UK Biobank 

This repo consists of scripts to: 
- extract common/rare genotype matrices from UKB VCF/pVCF,
- extract phenotypes,
- run GWAS and burden tests,
- try late integration of CV+RV signals,
- compare predictive models across variant sets (CV vs RV),

## Repository Structure 
```text
UKB_code/
├─ LICENSE
├─ README.md
├─ requirements.txt
├─ association_tests/
│ ├─ burden/
│ │ ├─ burden.sh
│ └─ gwas/
│ ├─ gwas.sh
│ └─ clumping.sh
├─ extracting_genotype/
│ ├─ extract_cv.py
│ └─ extract_rv.py
├─ integration_comparison/
│ └─ late_integration.py
├─ phenotype_extraction/
│ └─ continuous_phenotype.ipynb
└─ variant_comparison/
├─ lasso_aa_cv.py
├─ lasso_aa_rv.py
├─ ridge_aa_cv.py
├─ ridge_aa_rv.py
├─ ols_aa_cv.py
├─ ols_aa_rv.py
├─ rf_aa_cv.py
├─ rf_aa_rv.py
├─ xgb_aa_cv.py
└─ xgb_aa_rv.py
```

## Prerequisites 
- Python 3.9.12
- Required libraries: pandas, numpy, scipy, etc.
- Access to UK Biobank data through UKB RAP
- All necessary Python libraries (e.g., pandas, numpy, scipy) are listed in the requirements.txt 

## Installation

    git clone https://github.com/waldenhue/UKB_code.git  
    cd UKB_code
    pip install -r requirements.txt

## Workflows

A) Extracting Genotypes
  
  Common Variants (CV)
    
    python extracting_genotype/extract_cv.py

  Rare Variants (RV)

    python extracting_genotype/extract_rv.py

B) Extracting Phenotype

    jupyter notebook phenotype_extraction/continuous_phenotype.ipynb

C) Association Tests

  GWAS

    ./association_tests/gwas/gwas.sh

  Clumping

    ./association_tests/gwas/clumping.sh

  Burden Testing

    ./association_tests/burden/burden.sh

D) Late Integration (CV + RV)

    python integration_comparison/late_integration.py

E) Variant-set Model Comparison

  Compare common variants (CV) vs rare variants (RV) using:
  
  Lasso:
      
      python variant_comparison/lasso_aa_cv.py
      python variant_comparison/lasso_aa_rv.py

  Similarly there are python scripts for Ridge, OLS, Random Forest and XGBoost
  
## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.




