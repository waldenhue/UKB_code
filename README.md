# UKB_code: analysis pipeline for UK Biobank 

This repo consists of scripts to: 
- extract common/rare genotype matrices from UKB VCF/pVCF,
- extract phenotypes,
- run GWAS and burden tests,
- try late integration of CV+RV signals,
- compare predictive models across variant sets (CV vs RV),

## Repository Structure 
```text
UKB_code
├── LICENSE
├── README.md
├── association_tests
│   ├── burden
│   │   └── burden.sh
│   └── gwas
│       ├── clumping.sh
│       └── gwas.sh
├── extracting_genotype
│   ├── extract_cv.py
│   └── extract_rv.py
├── model_train
│   ├── model.py
│   ├── run_prs.py
│   └── settle_data.py
├── model_train_old
│   ├── integration_comparison
│   │   └── late_integration.py
│   └── variant_comparison
│       ├── lasso_aa_cv.py
│       ├── lasso_aa_rv.py
│       ├── ols_aa_cv.py
│       ├── ols_aa_rv.py
│       ├── rf_aa_cv.py
│       ├── rf_aa_rv.py
│       ├── ridge_aa_cv.py
│       ├── ridge_aa_rv.py
│       ├── xgb_aa_cv.py
│       └── xgb_aa_rv.py
├── phenotype_extraction
│   └── continuous_phenotype.ipynb
└── requirements.txt
```

## Prerequisites 
- Python 3.9.12
- All necessary Python libraries (e.g., pandas, numpy, scipy) are listed in the requirements.txt 
- Access to UK Biobank data through UKB RAP


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

D) Model Training and Testing

    python run_train.py --param_id 1 --pheno_symbol st --variant cv --strategy early --model lasso --output_dir ./results --data_dir ./data
  
## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.




