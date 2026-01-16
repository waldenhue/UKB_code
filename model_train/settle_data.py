from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pandas as pd
import numpy as np

def adjust_phenotype_sklearn(df, phenotype_col, covariate_cols, standardize=False):

    # dealing with missing value
    data = df[[phenotype_col] + covariate_cols].dropna()
    
    if len(data) == 0:
        raise ValueError("There is missing value in data")
    
    # extract variables
    y = data[phenotype_col].values.reshape(-1, 1)
    X = data[covariate_cols].values
    
    # building linear model
    model = LinearRegression()
    model.fit(X, y)
    
    # get the prediction
    y_pred = model.predict(X)
    
    # calculate residuals regressing on covariables
    residuals = y.flatten() - y_pred.flatten()
    
    # choosing whether conducting standardization on 
    if standardize:
        residuals = (residuals - np.mean(residuals)) / np.std(residuals)
    
    # creating Series
    result = pd.Series(np.nan, index=df.index)
    result.loc[data.index] = residuals
    
    return result, model

def deal_with_strategy(df, strategy,variant_type):
    if strategy == 'late':
        # Early combination: combine rare variants into a single feature
        df_combined = df.sum(axis=1).to_frame(name=f'{variant_type}_prs')
        return df_combined
    elif strategy == 'early':
        # Late combination: keep rare variants separate
        return df

def read_info(pheno_symbol,variant='cv' ,base_dir=Path("/data/med-hudh/grouping2"),strategy='late'):
    """
    Read phenotype and genotype (common and optional rare variants) data
    for a given phenotype symbol.

    Returns:
        (df_pheno, df_cv) or (df_pheno, df_cv, df_rv)
    """
    ##folder = base_dir / pheno_symbol
    folder = base_dir
    # reading phenotypea and common variants

    ## dealing with phenotype data
    df_pheno = pd.read_parquet(folder / "pheno.parquet", engine="pyarrow")
    df_pheno.index = df_pheno.index.astype(str)
    pheno_value,_ = adjust_phenotype_sklearn(df_pheno, pheno_symbol, df_pheno.drop(pheno_symbol,axis=1).columns.tolist(), standardize=False)
    df_pheno.loc[:,pheno_symbol] = pheno_value
    df_pheno = df_pheno[pheno_symbol]

    ## dealing with common variants data
    df_cv = pd.read_parquet(folder / "cv.parquet", engine="pyarrow")
    df_cv.index = df_cv.index.astype(str)
    df_cv = deal_with_strategy(df_cv, strategy, 'cv')

    ## dealing with rare variants data
    df_rv = pd.read_parquet(folder / "rv.parquet", engine="pyarrow")
    df_rv.index = df_rv.index.astype(str)
    df_rv = deal_with_strategy(df_rv, strategy, 'rv')
    if variant == 'allin':
        return df_pheno, df_cv, df_rv
    elif variant == 'rv':
        return df_pheno, df_rv
    return df_pheno, df_cv




def return_X_y(id, df_list,output_path,phe,tag=""):
    "this function returns X and y based on given ids and dataframe list"
    df_id_list = [df.index.tolist() for df in df_list]
    df_id_set = set(df_id_list[0])
    for df_id in df_id_list[1:]:
        df_id_set = df_id_set & set(df_id)

    inter = set(id) & df_id_set
    id = list(inter)


    df_list = [x.loc[id] for x in df_list]
    y = df_list[0]
    X_list = df_list[1:]
    X = pd.concat(X_list, axis=1)
    return X, y



def split_train_test(df_list,train_id,test_id,carriers_id,output_dir,phe):
    "this function splits data into train and test sets based on given ids"

    ## splitting phenotype data
    ## reading train and test ids both having pheno and genotype data

    X_train, y_train = return_X_y(train_id, df_list,output_dir,phe,tag="train")
    X_test, y_test = return_X_y(test_id, df_list,output_dir,phe,tag="test")

    ## the carriers refer to sampes carrying the rare variants in testing set
    X_carriers, y_carriers = X_test[X_test.index.isin(carriers_id)], y_test[y_test.index.isin(carriers_id)]
    X_non_carriers, y_non_carriers = X_test[~X_test.index.isin(carriers_id)], y_test[~y_test.index.isin(carriers_id)]
    return (X_train, y_train), (X_test, y_test), (X_carriers, y_carriers),(X_non_carriers, y_non_carriers)

    



