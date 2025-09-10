from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.pipeline import Pipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor

## We take apolipoprotein A as an example here

def read_info(pheno_symbol, pheno_name, rv_type):
    "this function reads phenotype and genotype data from samples and returns train and test data"
    ## reading information of phenotpye and genotype from common and rare variants
    file_path = './'+pheno_symbol+'/'
    df_pheno = pd.read_parquet(file_path+'pheno.parquet',engine='pyarrow')
    df_cv = pd.read_parquet(file_path+'cv.parquet',engine='pyarrow')
    df_rv = pd.read_parquet(file_path+'rv.parquet',engine='pyarrow')
    ## standardize the index of cv dataframe
    cv_index = [i.split('_')[0] for i in df_cv.index.tolist()]
    df_cv.index = cv_index
    
    ## reading train and test ids both having pheno, cv and rv info
    with open(file_path+'test_id.txt','r') as file:
        test_ids = [line.strip() for line in file]
    with open(file_path+'train1_id.txt','r') as file:
        train_ids = [line.strip() for line in file]
    train_ids = list(set(df_pheno.index.tolist())&set(df_cv.index.tolist())&set(df_rv.index.tolist())&set(train_ids))
    test_ids = list(set(df_pheno.index.tolist())&set(df_cv.index.tolist())&set(df_rv.index.tolist())&set(test_ids))
    
    ## reading rare variant carriers in test set
    with open(file_path+'rv_carriers.txt','r') as file:
        rv_carriers_ids = [line.strip() for line in file]
    rv_carriers_test = list(set(rv_carriers_ids)&set(test_ids))
    
    ## making train and test dataframes
    df_pheno_train = df_pheno[df_pheno.index.isin(train_ids)]
    df_cv_train = df_cv[df_cv.index.isin(train_ids)]
    df_rv_train = df_rv[df_rv.index.isin(train_ids)]
    
    df_pheno_test = df_pheno[df_pheno.index.isin(test_ids)]
    df_cv_test = df_cv[df_cv.index.isin(test_ids)]
    df_rv_test = df_rv[df_rv.index.isin(test_ids)]

    df_rv_train = df_rv_train
    df_rv_test = df_rv_test
    rv_carriers_test = rv_carriers_test

    df_train_final  = pd.concat([df_pheno_train[pheno_name],df_cv_train,df_rv_train],axis=1)
    df_test_final  = pd.concat([df_pheno_test[pheno_name],df_cv_test,df_rv_test],axis=1)
    df_test_carriers_final = df_test_final[df_test_final.index.isin(rv_carriers_test)]

    feature_names = df_train_final.drop(pheno_name, axis=1).columns.tolist()

    X_train = df_train_final.drop(pheno_name,axis=1).values
    y_train = df_train_final[pheno_name].values
    X_test = df_test_final.drop(pheno_name,axis=1).values
    y_test = df_test_final[pheno_name].values
    X_carriers_test = df_test_carriers_final.drop(pheno_name,axis=1).values
    y_carriers_test = df_test_carriers_final[pheno_name].values
    
    return [X_train,X_test,X_carriers_test], [y_train,y_test,y_carriers_test],feature_names

# ===================== main =====================
if __name__ == "__main__":

    X_list, y_list, feature_names = read_info(pheno_symbol='aa', pheno_name='aa', rv_type='rv')
    X_train, X_test, X_test_rv = X_list
    y_train, y_test, y_test_rv = y_list

    # Pipeline
    linear_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("linear", LinearRegression())
    ])

    # model traning
    linear_pipeline.fit(X_train, y_train)
    
    # model estimation
    def estimate_model(model, X, y):
        y_pred = model.predict(X)
        return {
            "r2": r2_score(y, y_pred),
            "mse": mean_squared_error(y, y_pred)
        }
    
    all_metrics = estimate_model(linear_pipeline, X_test, y_test)
    carriers_metrics = estimate_model(linear_pipeline, X_test_rv, y_test_rv)

    list1 = ['AA','linear','cv+rv','all',all_metrics['r2'],all_metrics['mse']]
    list2 = ['AA','linear','cv+rv','carriers',carriers_metrics['r2'],carriers_metrics['mse']]

    ## output results
    df_report = pd.DataFrame([list1,list2],columns=['Trait','Model','Variant','Group','R2','MSE'])
    df_report.to_csv('./report/linear_aa_rv.csv')



