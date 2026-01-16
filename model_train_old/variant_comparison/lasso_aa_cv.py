from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, Lasso  
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.pipeline import Pipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor

# We take apolipoprotein A as an example here

def read_info(pheno_symbol, pheno_name):
    "this function reads phenotype and genotype data from samples and returns train and test data"
    ## reading information of phenotpye and genotype from common and rare variants
    file_path = './'+pheno_symbol+'/'
    df_pheno = pd.read_parquet(file_path+'pheno.parquet',engine='pyarrow')
    df_cv = pd.read_parquet(file_path+'cv.parquet',engine='pyarrow')
    df_rv = pd.read_parquet(file_path+'rv.parquet',engine='pyarrow')
    ## standardize the index of cv dataframe
    cv_index = [i.split('_')[0] for i in df_cv.index.tolist()]
    df_cv.index = cv_index
    
    ## reading train and test ids both having pheno and cv info
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
    
    df_pheno_test = df_pheno[df_pheno.index.isin(test_ids)]
    df_cv_test = df_cv[df_cv.index.isin(test_ids)]


    df_train_final = pd.concat([df_pheno_train[pheno_name], df_cv_train], axis=1)
    df_test_final = pd.concat([df_pheno_test[pheno_name], df_cv_test], axis=1)
    df_test_rv_final = df_test_final[df_test_final.index.isin(rv_carriers_test)]
    
    feature_names = df_train_final.drop(pheno_name, axis=1).columns.tolist()
    
    X_train = df_train_final.drop(pheno_name, axis=1)  
    y_train = df_train_final[pheno_name].values
    X_test = df_test_final.drop(pheno_name, axis=1)
    y_test = df_test_final[pheno_name].values

    X_test_rv = df_test_rv_final.drop(pheno_name, axis=1)
    y_test_rv = df_test_rv_final[pheno_name].values

    return (X_train, X_test, X_test_rv), (y_train, y_test, y_test_rv), feature_names

# ===================== main =====================
if __name__ == "__main__":
    
    X_list, y_list, feature_names = read_info(pheno_symbol='aa', pheno_name='aa')
    X_train, X_test, X_test_rv = X_list
    y_train, y_test, y_test_rv= y_list
       
    extended_alphas = np.logspace(-5, 1, 20)  
    
    # Pipeline
    lasso_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", LassoCV(alphas=extended_alphas, cv=5, max_iter=10000))  
    ])
  
    # model training
    lasso_pipeline.fit(X_train, y_train)
        
    # best parameter and model
    best_alpha = lasso_pipeline.named_steps["lasso"].alpha_
    best_model = lasso_pipeline.named_steps["lasso"]
       
    # model estimation
    def estimate_model(model, X, y):
        y_pred = model.predict(X)
        return {
            "r2": r2_score(y, y_pred),
            "mse": mean_squared_error(y, y_pred)
        }
    
    all_metrics = estimate_model(lasso_pipeline, X_test, y_test)
    carriers_metrics = estimate_model(lasso_pipeline, X_test_rv, y_test_rv)

    list1 = ['AA','lasso','cv','all',all_metrics['r2'],all_metrics['mse']]
    list2 = ['AA','lasso','cv','carriers',carriers_metrics['r2'],carriers_metrics['mse']]
    list3 = ['AA','lasso','cv',best_alpha]

    ## output results
    df_report = pd.DataFrame([list1,list2],columns= ['Trait','Model','Variant','Group','R2','MSE'])
    df_para = pd.DataFrame([list3],columns= ['Trait','Model','Variant','Best'])
    df_report.to_csv('./report/lasso_aa_cv.csv')
    df_para.to_csv('./para/lasso_aal_cv.csv')

    



