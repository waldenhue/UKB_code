import xgboost as xgb
from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import learning_curve
from scipy.stats import uniform, randint

## We take apolipoprotein A as an example here

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

    return (X_train, X_test,X_test_rv), (y_train, y_test,y_test_rv), feature_names

# ===================== main =====================
if __name__ == "__main__":

    X_list, y_list, feature_names = read_info(pheno_symbol='aa', pheno_name='aa')
    X_train, X_test, X_test_rv = X_list
    y_train, y_test, y_test_rv = y_list

    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',  
        n_jobs=-1,
        random_state=42
    )

    param_grid = {
        'n_estimators': [500, 800, 1000],
        'max_depth': [4, 5, 6],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0.5, 1, 1.5]
    }

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # model estimation
    def evaluate(model, X, y):
        pred = model.predict(X)
        return {
            'r2': r2_score(y, pred),
            'mse': mean_squared_error(y, pred)
        }
    
    all_metrics = evaluate(best_model, X_test, y_test)
    carriers_metrics = evaluate(best_model, X_test_rv, y_test_rv)

    list1 = ['AA','xgb','cv','all',all_metrics['r2'],all_metrics['mse']]
    list2 = ['AA','xgb','cv','carriers',carriers_metrics['r2'],carriers_metrics['mse']]
    list3  = ['AA','xgb','cv',grid_search.best_params_]
    df_report = pd.DataFrame([list1,list2],columns=['Trait','Model','Variant','Group','R2','MSE'])
    df_para = pd.DataFrame([list3],columns= ['Trait','Model','Variant','Best'])

    ## output results
    df_report.to_csv('/data/med-hudh/report/xgb_aa_all.csv')
    df_para.to_csv('/data/med-hudh/para/xgb_aa_all.csv')





