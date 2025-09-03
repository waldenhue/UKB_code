import xgboost as xgb
from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import learning_curve
from scipy.stats import uniform, randint

def read_info(pheno_symbol, pheno_name):
    file_path = '/data/med-hudh/taiyiv2/'+pheno_symbol+'/'
    df_pheno = pd.read_parquet(file_path+'pheno.parquet',engine='pyarrow')
    df_cv = pd.read_parquet(file_path+'cv.parquet',engine='pyarrow')
    df_rv = pd.read_parquet(file_path+'rv.parquet',engine='pyarrow')
    cv_index = [i.split('_')[0] for i in df_cv.index.tolist()]
    df_cv.index = cv_index
    
    with open(file_path+'train1_id.txt','r') as file:
        test_ids = [line.strip() for line in file]
    with open(file_path+'test_id.txt','r') as file:
        train_ids = [line.strip() for line in file]
    train_ids = list(set(df_pheno.index.tolist())&set(df_cv.index.tolist())&set(df_rv.index.tolist())&set(train_ids))
    test_ids = list(set(df_pheno.index.tolist())&set(df_cv.index.tolist())&set(df_rv.index.tolist())&set(test_ids))
    
    with open(file_path+'rv_carriers.txt','r') as file:
        rv_carriers_ids = [line.strip() for line in file]
    rv_carriers_test = list(set(rv_carriers_ids)&set(test_ids))
    
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

(X_train, X_test,X_test_rv), (y_train, y_test, y_test_rv), feature_names = read_info(
    pheno_symbol='aa', pheno_name='aa')


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

def evaluate_model(model, X, y, dataset_name):
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    print(f"{dataset_name}estimation result:")
    print(f"R²: {r2:.4f}")
    print(f"MSE: {mse:.4f}\n")
    return r2, mse

_ = evaluate_model(best_model, X_train, y_train, "training set")
test_r2, test_mse = evaluate_model(best_model, X_test, y_test, "testing set")
carriers_r22, carriers_mse2 = evaluate_model(best_model, X_test_rv, y_test_rv, "carrier set")

def plot_learning_curve(estimator, title, X, y, cv=None, ylim=None):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("r² score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring='r2',
        train_sizes=np.linspace(.1, 1.0, 5), n_jobs=-1)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Score in training set")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Score in validation set")
    plt.legend(loc="best")
    plt.savefig('/data/med-hudh/learning/xgb_aa_all_learning_curve.png')
    plt.close()

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
plot_learning_curve(best_model, "XGBoost Learning Curve", X_train, y_train, cv=cv)


list1 = ['AA','xgb','cv','all',test_r2,test_mse]
list2 = ['AA','xgb','cv','carriers',carriers_r22,carriers_mse2]
df_report = pd.DataFrame([list1,list2],columns=['Trait','Model','Variant','Group','R2','MSE'])
list3  = ['AA','xgb','cv',grid_search.best_params_]
df_para = pd.DataFrame([list3],columns= ['Trait','Model','Variant','Best'])
df_report.to_csv('/data/med-hudh/report/xgb_aa_all.csv')
df_para.to_csv('/data/med-hudh/para/xgb_aa_all.csv')





