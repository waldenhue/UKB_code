from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

def read_info(pheno_symbol, pheno_name, rv_type):
    file_path = '/data/med-hudh/taiyiv2/'+pheno_symbol+'/'
    df_pheno = pd.read_parquet(file_path+'pheno.parquet',engine='pyarrow')
    df_cv = pd.read_parquet(file_path+'cv.parquet',engine='pyarrow')
    df_rv = pd.read_parquet(file_path+'rv.parquet',engine='pyarrow')
    cv_index = [i.split('_')[0] for i in df_cv.index.tolist()]
    df_cv.index = cv_index
    
    with open(file_path+'test_id.txt','r') as file:
        test_ids = [line.strip() for line in file]
    with open(file_path+'train1_id.txt','r') as file:
        train_ids = [line.strip() for line in file]
    train_ids = list(set(df_pheno.index.tolist())&set(df_cv.index.tolist())&set(df_rv.index.tolist())&set(train_ids))
    test_ids = list(set(df_pheno.index.tolist())&set(df_cv.index.tolist())&set(df_rv.index.tolist())&set(test_ids))
    
    with open(file_path+'rv_carriers.txt','r') as file:
        rv_carriers_ids = [line.strip() for line in file]

    rv_carriers_test = list(set(rv_carriers_ids)&set(test_ids))
    
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
    return [X_train,X_test,X_carriers_test], [y_train,y_test,y_carriers_test], feature_names

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        scoring='r2')
    
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
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    return plt


# ===================== main =====================
if __name__ == "__main__":
    X_list, y_list, feature_names = read_info(pheno_symbol='aa', pheno_name='aa', rv_type='rv')
    X_train, X_test, X_test_rv = X_list
    y_train, y_test, y_test_rv = y_list
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt', 0.8]
    }
    
    rf_pipeline = Pipeline([
        ("rf", RandomForestRegressor(
            n_jobs=-1, 
            random_state=42,
            oob_score=True  
        ))
    ])
    
    grid_search = GridSearchCV(
        estimator=rf_pipeline,
        param_grid={'rf__' + k: v for k, v in param_grid.items()}, 
        cv=5,
        scoring='r2',
        n_jobs=2 
    )
    
    # learning curve
    cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)  
    plot_learning_curve(rf_pipeline, "Random Forest Learning Curve", 
                       X_train, y_train, cv=cv, ylim=(0.0, 1.01))
    plt.savefig('/data/med-hudh/learning/rf_learning_curve_aa_rv.png')
    plt.close()

    # model training
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    rf_model = best_model.named_steps['rf']
      
    # model estimation
    def evaluate(model, X, y):
        pred = model.predict(X)
        return {
            'r2': r2_score(y, pred),
            'mse': mean_squared_error(y, pred)
        }
    
    all_metrics = evaluate(best_model, X_test, y_test)
    carriers_metrics = evaluate(best_model, X_test_rv, y_test_rv)

    list1 = ['AA','rf','cv+rv','all',all_metrics['r2'],all_metrics['mse']]
    list2 = ['AA','rf','cv+rv','carriers',carriers_metrics['r2'],carriers_metrics['mse']]
    list3  = ['AA','rf','cv+rv',grid_search.best_params_]

    # output results
    df_report = pd.DataFrame([list1,list2],columns=['Trait','Model','Variant','Group','R2','MSE'])
    df_para = pd.DataFrame([list3],columns= ['Trait','Model','Variant','Best'])
    df_report.to_csv('/data/med-hudh/report/rf_aa_rv.csv')
    df_para.to_csv('/data/med-hudh/para/rf_aa_rv.csv')
    



