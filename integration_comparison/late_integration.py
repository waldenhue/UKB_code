from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression  
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.pipeline import Pipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor

def read_info(pheno_symbol,rv_type):
    if pheno_symbol == 'st':
        pheno_name = 'standing_height'
    elif pheno_symbol == 'hdl':
        pheno_name = 'hdl_cholesterol'
    elif pheno_symbol == 'tg':
        pheno_name = 'triglycerides'
    else:
        pheno_name = pheno_symbol
    file_path = '/data/med-hudh/taiyiv2/'+pheno_symbol+'/'
    df_pheno = pd.read_parquet(file_path+'pheno.parquet',engine='pyarrow')
    df_cv = pd.read_parquet(file_path+'cv.parquet',engine='pyarrow')
    df_rv = pd.read_parquet(file_path+'rv.parquet',engine='pyarrow')
    cv_index = [i.split('_')[0] for i in df_cv.index.tolist()]
    df_cv.index = cv_index
    df_cv['cvPRS'] = df_cv.sum(axis=1)
    df_cv = pd.DataFrame(df_cv['cvPRS'])
    df_rv['rvPRS'] = df_rv.sum(axis=1)
    df_rv = pd.DataFrame(df_rv['rvPRS'])
    
    with open(file_path+'test_id.txt','r') as file:
        test_ids = [line.strip() for line in file]
    with open(file_path+'tran1_id.txt','r') as file:
        train_ids = [line.strip() for line in file]
    train_ids = list(set(df_pheno.index.tolist())&set(df_cv.index.tolist())&set(df_rv.index.tolist())&set(train_ids))
    test_ids = list(set(df_pheno.index.tolist())&set(df_cv.index.tolist())&set(df_rv.index.tolist())&set(test_ids))
    

    
    df_pheno_train = df_pheno[df_pheno.index.isin(train_ids)]
    df_cv_train = df_cv[df_cv.index.isin(train_ids)]
    df_rv_train = df_rv[df_rv.index.isin(train_ids)]
    
    df_pheno_test = df_pheno[df_pheno.index.isin(test_ids)]
    df_cv_test = df_cv[df_cv.index.isin(test_ids)]
    df_rv_test = df_rv[df_rv.index.isin(test_ids)]

    df_rv_train = df_rv_train
    df_rv_test = df_rv_test
    if rv_type == 'rv':
        df_train_final  = pd.concat([df_pheno_train[pheno_name],df_cv_train,df_rv_train],axis=1)
        df_test_final  = pd.concat([df_pheno_test[pheno_name],df_cv_test,df_rv_test],axis=1)
    else:
        df_train_final  = pd.concat([df_pheno_train[pheno_name],df_cv_train],axis=1)
        df_test_final  = pd.concat([df_pheno_test[pheno_name],df_cv_test,],axis=1)   
    feature_names = df_train_final.drop(pheno_name, axis=1).columns.tolist()    
    X_train = df_train_final.drop(pheno_name,axis=1).values
    y_train = df_train_final[pheno_name].values
    X_test = df_test_final.drop(pheno_name,axis=1).values
    y_test = df_test_final[pheno_name].values

    
    return [X_train,X_test], [y_train,y_test],feature_names

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
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
def result_output(pheno_symbol,rv_type):
    X_list, y_list, feature_names = read_info(pheno_symbol, rv_type)
    X_train = X_list[0]
    X_test = X_list[1]
    X_test_m1 = X_list[2]
    y_train = y_list[0]
    y_test = y_list[1]
    y_test_m1 = y_list[2]
    
    
    # Pipeline of OLS
    linear_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("linear", LinearRegression())
    ])

    # Learning curve
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    plot_learning_curve(linear_pipeline, "Linear Regression Learning Curve", 
                       X_train, y_train, cv=cv, ylim=(0.0, 1.01))
    plt.savefig('/data/med-hudh/format_output/learning/linear_learning_curve_'+pheno_symbol+'_'+rv_type+'.png') 
    plt.close()

    # Model training
    linear_pipeline.fit(X_train, y_train)
    
    # Model estimation
    y_pred = linear_pipeline.predict(X_test)
    test_score = r2_score(y_test, y_pred)
    mse_all = mean_squared_error(y_test,y_pred)
    y_pred_carriers = linear_pipeline.predict(X_test_m1)
    test_score_carriers = r2_score(y_test_m1, y_pred_carriers)
    mse_carriers = mean_squared_error(y_test_m1,y_pred_carriers)
    if pheno_symbol == 'aa':
        output_symbol = 'AA'
    elif pheno_symbol == 'igf1':
        output_symbol = 'IGF-1'
    elif pheno_symbol == 'tg':
        output_symbol = 'TG'
    elif pheno_symbol == 'st':
        output_symbol = 'ST'
    else:
        output_symbol = 'HDL'
    list1 = [output_symbol,'linear','cv+rv','all',test_score,mse_carriers]
    list2 = [output_symbol,'linear','cv+rv','carriers',test_score_carriers,mse_all]
    df_report = pd.DataFrame([list1,list2],columns=['Trait','Model','Variant','Group','R2','MSE'])
    df_report.to_csv('/data/med-hudh/format_output/report/linear_'+pheno_symbol+'_'+rv_type+'.csv')

for trait in ['st','igf1','aa','tg','hdl']:
    for variant in ['cv','rv']:
        result_output(trait,variant)


