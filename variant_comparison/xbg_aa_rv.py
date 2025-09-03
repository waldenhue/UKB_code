import xgboost as xgb
from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import learning_curve
from scipy.stats import uniform, randint

def read_info(pheno_symbol, pheno_name,rv_type):
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
    '''
    with open(file_path+'m1_carriers.txt','r') as file:
        m1_carriers_ids = [line.strip() for line in file]
    '''
    with open(file_path+'rv_carriers.txt','r') as file:
        rv_carriers_ids = [line.strip() for line in file]
    ##m1_carriers_train = list(set(m1_carriers_ids)&set(train_ids))
    rv_carriers_train = list(set(rv_carriers_ids)&set(train_ids))
    ##m1_carriers_test = list(set(m1_carriers_ids)&set(test_ids))
    rv_carriers_test = list(set(rv_carriers_ids)&set(test_ids))
    
    df_pheno_train = df_pheno[df_pheno.index.isin(train_ids)]
    df_cv_train = df_cv[df_cv.index.isin(train_ids)]
    ##df_m1_train = df_m1[df_m1.index.isin(train_ids)]
    df_rv_train = df_rv[df_rv.index.isin(train_ids)]
    
    df_pheno_test = df_pheno[df_pheno.index.isin(test_ids)]
    df_cv_test = df_cv[df_cv.index.isin(test_ids)]
    ##df_m1_test = df_m1[df_m1.index.isin(test_ids)]
    df_rv_test = df_rv[df_rv.index.isin(test_ids)]


    df_rv_train = df_rv_train
    df_rv_test = df_rv_test
    rv_carriers_test = rv_carriers_test
    df_train_final = pd.concat([df_pheno_train[pheno_name], df_cv_train, df_rv_train], axis=1)
    df_test_final = pd.concat([df_pheno_test[pheno_name], df_cv_test, df_rv_test], axis=1)
    df_test_carriers_final = df_test_final[df_test_final.index.isin(rv_carriers_test)]
    
    # 保留特征名称
    feature_names = df_train_final.drop(pheno_name, axis=1).columns.tolist()
    
    X_train = df_train_final.drop(pheno_name, axis=1)  # 保持DataFrame格式
    y_train = df_train_final[pheno_name].values
    X_test = df_test_final.drop(pheno_name, axis=1)
    y_test = df_test_final[pheno_name].values
    X_test_m1 = df_test_carriers_final.drop(pheno_name, axis=1)
    y_test_m1 = df_test_carriers_final[pheno_name].values
    
    return (X_train, X_test, X_test_m1), (y_train, y_test, y_test_m1), feature_names

# 调用函数时获取特征名称
(X_train, X_test, X_test_m1), (y_train, y_test, y_test_m1), feature_names = read_info(
    pheno_symbol='aa', pheno_name='aa', rv_type='rv'
)


xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    tree_method='hist',  # 无GPU时使用'hist'，有GPU可改为'gpu_hist'
    n_jobs=-1,
    random_state=42
)

# 2. 定义参数搜索空间
param_dist = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(3, 8),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'gamma': uniform(0, 0.5),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 1)
}

# 3. 使用随机搜索进行初步调参
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=50,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

# 输出最佳参数
print("Best parameters found: ", random_search.best_params_)
best_model = random_search.best_estimator_

# 4. 精细化网格搜索（基于随机搜索的结果调整参数范围）
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

# 5. 模型评估
def evaluate_model(model, X, y, dataset_name):
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    print(f"{dataset_name}评估结果:")
    print(f"R²: {r2:.4f}")
    print(f"MSE: {mse:.4f}\n")
    return r2, mse

# 在三个数据集上评估
_ = evaluate_model(best_model, X_train, y_train, "训练集")
test_r2, test_mse = evaluate_model(best_model, X_test, y_test, "测试集")
carriers_r2, carriers_mse = evaluate_model(best_model, X_test_m1, y_test_m1, "特殊群体测试集")

# 6. 绘制学习曲线
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
    plt.savefig('/data/med-hudh/learning/xgb_aa_rv_learning_curve.png')
    plt.close()

# 使用ShuffleSplit生成交叉验证策略
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
plot_learning_curve(best_model, "XGBoost Learning Curve", X_train, y_train, cv=cv)

# 8. 保存结果和模型
results = f"""最佳参数: {grid_search.best_params_}
测试集评估:
- R²: {test_r2:.4f}
- MSE: {test_mse:.4f}
特殊群体评估:
- R²: {carriers_r2:.4f}
- MSE: {carriers_mse:.4f}"""

list1 = ['AA','xgb','cv+rv','all',test_r2,test_mse]
list2 = ['AA','xgb','cv+rv','carriers',carriers_r2,carriers_mse]
df_report = pd.DataFrame([list1,list2],columns=['Trait','Model','Variant','Group','R2','MSE'])
list3  = ['AA','xgb','cv+rv',grid_search.best_params_]
df_para = pd.DataFrame([list3],columns= ['Trait','Model','Variant','Best'])
df_report.to_csv('/data/med-hudh/report/xgb_aa_rv.csv')
df_para.to_csv('/data/med-hudh/para/xgb_aa_rv.csv')




