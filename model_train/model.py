from sklearn.model_selection import KFold, GridSearchCV,learning_curve
from sklearn.linear_model import LassoCV, Ridge, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint, uniform
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np


def generate_param_list(n, seed=42,model = 'xgb'):
    np.random.seed(seed)  
    param_list = []
    for i in range(n):
        if model == 'xgb':
            params = {
                "n_estimators": np.random.randint(100, 1001),
                "max_depth": np.random.randint(3, 16),
                "learning_rate": 10**np.random.uniform(-2, 0),
                "subsample": np.random.uniform(0.6, 1.0),
                "colsample_bytree": np.random.uniform(0.6, 1.0),
                "min_child_weight": np.random.uniform(1, 10),
                "gamma": np.random.uniform(0, 5),
                "reg_alpha": np.random.uniform(0, 1),
                "reg_lambda": np.random.uniform(1, 10),
                "tree_method": "hist",
                "random_state": 42
            }
        elif model == 'rf':
            params = {
                "n_estimators": np.random.randint(100, 1001),
                "max_depth": np.random.randint(3, 16),
                'min_samples_split': np.random.randint(2,11),
                'min_samples_leaf': np.random.randint(1,20),
                'max_features': np.random.uniform(0.3,1),
                'bootstrap': bool(np.random.choice([True, False]))
            }
        elif model =='lasso':
            params = {
                "alpha": 10**np.random.uniform(-5,1)}
        elif model == 'ridge':
            params = {
                "alpha":10**np.random.uniform(-4,4)
            }

        param_list.append(params)

    return param_list


def building_pipeline(model):
    """
    this function builds the pipeline / CV model
    Args:
        model: str, model name, options are 'ols'
    Returns:
        pipeline: an estimator with fit/predict (Pipeline )
    """

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    if model == "ols":
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ols', LinearRegression())
        ])

    else:
        raise ValueError(f"Unknown model: {model}")

    return pipeline


def training_model(pipeline, X_train, y_train):
    "this function trains the given model with data"
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(pipeline, X_test, y_test,X_train):
    "this function evaluates the trained model with test data"
    y_pred_train = pipeline.predict(X_train)
    mu = np.mean(y_pred_train)
    sd = np.std(y_pred_train)    
    y_pred_test = pipeline.predict(X_test)
    y_pred_test_z = (y_pred_test - mu) / sd
    r2 = r2_score(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    model = sm.OLS(y_test, sm.add_constant(y_pred_test_z)).fit()
    beta_per_sd_pred = model.params[1]
    return r2, mse,beta_per_sd_pred
