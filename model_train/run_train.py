from sklearn.model_selection import cross_val_score,KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Lasso,Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import model, settle_data
from pathlib import Path
import pandas as pd
import numpy as np
import textwrap
import argparse
import json
import os


def parse_args():
    """The data dir should include [pheno/cv/rv].parquet and [train/test/carriers].txt ."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate regression models.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=textwrap.dedent("""
        Example usage:
        python run_train.py \\
        --param_id ${LSB_JOBINDEX} \\
        --pheno_symbol st \\
        --variant cv \\ 
        --strategy early \\
        --model rf \\
        --output_dir ./results \\
        --data_dir ./data
        """)
    )

    parser.add_argument('--pheno_symbol', required=True)
    parser.add_argument('--model', required=True, choices=['lasso','ridge','ols','rf','xgb'])
    parser.add_argument('--variant', choices=['cv','rv','allin'], default='cv')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--strategy', required=True, choices=['early','late'])
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--param_id', type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    # -------------------------
    # Load phenotype + genotype data
    # -------------------------
    if args.variant == 'allin':
        df_pheno, df_cv, df_rv = settle_data.read_info(
            args.pheno_symbol, 'allin', base_dir=Path(args.data_dir), strategy=args.strategy
        )
        df_list = [df_pheno, df_cv, df_rv]
        variant_str = 'allin'

    elif args.variant == 'cv':
        df_pheno, df_cv = settle_data.read_info(
            args.pheno_symbol, 'cv', base_dir=Path(args.data_dir), strategy=args.strategy
        )
        df_list = [df_pheno, df_cv]
        variant_str = 'cv'

    else:
        df_pheno, df_rv = settle_data.read_info(
            args.pheno_symbol, 'rv', base_dir=Path(args.data_dir), strategy=args.strategy
        )
        df_list = [df_pheno, df_rv]
        variant_str = 'rv'

    # -------------------------
    # Load train/test/carrier IDs
    # -------------------------
    train_id = np.loadtxt(Path(args.data_dir)/'train.txt', dtype=str).tolist()
    test_id = np.loadtxt(Path(args.data_dir)/'test.txt', dtype=str).tolist()
    carriers_id = np.loadtxt(Path(args.data_dir)/'carriers.txt', dtype=str).tolist()

    # -------------------------
    # Train/test split (read-only)
    # -------------------------
    (X_train, y_train), (X_test, y_test), (X_carriers, y_carriers),(X_non_carriers,y_non_carriers) = settle_data.split_train_test(
        df_list, train_id, test_id, carriers_id,
        output_dir=Path(args.output_dir),  # no writing by split function
        phe=args.pheno_symbol
    )

    # ======================================================
    #           job array mode for RF/XGB/Lasso/Ridge
    # ======================================================
    if args.model in ['rf','xgb','lasso','ridge']:

        param_list = model.generate_param_list(50,42,args.model)

        if args.param_id is None:
            raise ValueError("RF requires --param_id when running job array.")

        params = param_list[args.param_id - 1]

        cv = KFold(n_splits = 5,shuffle=True,random_state=42)

        if args.model == 'rf':
            train_model = RandomForestRegressor(**params,n_jobs=-1)
        elif args.model =='xgb':
            train_model = XGBRegressor(**params,n_jobs=-1)
        elif args.model =='lasso':
            train_model = Lasso(**params)
        elif args.model == 'ridge':
            train_model = Ridge(**params)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_carriers = scaler.transform(X_carriers)
        X_non_carriers = scaler.transform(X_non_carriers)
        mse_scores = -cross_val_score(train_model,X_train,y_train,scoring='neg_mean_squared_error',cv=cv,n_jobs=-1)
        mean_mse = mse_scores.mean()


        train_model.fit(X_train, y_train)
        y_pred = train_model.predict(X_test)
        y_car_pred = train_model.predict(X_carriers)
        # Evaluate
        r2,mse,bps = model.evaluate_model(train_model, X_test, y_test,X_train)
        r2_car, mse_car, bps_car = model.evaluate_model(train_model, X_carriers, y_carriers,X_train)
        r2_no_car, mse_no_car, bps_no_car = model.evaluate_model(train_model, X_non_carriers, y_non_carriers,X_train)
        # Save result
        result = {
            "param_id": args.param_id,
            "params": params,
            "mse": mse,
            "r2": r2,
            "beta_per_sd_pred": bps,
            "mse_carriers":mse_car,
            "r2_carriers":r2_car,
            "beta_per_sd_pred_carriers":bps_car,
            "non_carriers_mse":mse_no_car,
            "non_carriers_r2":r2_no_car,
            "non_carriers_beta_per_sd_pred":bps_no_car,
            "mse_cv":mean_mse,
            "mse_cv_std":mse_scores.std(),
            "strategy":args.strategy,
            "model":args.model,
            "variant":variant_str,
            "pheno":args.pheno_symbol  
        }
        
        output_path = os.path.join(args.output_dir,f"{args.pheno_symbol}_{variant_str}_{args.strategy}_{args.model}_json")
        os.makedirs(output_path, exist_ok=True)
        out_file = f"result_{args.param_id}.json"
        with open(os.path.join(output_path,out_file), "w") as f:
            json.dump(result, f, indent=4)

        return  # Important! Do not fall through to below.

    # ======================================================
    #     single job mode for OLS
    # ======================================================
    pipeline = model.building_pipeline(args.model)
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    r2,mse,bps = model.evaluate_model(pipeline, X_test, y_test,X_train)
    r2_car, mse_car, bps_car = model.evaluate_model(pipeline, X_carriers, y_carriers,X_train)
    r2_no_car, mse_no_car, bps_no_car = model.evaluate_model(pipeline, X_non_carriers, y_non_carriers,X_train)
    df_results = pd.DataFrame({
        'Overall_R2': [r2],
        'Overall_MSE': [mse],
        'Overall_Beta_per_SD': [bps],
        'Carriers_R2': [r2_car],
        'Carriers_MSE': [mse_car],
        'Carriers_Beta_per_SD': [bps_car],
        'Non_Carriers_R2': [r2_no_car],
        'Non_Carriers_MSE': [mse_no_car],
        'Non_Carriers_Beta_per_SD': [bps_no_car],
        'variant': [variant_str],
        'model': [args.model],
        'strategy': [args.strategy],
        'pheno': [args.pheno_symbol]
    })

    os.makedirs(args.output_dir, exist_ok=True)

    df_results.to_csv(
        os.path.join(args.output_dir,
                     f"{args.pheno_symbol}_{variant_str}_{args.strategy}_{args.model}.csv"),
        index=False
    )



if __name__ == "__main__":
    main()
