import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import optuna
import glob
import json
import os

# Load configuration
with open('configs/credentials.json', 'r') as config_file:
    config = json.load(config_file)

# Construct the filename from config variables
filename = f"{config['team_name']}_v3_{config['code']}.csv"
file_path = os.path.join(config['output_folder'], filename)

# # Define a mapping for aircraft categories

aircraft_subgroups = {
    "A333": ["A333"],
    "B77W": ["B77W"],
    "A332": ["A332"],
    "A359": ["A359"],
    "B789": ["B789"],
    "B788": ["B788"],
    "B772": ["B772"],
    "B763": ["B763"],
    "A321": ["A321"],
    "A21N": ["A21N"],
    "A20N": ["A20N"],
    "BCS3": ["BCS3"],
    "E195": ["E195"],
    "B738": ["B738"],
    "A320": ["A320"],
    "B739": ["B739"],
    "B38M": ["B38M"],
    "CRJ9": ["CRJ9"],
    "A319": ["A319"],
    "S": ["B39M","A343","C56X","B773","E290",'A310','B752'],
    "L": ["B737", "BCS1", "E190", "AT76"],
}

def explore_flight():
    train_df = pd.read_csv("csv_set/challenge_mean_tow_diff.csv")
    test_df = pd.read_csv("csv_set/final_submission_set.csv")
    df = pd.concat([train_df, test_df])
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["dayofweek"] = pd.to_datetime(df["date"]).dt.dayofweek
    return df

def get_data(use_traffic):
    flight_df = explore_flight()
    flight_df["flight_id"] = flight_df["flight_id"].astype(int)

    if use_traffic:
        features = pd.read_parquet("./features/trafic_features_v8/all_traffic_features_climb.parquet")
        features["flight_id"] = features["flight_id"].astype(int)
        #features['takeoff_duration'] = features['takeoff_duration'].dt.total_seconds()
        flight_with_journey_df = pd.merge(flight_df, features, on="flight_id", how="left", indicator=True)
    else:
        flight_with_journey_df = flight_df

    assert flight_with_journey_df["flight_id"].is_unique

    columns_to_remove = ["callsign", "name_adep", "name_ades", "icao24", "timestamp", "actual_offblock_time", "arrival_time", "_merge"]
    category_cols = ["adep", "ades", "aircraft_type", "wtc", "airline", "month", "dayofweek","country_code_adep", "country_code_ades"]
    
    flight_with_journey_df = flight_with_journey_df.drop(columns=columns_to_remove, errors="ignore")
    
    for cat in category_cols:
        flight_with_journey_df[cat] = flight_with_journey_df[cat].astype("category")

    flight_with_journey_df["date"] = pd.to_datetime(flight_with_journey_df["date"]).dt.dayofyear
    flight_with_journey_df["day_sin"] = np.sin(2 * np.pi * flight_with_journey_df["date"] / 365.0)
    flight_with_journey_df["day_cos"] = np.cos(2 * np.pi * flight_with_journey_df["date"] / 365.0)

    train = flight_with_journey_df.dropna(subset="tow")
    submit_set = flight_with_journey_df[flight_with_journey_df["tow"].isna()]

    assert len(train) + len(submit_set) == len(flight_with_journey_df)

    X_train = train.drop(columns=["tow", "flight_id", "mean_tow_per_aircraft_type", "mean_tow_difference"])
    y_train = train["mean_tow_difference"]
    y_mean_train = train[["aircraft_type", "tow", "mean_tow_per_aircraft_type", "mean_tow_difference"]]
    X_submission = submit_set.drop(columns="tow")
    return X_train, y_train, y_mean_train, X_submission

def retreive_means():
    df = pd.read_csv('./csv_set/mean_tow_per_aircraft.csv')
    return df

def xgboost_subgroup_models(use_traffic=True):
    X, y, y_mean_train, X_submission = get_data(use_traffic)
    
    results = {}
    
    for subgroup, aircraft_types in aircraft_subgroups.items():
        # Filter data for the current subgroup
        subgroup_mask = X['aircraft_type'].isin(aircraft_types)
        
        if subgroup_mask.sum() > 0:
            X_subgroup = X[subgroup_mask]
            y_subgroup = y[subgroup_mask]
            y_mean_train_subgroup = y_mean_train[subgroup_mask]

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X_subgroup, y_subgroup, test_size=0.3, random_state=42)

            # Define the Optuna objective function
            def objective(trial):
                param = {
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 750),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 1),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'alpha': trial.suggest_float('alpha', 0, 10),
                    'lambda': trial.suggest_float('lambda', 0, 10),
                }
                model = XGBRegressor(**param, enable_categorical=True)
                model.fit(X_train, y_train)
                return root_mean_squared_error(y_test, model.predict(X_test))

            # Optimize hyperparameters using Optuna
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=20)

            # Train the best model with the best parameters
            best_params = study.best_params
            best_model = XGBRegressor(**best_params, enable_categorical=True)
            best_model.fit(X_subgroup, y_subgroup)

            # Store results
            results[subgroup] = best_model

            # Evaluate the model
            mean_gt_tow_diff_pred = best_model.predict(X_subgroup)
            y_pred_tow= y_mean_train_subgroup['mean_tow_per_aircraft_type'] - mean_gt_tow_diff_pred
            rmse = root_mean_squared_error(y_mean_train_subgroup["tow"], y_pred_tow)
            print(f"Final RMSE all : {rmse}")
    # Handle predictions for submission
    for subgroup, model in results.items():
        subgroup_aircraft_types = aircraft_subgroups[subgroup]
        submission_mask = X_submission['aircraft_type'].isin(subgroup_aircraft_types)
        if submission_mask.sum() > 0:
            X_sub_group = X_submission[submission_mask].copy()
            X_sub_group["tow_diff_pred"] = model.predict(X_sub_group.drop(columns=["flight_id", "mean_tow_per_aircraft_type", "mean_tow_difference"]))
            df_mean_per_aircraft = retreive_means()
            test_df = X_sub_group.merge(df_mean_per_aircraft, on='aircraft_type', how='left', suffixes=('', '_mean'))
            test_df['tow'] = test_df['tow_mean'] - test_df['tow_diff_pred']
            test_df[["flight_id", "tow"]].astype(int).to_csv(f"./submission_files/team_youthful_xerox_{subgroup}_vX.csv", index=False)

    return results

def fuse_cat_result(version):
    files = glob.glob(f'../submission_files/*_v{version}.csv')
    main_df = pd.read_csv('../data/final_submission_set.csv')

    dataframes = [pd.read_csv(file) for file in files]
    for i, df in enumerate(dataframes):
        suffix = f'_df{i+1}'
        # Merge the current dataframe on 'flight_id'
        main_df = pd.merge(main_df, df[['flight_id', 'tow']], on='flight_id', how='left', suffixes=('', suffix))
        # Fill missing values in the 'tow' column
        main_df['tow'] = main_df['tow'].fillna(main_df[f'tow{suffix}'])
        # Drop the extra 'tow_dfX' column to avoid clutter
        main_df.drop(columns=[f'tow{suffix}'], inplace=True)
    main_df[['flight_id','tow']].to_csv(file_path, index=False)

if __name__ == '__main__':
    version = 000
    xgboost_subgroup_models(use_traffic=True)
    fuse_cat_result(version)
