import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import TargetEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import optuna
import os
import json

# Load configuration
with open('configs/credentials.json', 'r') as config_file:
    config = json.load(config_file)

# Construct the filename from config variables
filename = f"{config['team_name']}_vXX_{config['code']}.csv"
file_path = os.path.join(config['output_folder'], filename)

def explore_flight():
    train_df = pd.read_csv("csv_set/challenge_set.csv")
    test_df = pd.read_csv("csv_set/submission_set.csv")

    df = pd.concat([train_df, test_df])

    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["dayofweek"] = pd.to_datetime(df["date"]).dt.dayofweek
    #df[["year", "week", "day"]] = pd.to_datetime(df["date"]).dt.isocalendar()
    return df


def get_data(use_traffic):
    flight_df = explore_flight()
    flight_df["flight_id"] = flight_df["flight_id"].astype(int)
    
    if use_traffic:
        features = pd.read_parquet("./features/trafic_features_v6")#.dropna()
        features["flight_id"] = features["flight_id"].astype(int)
        features['takeoff_duration'] = features['takeoff_duration'].dt.total_seconds()
        flight_with_journey_df = pd.merge(flight_df, features, on="flight_id", how="left", indicator=True)
    else:
        flight_with_journey_df = flight_df

    assert flight_with_journey_df["flight_id"].is_unique

    columns_to_remove = ["callsign", "name_adep", "name_ades",
                         "icao24", "timestamp", "actual_offblock_time", "arrival_time","country_code_adep","country_code_ades","_merge",
                         ]

    category_cols =["adep","ades","aircraft_type", "wtc", "airline"] # ["adep","ades",
    flight_with_journey_df = flight_with_journey_df.drop(columns=columns_to_remove, errors="ignore")
    #flight_with_journey_df = flight_with_journey_df[flight_with_journey_df.drop(columns=['tow']).notna().all(axis=1)]
    
    for cat in category_cols:
        flight_with_journey_df[cat] = flight_with_journey_df[cat].astype("category")

    flight_with_journey_df["date"] = pd.to_datetime(flight_with_journey_df["date"]).dt.dayofyear
    flight_with_journey_df["day_sin"] = np.sin(2 * np.pi * flight_with_journey_df["date"] / 365.0)
    flight_with_journey_df["day_cos"] = np.cos(2 * np.pi * flight_with_journey_df["date"] / 365.0)
    flight_with_journey_df = flight_with_journey_df.drop(columns=['date'], errors="ignore")
    
    train = flight_with_journey_df.dropna(subset="tow")
    submit_set = flight_with_journey_df[flight_with_journey_df["tow"].isna()]

    assert len(train) + len(submit_set) == len(flight_with_journey_df)

    X_train = train.drop(columns=["tow", "flight_id"])
    y_train = train["tow"]
    X_submission = submit_set.drop(columns="tow")
    return X_train, y_train, X_submission


def xgboost_experimentation(use_traffic=True):
    # Objective function for Optuna to optimize
    def objective(trial):
        # Define the hyperparameter search space
        param = {
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 1),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'alpha': trial.suggest_float('alpha', 0, 10),
            'lambda': trial.suggest_float('lambda', 0, 10),
        }
        # Create the model with the current set of hyperparameters
        model = XGBRegressor(**param, enable_categorical=True)

        # Train the model
        model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = model.predict(X_test)

        # Calculate RMSE (Root Mean Squared Error)
        rmse = root_mean_squared_error(y_test, y_pred)
        return rmse

    X, y, X_submission = get_data(use_traffic)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Create a study and optimize
    study = optuna.create_study(direction='minimize')  # Minimize RMSE
    study.optimize(objective, n_trials=30)

    # Best parameters
    best_params = study.best_params
    print("Best parameters:", best_params)
    # Train the final model with the best parameters
    best_model = XGBRegressor(**best_params, enable_categorical=True)
    best_model.fit(X, y)

    y_pred = best_model.predict(X)
    rmse = root_mean_squared_error(y, y_pred)
    print(f"Final RMSE all : {rmse}")
    # Create a DataFrame from X_test
    results_df = X.copy()  # Make a copy of X_test to preserve the original
    results_df['tow_gt'] = y
    results_df['tow_pred'] = y_pred  # Add predictions as a new column

    # Save the DataFrame to a CSV file
    #results_df.to_csv('./reporting/result_analysis/predictions_alldata_XX.csv', index=False)

    X_submission["tow"] = best_model.predict(X_submission.drop(columns="flight_id"))
    X_submission[["flight_id", "tow"]].astype(int).to_csv(file_path, index=False)

    # Plotting feature importance with a larger figure size
    plt.figure(figsize=(25, 20))  # Specify the desired figure size (width, height)
    ax = plot_importance(best_model, importance_type='weight' ,max_num_features=15)  # Adjust as needed
    ax.figure.tight_layout()
    ax.figure.savefig('./reporting/xgboost_feat_importance_vXX.png')


if __name__ == '__main__':
    xgboost_experimentation(use_traffic=True)
