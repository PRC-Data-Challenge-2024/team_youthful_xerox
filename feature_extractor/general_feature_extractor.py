import polars as pl
import glob
import numpy as np
from loguru import logger

def process_flight_raw(day_acquisition):
    """Process flight data from a given parquet file."""
    all_traffic = []
    day_flights_df = pl.read_parquet(day_acquisition)

    # Ensure timestamp is in datetime format
    #day_flights_df = day_flights_df.with_columns(pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"))

    unique_flight_ids = day_flights_df.select("flight_id").unique().to_series().to_list()
    logger.info(f"Extracting features for {day_acquisition}")

    for flight_id_value in unique_flight_ids:
        synth_df = create_synth_unprocessed_df(flight_id_value)
        sample_traj = day_flights_df.filter(pl.col("flight_id") == flight_id_value).sort("timestamp")

        # Calculate features
        feature_values = {
            #"flight_duration": (sample_traj["timestamp"].max() - sample_traj["timestamp"].min()).total_seconds(),
            #"mean_latitude": sample_traj["latitude"].mean(),
            #"mean_longitude": sample_traj["longitude"].mean(),
            "mean_altitude": sample_traj["altitude"].mean(),
            "std_altitude": sample_traj["altitude"].std(),
            "max_altitude": sample_traj["altitude"].max(),
            "min_altitude": sample_traj["altitude"].min(),
            "avg_ground_speed": sample_traj["groundspeed"].mean(),
            "max_ground_speed": sample_traj["groundspeed"].max(),
            "std_ground_speed": sample_traj["groundspeed"].std(),
            "avg_vertical_rate": sample_traj["vertical_rate"].mean(),
            "std_vertical_rate": sample_traj["vertical_rate"].std(),
            "avg_temperature": sample_traj["temperature"].mean(),
            "avg_specific_humidity": sample_traj["specific_humidity"].mean(),
            "avg_u_wind": sample_traj["u_component_of_wind"].mean(),
            "avg_v_wind": sample_traj["v_component_of_wind"].mean(),
            "std_u_wind": sample_traj["u_component_of_wind"].std(),
            "std_v_wind": sample_traj["v_component_of_wind"].std(),
        }

        # Calculate derived wind speed
        feature_values["avg_wind_speed"] = np.sqrt(feature_values["avg_u_wind"]**2 + feature_values["avg_v_wind"]**2)

        # Calculate start and end times in hours as float
        start_time = sample_traj["timestamp"].min()
        end_time = sample_traj["timestamp"].max()

        # Extract hours and minutes
        start_hour = start_time.hour + start_time.minute / 60.0
        end_hour = end_time.hour + end_time.minute / 60.0

        # Store as float in feature_values
        feature_values['start_hour'] = start_hour
        feature_values['end_hour'] = end_hour

        # Fill the DataFrame with feature values
        synth_df = fill_synth_df(synth_df, feature_values)
        all_traffic.append(synth_df)

    # Concatenate all filled DataFrames into one
    final_traffic_features = pl.concat(all_traffic)

    # Save the final DataFrame to a Parquet file
    logger.info(f"Saving features for {day_acquisition}")
    final_traffic_features.write_parquet(f"./trafic_features_v2/trafic_features_{day_acquisition.split('.parquet')[0].split('/')[-1]}.parquet")

def create_synth_unprocessed_df(flight_id_value):
    """Create a DataFrame with NaN values for specified columns."""
    columns = {
        #"flight_duration": pl.Float64,
        #"mean_latitude": pl.Float64,
        #"mean_longitude": pl.Float64,
        "mean_altitude": pl.Float64,
        "std_altitude": pl.Float64,
        "max_altitude": pl.Float64,
        "min_altitude": pl.Float64,
        "avg_ground_speed": pl.Float64,
        "max_ground_speed": pl.Float64,
        "std_ground_speed": pl.Float64,
        "avg_vertical_rate": pl.Float64,
        "std_vertical_rate": pl.Float64,
        "avg_temperature": pl.Float64,
        "avg_specific_humidity": pl.Float64,
        "avg_u_wind": pl.Float64,
        "avg_v_wind": pl.Float64,
        "std_u_wind": pl.Float64,
        "std_v_wind": pl.Float64,
        "avg_wind_speed": pl.Float64,
        "start_hour": pl.Float64,
        "end_hour": pl.Float64,
    }

    nan_df = pl.DataFrame({
            "flight_id": pl.Series([flight_id_value],dtype=pl.Int64),  # flight_id column as str (Utf8 in Polars)
            **{
                col: pl.Series([None], dtype=dtype) if dtype in [pl.Duration("ns"), pl.UInt32] else pl.Series([float('nan')], dtype=dtype)
                for col, dtype in columns.items()
            }
        })


    return nan_df

def fill_synth_df(nan_df, feature_values):
    """Fill the DataFrame with feature values."""
    for col, value in feature_values.items():
        nan_df = nan_df.with_columns(pl.lit(value).alias(col))
    return nan_df

if __name__ == "__main__":
    parquet_folder = "./data/*.parquet"
    parquet_list = glob.glob(parquet_folder)
    for day_acquisition in parquet_list:
        process_flight_raw(day_acquisition)
