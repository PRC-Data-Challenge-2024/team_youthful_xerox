import glob
from tqdm import tqdm
from loguru import logger
import os
import numpy as np
from datetime import timedelta
import polars as pl

# Configure loguru to log to a file (optional, for detailed logs)
logger.add(
    "processing_log.log", rotation="10 MB"
)  # Optional: logs to a file with rotation


# Function to calculate True Air Speed (TAS)
def deduce_TAS_old(GS, WS, track, U_wind, V_wind):
    # Calculate the necessary components
    wind_angle = np.deg2rad(track) + np.arctan2(V_wind, U_wind)
    cos_component = np.cos(wind_angle)

    # Coefficients for the quadratic equation
    a = 2 * WS * cos_component
    b = GS**2 - WS**2

    # Solve the quadratic equation for TAS
    TAS_squared = np.maximum((-a + np.sqrt(a**2 + 4 * b)) / 2, 0)
    TAS = np.sqrt(TAS_squared)
    return TAS


def deduce_TAS(GS, WS, track, U_wind, V_wind):
    # Calculate the wind direction and wind speed components in the direction of travel
    # https://www.researchgate.net/publication/362258965_Data-Driven_Analysis_for_Calculated_Time_Over_in_Air_Traffic_Flow_Management
    wind_direction = np.arctan2(V_wind, U_wind)  # Calculate the wind direction
    wind_angle = (
        np.deg2rad(track) - wind_direction
    )  # Relative angle between wind and track

    # Wind component in the direction of travel
    wind_component_a = WS * np.cos(wind_angle)
    wind_componen_c = WS * np.sin(wind_angle)
    TAS = np.sqrt((GS - wind_component_a) ** 2 + wind_componen_c**2)

    return TAS


# Function to calculate track deviation
def calculate_track_deviation(track_series):
    track_deviation = np.gradient(track_series)  # Rate of change of track over time
    return track_deviation


def limit_data_duration(
    df: pl.DataFrame, time_column: str = "timestamp", duration_minutes: int = 30
) -> pl.DataFrame:
    """
    Limit the data to the first specified number of minutes based on the time column.

    Parameters:
    df (pl.DataFrame): The input Polars dataframe.
    time_column (str): The name of the time-related column (default is 'timestamp').
    duration_minutes (int): The number of minutes to retain from the start (default is 30).

    Returns:
    pl.DataFrame: A new dataframe with data limited to the specified duration, maintaining the original time format.
    """

    # Ensure the timestamp is in datetime[ns, UTC] format
    if df[time_column].dtype != pl.Datetime("ns", "UTC"):
        df = df.with_columns(
            [pl.col(time_column).cast(pl.Datetime("ns", "UTC")).alias(time_column)]
        )

    # Get the start time as a Python datetime object
    start_time = df.select(pl.col(time_column).min())[0, 0]

    # Define the duration as a timedelta
    limit_duration = timedelta(minutes=duration_minutes)

    # Calculate the end time by adding the specified duration to the start time
    end_time = start_time + limit_duration

    # Filter the dataframe to keep only rows within the specified duration
    df_limited = df.filter(pl.col(time_column) <= pl.lit(end_time))

    return df_limited


def segment_signal_with_duration_check(
    dataframe,
    vertical_rate_threshold,
    discontinuity_threshold_seconds=2,
    max_duration_minutes=30,
):
    # Convert the maximum duration to seconds
    max_duration_seconds = max_duration_minutes * 60

    # Filter for the portion of the signal where the vertical rate is above the threshold
    takeoff_phase_df = dataframe.filter(
        pl.col("vertical_rate") > vertical_rate_threshold
    )

    # If no takeoff phase is detected, return an empty dataframe
    if takeoff_phase_df.is_empty():
        return pl.DataFrame()

    # Compute the time difference between consecutive rows
    takeoff_phase_segments_lab = takeoff_phase_df.with_columns(
        (pl.col("timestamp").diff().fill_null(pl.duration(seconds=0))).alias(
            "timestamp_diff"
        )
    )

    # Set the threshold for detecting discontinuities (e.g., 5 seconds)
    discontinuity_threshold = pl.duration(seconds=discontinuity_threshold_seconds)

    # Label segments based on discontinuities
    takeoff_phase_segments_lab = takeoff_phase_segments_lab.with_columns(
        (
            (pl.col("timestamp_diff") > discontinuity_threshold)
            .cum_sum()
            .alias("segment_id")
        )
    )

    # Compute the length (duration) of each segment in nanoseconds and convert to seconds
    takeoff_segments_with_duration = (
        takeoff_phase_segments_lab.group_by("segment_id")
        .agg(
            [
                pl.col("timestamp").max().alias("end_time"),
                pl.col("timestamp").min().alias("start_time"),
                pl.len().alias("num_points"),
            ]
        )
        .with_columns(
            (
                (
                    pl.col("end_time").cast(pl.Int64)
                    - pl.col("start_time").cast(pl.Int64)
                )
                / 1_000_000_000
            ).alias("segment_duration_seconds")
        )
    )

    # Filter out segments that are longer than the max duration
    valid_segments = takeoff_segments_with_duration.filter(
        pl.col("segment_duration_seconds") <= max_duration_seconds
    )

    # Get the segment ids for valid segments
    valid_segment_ids = valid_segments.select("segment_id")

    # Filter the original dataframe to only include valid segments
    filtered_segments_df = takeoff_phase_segments_lab.filter(
        pl.col("segment_id").is_in(valid_segment_ids)
    )

    # Concatenate all valid segments into one large continuous segment
    concatenated_segment = filtered_segments_df.sort("timestamp")

    if not concatenated_segment.is_empty():
        concatenated_segment = limit_data_duration(concatenated_segment)
    return concatenated_segment


# Function to process and validate individual flights with additional features
def process_flight(
    flight_df,
    flight_id,
    vertical_rate_threshold,
    min_duration_threshold_minutes,
    noise_tolerance,
    plot_output_dir=None,
    enable_plotting=True,
):

    # Sort the flight's data by timestamp
    flight_df = flight_df.sort("timestamp")
    columns_to_clip = ["altitude", "groundspeed"]
    flight_df = flight_df.with_columns(
        [
            pl.when(pl.col(col) < 0).then(0).otherwise(pl.col(col)).alias(col)
            for col in columns_to_clip
        ]
    )

    # Extract the actual takeoff phase based on vertical rate
    takeoff_segment = segment_signal_with_duration_check(
        flight_df,
        vertical_rate_threshold,
        discontinuity_threshold_seconds=5,
        max_duration_minutes=30,
    )
    # Check if the takeoff segment exists
    if takeoff_segment.is_empty():
        logger.warning(f"Skipping flight {flight_id} due to missing takeoff data.")
        return None

    # Calculate the duration of the takeoff segment in minutes
    time_diff = (
        takeoff_segment["timestamp"].max() - takeoff_segment["timestamp"].min()
    ).total_seconds() / 60

    # Skip if the duration is less than the threshold in minutes
    if time_diff < min_duration_threshold_minutes:
        logger.warning(
            f"Skipping flight {flight_id} due to insufficient takeoff duration ({time_diff:.2f} minutes)."
        )
        return None
    takeoff_segment = takeoff_segment.with_columns(
        (pl.col("u_component_of_wind") ** 2 + pl.col("v_component_of_wind") ** 2)
        .sqrt()
        .alias("wind_speed")
    )
    # Additional feature calculations
    tas = deduce_TAS(
        takeoff_segment["groundspeed"],
        takeoff_segment["wind_speed"],
        takeoff_segment["track"],
        takeoff_segment["u_component_of_wind"],
        takeoff_segment["v_component_of_wind"],
    )
    # Compute the difference in groundspeed and timestamp
    groundspeed_diff = takeoff_segment["groundspeed"].diff().fill_nan(0)

    # Compute track deviation
    track_deviation = calculate_track_deviation(takeoff_segment["track"])

    # Compute track variance (stability)
    track_variance = takeoff_segment["track"].var()

    # Combine the takeoff segment with new features
    takeoff_segment = takeoff_segment.with_columns(
        [
            pl.Series(tas).alias("tas"),  # True Air Speed
            pl.Series(groundspeed_diff).alias(
                "groundspeed_diff"
            ),  # Groundspeed difference
            pl.Series(track_deviation).alias(
                "track_deviation"
            ),  # Rate of change of track
            pl.Series([track_variance] * len(takeoff_segment)).alias(
                "track_variance"
            ),  # Track stability
        ]
    )

    return takeoff_segment


# Function to lazily load and process each Parquet file
def process_parquet_file_lazy(
    file,
    vertical_rate_threshold=500,
    min_duration_threshold_minutes=1,
    noise_tolerance=50,
    plot_output_dir=None,
    enable_plotting=True,
):
    try:
        # Use Polars LazyFrame for memory-efficient loading
        lazy_df = pl.scan_parquet(file)

        # Select relevant columns
        lazy_df = lazy_df.select(
            [
                "flight_id",
                "timestamp",
                "altitude",
                "groundspeed",
                "vertical_rate",
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
                "track",
            ]
        )

        # Collect the result (execute the lazy operations)
        processed_df = lazy_df.collect()

        # Process each flight individually
        flight_segments = []
        for flight_id in processed_df["flight_id"].unique():
            flight_df = processed_df.filter(pl.col("flight_id") == flight_id)
            takeoff_segment = process_flight(
                flight_df,
                flight_id,
                vertical_rate_threshold,
                min_duration_threshold_minutes,
                noise_tolerance,
                plot_output_dir,
                enable_plotting,
            )

            if takeoff_segment is not None:
                flight_segments.append(takeoff_segment)

        # Combine valid takeoff segments for all flights
        if flight_segments:
            return pl.concat(flight_segments)
        else:
            return None

    except Exception as e:
        logger.error(f"Error processing {file}: {e}")
        return None


# Function to extract takeoff features for each flight, based on segmented signal
def extract_takeoff_features_lazy(
    directory_path,
    output_path,
    noise_tolerance=50,
    vertical_rate_threshold=500,
    min_duration_threshold_minutes=1,
    plot_output_dir=None,
    enable_plotting=True,
):
    # Load all parquet files from the directory
    all_files = glob.glob(directory_path + "/*.parquet")

    # Ensure the output path exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Ensure the plot output directory exists
    if plot_output_dir and not os.path.exists(plot_output_dir):
        os.makedirs(plot_output_dir)

    # Process each file one by one
    for file in tqdm(all_files, desc="Processing Parquet files"):
        takeoff_segment = process_parquet_file_lazy(
            file,
            vertical_rate_threshold,
            min_duration_threshold_minutes,
            noise_tolerance,
            plot_output_dir,
            enable_plotting,
        )
        if takeoff_segment is not None:
            # Group by flight_id and compute takeoff-specific features
            features = takeoff_segment.group_by("flight_id").agg(
                [
                    pl.col("altitude").mean().alias("mean_altitude"),
                    pl.col("altitude").max().alias("max_altitude"),
                    pl.col("groundspeed").mean().alias("mean_groundspeed"),
                    pl.col("groundspeed").max().alias("max_groundspeed"),
                    pl.col("vertical_rate").mean().alias("mean_vertical_rate"),
                    pl.col("vertical_rate").max().alias("max_vertical_rate"),
                    pl.col("tas").mean().alias("mean_tas"),
                    pl.col("tas").max().alias("max_tas"),
                    pl.col("groundspeed_diff").mean().alias("mean_groundspeed_diff"),
                    pl.col("groundspeed_diff").max().alias("max_groundspeed_diff"),
                    pl.col("track_deviation").mean().alias("mean_track_deviation"),
                    pl.col("track_variance")
                    .mean()
                    .alias("track_variance"),  # Track variance (constant)
                    (pl.col("timestamp").max() - pl.col("timestamp").min()).alias(
                        "takeoff_duration"
                    ),
                ]
            )
            # Save the processed features for each file in the same structure and format as input
            output_file = os.path.join(
                output_path, os.path.basename(file)
            )  # Keep the same filename
            features.write_parquet(output_file)
            logger.info(f"Processed and saved features for {file}")


# Example usage:
directory_path = "data"
output_path = (
    "features/trafic_features_v6"  # Output directory where results will be saved
)
plot_output_dir = (
    "reporting/plot_signals"  # Directory to save plots for skipped flights
)
vertical_rate_threshold = 500  # Threshold for vertical rate
min_duration_threshold_minutes = 1.5  # Minimum takeoff duration in minutes
noise_tolerance = 800  # Allowed noise for small decreases in

# This will process each file, segment the signal based on the vertical rate threshold, and extract features
extract_takeoff_features_lazy(
    directory_path,
    output_path,
    noise_tolerance,
    vertical_rate_threshold,
    min_duration_threshold_minutes,
    plot_output_dir,
)
