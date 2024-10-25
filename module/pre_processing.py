import pandas as pd

# Load the dataset from the CSV file
train_df = pd.read_csv('csv_set/challenge_set.csv')
airport_dataset_df = pd.read_csv('csv_set/airports_database.csv', on_bad_lines='skip')

# Select only the necessary columns from the airports database
airports_db = airport_dataset_df[['icao', 'elevation']].drop_duplicates(subset='icao').rename(columns={'icao': 'adep', 'elevation': 'dep_airport_altitude'})

# Merge the airport altitude based on the 'adep' column
train_df_with_altitude = train_df.merge(airports_db, on='adep', how='left')
train_df_with_altitude.to_csv('csv_set/challenge_set_v2.csv', index=False)

# Calculate mean tow for each aircraft type
mean_tow_per_type = train_df_with_altitude.groupby('aircraft_type')['tow'].mean().reset_index()
train_df_merged_mean = train_df_with_altitude.merge(mean_tow_per_type, on='aircraft_type', suffixes=('', '_mean'))

# Rename the new column
train_df_merged_mean.rename(columns={'tow_mean': 'mean_tow_per_aircraft_type'}, inplace=True)

# Compute the difference between mean tow and actual tow
train_df_merged_mean['mean_tow_difference'] = train_df_merged_mean['mean_tow_per_aircraft_type'] - train_df_merged_mean['tow']

# Save the updated DataFrame with mean tow difference
train_df_merged_mean.to_csv('csv_set/challenge_mean_tow_diff.csv', index=False)
mean_tow_per_type.to_csv('csv_set/mean_tow_per_aircraft.csv', index=False)

# Reload the altitude data to avoid overwriting columns
train_df_with_altitude = pd.read_csv('csv_set/challenge_set_v2.csv')

# Calculate median tow for each aircraft type
median_tow_per_type = train_df_with_altitude.groupby('aircraft_type')['tow'].median().reset_index()
train_df_merged_median = train_df_with_altitude.merge(median_tow_per_type, on='aircraft_type', suffixes=('', '_median'))

# Rename the new column for median tow
train_df_merged_median.rename(columns={'tow_median': 'median_tow_per_aircraft_type'}, inplace=True)

# Compute the difference between median tow and actual tow
train_df_merged_median['median_tow_difference'] = train_df_merged_median['median_tow_per_aircraft_type'] - train_df_merged_median['tow']

# Save the updated DataFrame with median tow difference
train_df_merged_median.to_csv('csv_set/challenge_median_tow_diff.csv', index=False)
median_tow_per_type.to_csv('csv_set/median_tow_per_aircraft.csv', index=False)
