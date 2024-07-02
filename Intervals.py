import pandas as pd

# Path to the CSV file with filenames
csv_file_path = 'day_1_cloud_intervals.csv'

# Read the CSV file
data = pd.read_csv(csv_file_path)

# Group the data by 'Oribt #' and collect intervals for each orbit
grouped_data = data.groupby('Oribt #').apply(lambda x: list(zip(x['Start'], x['End']))).reset_index()
grouped_data.columns = ['Oribt #', 'Intervals']

# Print out the intervals for each orbit
for index, row in grouped_data.iterrows():
    orbit_number = row['Oribt #']
    intervals = row['Intervals']
    print(f"Orbit {orbit_number}:")
    for interval in intervals:
        print(f"  Interval: Start = {interval[0]}, End = {interval[1]}")
