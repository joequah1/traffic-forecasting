import warnings
import json
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def generate_data():
    # Generate sample data as a placeholder for historical traffic data
    # This function can be replaced with a function to load data from JSON or CSV files
    
	# Generate sample data for 14 days with hourly timestamps and random traffic volumes
	start_date = pd.Timestamp("2024-03-01 00:00")
	end_date = pd.Timestamp("2024-03-14 23:00")
	num_hours = int((end_date - start_date).total_seconds() / 3600) + 1

	timestamps = pd.date_range(start=start_date, end=end_date, freq='H')
	traffic_volumes = np.random.randint(50, 400, size=num_hours)  # Random traffic volumes between 50 and 300

	sample_data = [{"timestamp": ts, "traffic_volume": vol} for ts, vol in zip(timestamps, traffic_volumes)]
	
	# Open and read the JSON file
	with open('data.json', 'r') as file:
	    sample_data = json.load(file)

	return sample_data

# Load historical traffic data (Example: Load from JSON file)
traffic_data = generate_data()

# Preprocess the data (Convert to DataFrame and set timestamp as index)
traffic_df = pd.DataFrame(traffic_data)
traffic_df['timestamp'] = pd.to_datetime(traffic_df['timestamp'])
traffic_df.set_index('timestamp', inplace=True)

# Fit ARIMA model to historical traffic data
model = ARIMA(traffic_df['traffic_volume'], order=(5,1,0))
model_fit = model.fit()

# Forecast future traffic volume using the fitted ARIMA model
forecast_steps = 24 * 7  # Forecast for the next 7 days, considering hourly data
forecast = model_fit.forecast(steps=forecast_steps)

# Determine the hours where congestion occurs
congestion_threshold = 200  # Adjust according to your congestion definition
congested_hours = traffic_df[traffic_df['traffic_volume'] > congestion_threshold].index

# Group the congested hours by day
congested_hours_by_day = congested_hours.groupby(congested_hours.date)

print("Hourly Congestion Forecast for the Next 7 Days:")
for date, hours in congested_hours_by_day.items():
    print(f"\n{date.strftime('%Y-%m-%d')}:")
    for hour in hours:
        print(hour.strftime('%H:%M'), end=' ')
