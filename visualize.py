import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib as mpl

# Font settings
mpl.rc('font', family='Arial')  # Change to a font that supports your language
mpl.rc('axes', unicode_minus=False)  # Fix for minus sign display

# Load the saved model
model = tf.keras.models.load_model('train50.h5')

# Load and preprocess Bitcoin data
data = pd.read_csv('btc_data.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

features = data[['Open', 'High', 'Low', 'Close']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)

# Create dataset function
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), :])
        y.append(data[i + time_step, 3])
    return np.array(X), np.array(y)

# Use the last 30 days of data for prediction
time_step = 30
start_index = len(scaled_features) - 31  # Start from 31 days ago
new_X = scaled_features[start_index:-1]  # Get data from 31 days ago to 1 day ago
new_X = new_X.reshape((1, time_step, new_X.shape[1]))  # Reshape for LSTM input

# Prediction
next_day_price = model.predict(new_X)

# Create array for predicted price
predicted_price_array = np.array([[0, 0, 0, next_day_price[0][0]]])
predicted_price_inversed = scaler.inverse_transform(predicted_price_array)[0][3]

# Get the actual value for the predicted day
actual_price = data['Close'].iloc[-1]  # Last day's actual close price
actual_price_inversed = actual_price  # Since it's already in the original scale

# Dates for plotting
dates = data.index[-31:-1]  # Get dates from 31 days ago to 1 day ago
predicted_date = data.index[-1] + pd.Timedelta(days=1)  # Next day after the last date

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(dates, scaler.inverse_transform(scaled_features[start_index:-1])[:, 3], label='Last 30 Days Close Price', marker='o', color='blue')  # Close price
plt.scatter(predicted_date, predicted_price_inversed, color='red', label='Predicted Next Day Price', s=100)  # Predicted price as a point
plt.scatter(predicted_date, actual_price_inversed, color='blue', label='Actual Next Day Price', s=100)  # Actual price as a point
plt.title('Bitcoin Last 30 Days Prices and Next Day Predicted Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.show()
