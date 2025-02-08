import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load Preprocessed Dataset
file_path = "dataset/cleaned_stock_data.csv"  # Update with your file path
data = pd.read_csv(file_path)

# Convert Date column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Select Close Price for prediction
close_prices = data[['Close']].values

# Normalize Data using MinMaxScaler
scaler = MinMaxScaler()
close_prices_scaled = scaler.fit_transform(close_prices)

# Create Time-Series Data
sequence_length = 50  # Use last 50 days to predict next day
X, y = [], []
for i in range(len(close_prices_scaled) - sequence_length):
    X.append(close_prices_scaled[i:i+sequence_length])
    y.append(close_prices_scaled[i+sequence_length])
X, y = np.array(X), np.array(y)

# Train-Test Split (80-20)
split = int(0.8 * len(X))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train Model
epochs = 50
batch_size = 32
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1)

# Make Predictions
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)  # Convert back to original scale
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot Predictions vs Actual
plt.figure(figsize=(12, 6))
plt.plot(data['Date'][split+sequence_length:], y_test, label='Actual Price', color='blue')
plt.plot(data['Date'][split+sequence_length:], y_pred, label='Predicted Price', color='red')
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("LSTM Model - Stock Price Prediction")
plt.legend()
plt.show()

# Save Model
model.save("lstm_stock_model.h5")