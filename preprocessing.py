import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
file_path = "D:/hack/dataset/cleaned_stock_data.csv"
data = pd.read_csv(file_path)

# Display basic info before preprocessing
print("Dataset Info Before Preprocessing:")
print(data.info())

# Convert Date column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Sort data by date (if not already sorted)
data = data.sort_values(by='Date')

# Reset index
data.reset_index(drop=True, inplace=True)

# Check for missing values and handle them (if any)
if data.isnull().sum().sum() > 0:
    print("Handling missing values...")
    data.fillna(method='ffill', inplace=True)  # Forward fill missing values

# Normalize stock prices using Min-Max Scaling
scaler = MinMaxScaler()
price_columns = ['High', 'Low', 'Open', 'Close', 'Adj Close']
data[price_columns] = scaler.fit_transform(data[price_columns])

# Display processed data
print("Preprocessed Data Sample:")
print(data.head())

# Save cleaned data (optional)
data.to_csv("cleaned_stock_data.csv", index=False)
print("Preprocessed dataset saved as 'cleaned_stock_data.csv'")
