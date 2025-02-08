import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Preprocessed Dataset
file_path = "dataset/cleaned_stock_data.csv"  
data = pd.read_csv(file_path)

# Convert Date column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Set seaborn style
sns.set_style("darkgrid")

# Closing Price Trend Over Time
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Close'], label="Closing Price", color='blue')
plt.xlabel("Date")
plt.ylabel("Normalized Closing Price")
plt.title("Stock Closing Price Trend Over Time")
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Moving Averages (SMA & EMA)
data['SMA_50'] = data['Close'].rolling(window=50).mean()  # 50-day Simple Moving Average
data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()  # 50-day Exponential Moving Average

plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Close'], label="Closing Price", color='blue', alpha=0.6)
plt.plot(data['Date'], data['SMA_50'], label="50-Day SMA", color='red', linestyle='dashed')
plt.plot(data['Date'], data['EMA_50'], label="50-Day EMA", color='green', linestyle='dotted')
plt.xlabel("Date")
plt.ylabel("Normalized Price")
plt.title("Stock Closing Price with Moving Averages")
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Stock Price Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data['Close'], bins=50, kde=True, color='purple')
plt.title("Stock Price Distribution")
plt.xlabel("Normalized Closing Price")
plt.ylabel("Frequency")
plt.show()

# Volume vs. Closing Price Scatter Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x=data['Volume'], y=data['Close'], alpha=0.5, color='orange')
plt.title("Stock Volume vs. Closing Price")
plt.xlabel("Trading Volume")
plt.ylabel("Normalized Closing Price")
plt.show()
