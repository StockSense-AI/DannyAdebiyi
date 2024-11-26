# Import necessary libraries
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import datetime

# Step 2: Feature Engineering
def preprocess_data(stock_data):
   stock_data['MA_10'] = stock_data['Close'].rolling(window=10).mean()
   stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()
   stock_data['Lag_1'] = stock_data['Close'].shift(1)
   stock_data.dropna(inplace=True)
   return stock_data