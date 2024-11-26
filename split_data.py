# Import necessary libraries
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import datetime

# Step 3: Train-Test Split
def split_data(stock_data):
   X = stock_data[['Open', 'High', 'Low', 'Volume', 'MA_10', 'MA_50', 'Lag_1']]
   y = stock_data['Close']
   return train_test_split(X, y, test_size=0.2, random_state=42)