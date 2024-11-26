# Import necessary libraries
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import datetime

# Step 1: Fetch Stock Data
def fetch_stock_data(ticker, start_date, end_date):
   stock_data = yf.download(ticker, start=start_date, end=end_date)
   return stock_data