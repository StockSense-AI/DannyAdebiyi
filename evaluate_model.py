# Import necessary libraries
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import datetime

# Step 5: Evaluate Model
def evaluate_model(model, X_test, y_test):
   y_pred = model.predict(X_test)
   mae = mean_absolute_error(y_test, y_pred)
   rmse = mean_squared_error(y_test, y_pred, squared=False)
   r2 = r2_score(y_test, y_pred)
   return y_pred, mae, rmse, r2