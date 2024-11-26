# Import necessary libraries
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import datetime

# Step 4: Train Linear Regression Model
def train_model(X_train, y_train):
   model = LinearRegression()
   model.fit(X_train, y_train)
   return model