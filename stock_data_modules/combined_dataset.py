#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 18:26:44 2022
combines stock of your choice, oil prices and 
@author: samkd
"""

# y finance client to grab the data 
import yfinance as yf
# matplotlib to plot stuff
import matplotlib.pyplot as plt
# for dataframe creation
import pandas as pd
# for arrays 
import numpy as np
import math
# for creating the lstm model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
# parameters for the lstm model
from keras.layers import LSTM, Dense, Dropout
import seaborn as sns
#sns.set_style('whitegrid')
#plt.style.use("fivethirtyeight")
from datetime import datetime
# for normalising dataset 
from sklearn.preprocessing import MinMaxScaler
# For reading stock data from yahoo
from pandas_datareader.data import DataReader
import datetime as dt
from tensorflow.keras.optimizers import Adam
import sys
np.set_printoptions(threshold=sys.maxsize)
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

def combined_stock_dataset(stock):
    df1 = oil_prices()
    #date = str(input("Please enter start date, cannot be earlier than 01-02-1988 (format Y-M-D): "))
    df = get_stock_data(stock)
    # Create a new dataframe with only the 'Close column 
    data = df.filter(['Close'])
    data.columns = [f"Close_price_{stock}"]
    data = df1.merge(data.filter([f"Close_price_{stock}"]), on = 'Date', how = 'left')
    
    us_and_uk_cpi = get_CPI()
    
    data = data.merge(us_and_uk_cpi, on = "Date", how = "left")
    data = data.dropna()
    
    
    return data