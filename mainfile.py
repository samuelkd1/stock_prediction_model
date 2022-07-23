#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 00:10:23 2022

will obtain stocks for the top 6 energy companies 

@author: samkd
"""
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
#np.set_printoptions(threshold=sys.maxsize)
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from stock_data_modules.stock_data import *

big_6 = ["NEE","DUK","NGG","SO", "D", "AEP"]

# get stock data big_6

for stock in big_6:
    get_stock_data(stock)

# get mean average data 
plot_mean_average_close_price("NEE","AEP","NGG","D","SO","D")

stocks_in_one_dataset("NEE","AEP","NGG","D","SO","DUK")

plot_stocks_close("NEE","AEP","NGG","D","SO","DUK")



retrieve_stock_summary()

for stocks in big_6:
    lstm_standalone_stock(stocks, "2020-01-01")

