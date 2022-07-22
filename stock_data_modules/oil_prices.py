#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 17:44:49 2022

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

# get oil prices
def oil_prices():
    date = str(input("Enter start date in y-m-d format: "))
    oil_prices = pd.read_csv("oil_price_brent.csv", skiprows = 4)
    #oil_prices = oil_prices.loc[:639]
    oil_prices["Day"] = pd.to_datetime(oil_prices["Day"], format = "%m/%d/%Y")
    oil_prices.columns = ["Date", "oil_price_per_barrel"]
    oil_prices = oil_prices.set_index("Date")
    oil_prices = oil_prices.loc[date:]
    return oil_prices

def oil_prices_2020():
    oil_prices = pd.read_csv("oil_price_brent.csv", skiprows = 4)
    #oil_prices = oil_prices.loc[:639]
    oil_prices["Day"] = pd.to_datetime(oil_prices["Day"], format = "%m/%d/%Y")
    oil_prices.columns = ["Date", "oil_price_per_barrel"]
    oil_prices = oil_prices.set_index("Date")
    oil_prices = oil_prices.loc["2020-01-01":]
    return oil_prices


def oil_price_graph():
    df = oil_prices()
    #plotting graph 
    plt.figure(figsize = (16,8))
    df["oil_price_per_barrel"].plot()
    plt.title("Price of oil from start date")
    plt.ylabel("Close Price($)", fontsize = 18)
    plt.show()