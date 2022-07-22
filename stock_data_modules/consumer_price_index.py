#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 17:58:55 2022

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

def get_CPI():
    date = str(input("Please enter start date, cannot be earlier than 01-02-1988 (format Y-M-D): "))
    #uk
    uk_cpi = pd.read_csv("uk_cpi_energy.csv",skiprows = 7)
    uk_cpi.columns = ["Date", "CPI_percentage"]
    uk_cpi["Date"] = pd.to_datetime(uk_cpi["Date"], format = "%Y %b")
    uk_cpi = uk_cpi.set_index("Date").resample('D').ffill()
    uk_cpi = uk_cpi.loc[date:]
    #us
    us_cpi = pd.read_excel("us_energy_cpi.xlsx",skiprows = 11)
    us_cpi = us_cpi[["Year", "Value"]]
    us_cpi.columns = ["Date", "CPI_rate"]
    us_cpi["Date"] = pd.to_datetime(us_cpi["Date"], format = "%Y-%M-%D")
    us_cpi = us_cpi.set_index("Date").resample('D').ffill()
    #both
    us_and_uk_cpi = us_cpi.merge(uk_cpi, on = 'Date', how = 'inner')
    us_and_uk_cpi.columns = ["CPI_rate_US", "CPI_rate_UK"]

    return us_and_uk_cpi

def plot_CPI():
    plt.figure(figsize = (16,8))
    get_CPI()["CPI_rate_US"].plot()
    get_CPI()["CPI_rate_UK"].plot()
    plt.title("Consumer_price_index")
    plt.show()
    
    
