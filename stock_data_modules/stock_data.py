# -*- coding: utf-8 -*-
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
#import seaborn as sns
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

def get_stock_data(stock):
    date = str(input("Enter stock start date (y-m-d format): "))
    # reading data from yahoo
    df = DataReader(stock, data_source ='yahoo', start = date, end = datetime.now())
    
    return df

def get_stock_data_2020(stock):
    # reading data from yahoo
    df = DataReader(stock, data_source ='yahoo', start = "2020-01-01", end = datetime.now())
    
    return df

def plot_mean_average_close_price(*stock):
    mean_list = []
    for stocks in stock:
        df = yf.download(stocks, "2020-01-01")
        mean_list.append(df["Close"].mean())
    plt.bar(stock, mean_list)
    plt.title(f"Average close price of the {len(stock)} energy companies from 2020")
    plt.ylabel('Price($)')
    for i in range(len(stock)):
        plt.text(i,mean_list[i],round(mean_list[i]),ha = "center")
    plt.show()
    
def stocks_in_one_dataset(stock,*args):
    date = str(input("Enter start date: "))
    df = yf.download(stock, date)
    df["Ticker"] = stock 
    if len(args)>= 1:
        for arg in args:
            df1 = yf.download(arg, date)
            df1["Ticker"] = arg
            df = pd.concat([df, df1])
    print(df.tail(5))
    return df


def plot_stocks_close(*stock):
    plt.figure(figsize=(16, 8))     
    date = str(input("Enter start date in y-m-d format: "))
    group = []
    #grab data
    for stocks in stock:
        data = yf.download(stock, date)
        group.append(data)
        
    # attach ticker
    
    for company, stockname in zip(group,stock):
        company["Close"].plot(label = stockname)
      
    plt.title(f"Close price of {len(stock)} energy companies from 2020", fontsize = 20)
    plt.ylabel("Close Price($)", fontsize = 18)
    plt.xlabel("DATE",fontsize = 18)
    plt.legend()
    plt.show()
    
def plot_stocks_close2(stock,date):
    group = []
    #grab data
    data = yf.download(stock, date)
    group.append(data)
        
    # attach ticker
    plt.figure(figsize=(16, 8))
    data["Close"].plot(label = stock)
            
    plt.title(f"Close price of {stock} from {date}", fontsize = 20)
    plt.ylabel("Close Price($)", fontsize = 18)
    plt.xlabel("DATE",fontsize = 18)
    plt.legend()
    plt.show()
    
def retrieve_stock_summary():
    stock = str(input("Enter ticker label of the stock you wish to summarise: "))
    df = get_stock_data(stock)
    df = df.describe()
    df["Stock"] = stock
    #print(df)
    return df


