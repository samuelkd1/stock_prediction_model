#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 20:08:33 2022

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
#np.set_printoptions(threshold=sys.maxsize)
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

def lstm_combined_dataset_2(stock):
    steps = int(input("How many timesteps do you wish to use?: "))
    df = combined_stock_dataset(stock)
    target = df[f"Close_price_{stock}"].values
    X_train, X_test, y_train, y_test = df.iloc[-round(len(df)*0.8):], df.iloc[:-round(len(df)*0.8)],df.iloc[-round(len(df)*0.8):][f"Close_price_{stock}"], df.iloc[:-round(len(df)*0.8)][f"Close_price_{stock}"]
    X_train, X_test, y_train, y_test = X_train.values, X_test.values, y_train.values, y_test.values
    
    # scale data x
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    #scale y
    sc2 = StandardScaler()
    y_train_scaled = sc2.fit_transform(y_train.reshape(-1, 1))
    
    y_test = y_test.reshape(-1, 1)
    
    samples = X_train_scaled.shape[0]
    features = X_train_scaled.shape[1]
    # shaping training and testing sets 
    X_train1 = []
    y_train1 = []
    
    for i in range(steps,samples):
        X_train1.append(X_train_scaled[i-steps: i])
        y_train1.append(y_train_scaled[i])
    
    X_train1, y_train1 = np.array(X_train1), np.array(y_train1)
    #print(X_train1.shape)
    #print(y_train1.shape)
    
    # creating model
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape = (X_train1.shape[1], X_train1.shape[2]), return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences = False))
    model.add(Dropout(0.2))
    model.add(Dense(y_train1.shape[1]))

    model.compile(optimizer='adam', loss='mse')
    print(model.summary())
    
    model.fit(X_train1, y_train1, batch_size = 2, epochs=10)
    
    # for test data
    samples = X_test.shape[0]
    features = X_test.shape[1]
    
    X_test1 = []
    y_test1 = []
    for i in range(steps, samples):
        X_test1.append(X_test[i - steps : i])
        y_test1.append(y_test[i])
    X_test1, y_test1 = np.array(X_test1), np.array(y_test1)
    

    y_pred = model.predict(X_test1)
    y_pred = sc2.inverse_transform(y_pred)
    
    rmse = np.sqrt(np.mean(((y_pred - y_test1) ** 2)))
    print(rmse)
    
    train = df[-round(len(df)*0.8):]
    valid = df.iloc[:-round(len(df)*0.8)-steps]
    valid['Predictions'] = y_pred
    plt.figure(figsize=(16,6))
    plt.title('stock Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train[f'Close_price_{stock}'])
    plt.plot(valid[[f'Close_price_{stock}', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()
    print(f"the rmse for {stock} is {rmse}")

    
    
