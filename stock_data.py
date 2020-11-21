#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data pre-processing and fitting random forest models

Created on Sat Nov 21 22:29:20 2020

@author: Jackson
"""

import pandas as pd 

import datetime


###################################################
# Global Param

STOCK_DATA_PATH = "Stock_data/"

FEATURE_PATH_CSV =  "features/"



###################################################
# Stock preprocess

def string_to_date_stock(inp):
    """Convert date format for stock data"""
    
    # minus 1 to denote as last date of the month    
    date = datetime.datetime.strptime(inp, "%Y-%m-%d") - datetime.timedelta(1)
    
    return date.strftime("%Y%m")



def read_stock_data(filename):
    """load stock data"""
    
    df = pd.read_csv(STOCK_DATA_PATH + filename, index_col = False)
    df.drop(df.tail(1).index,inplace=True)
    
    # Change date format and col namez
    df.rename({'Date': 'date'}, axis=1, inplace=True)
    df["yyyymm"] = df["date"].apply(string_to_date_stock)
    
    return df



def pre_process_stock_data(df):
    
    # lag values
    df["volume_next_1"] = df["Volume"].shift(periods = -1)
    df["adj_close_next_1"] = df["Adj Close"].shift(periods = -1)
    
    # calcuate returns
    df["ret_next_1"] = (df["adj_close_next_1"] / df["Adj Close"]) - 1
    df["direction_up_next_1"] = df["ret_next_1"].apply(lambda x: x > 0)
    
    
    
#%%###################################################
# AMD
    
df_amd = read_stock_data("AMD.csv")
pre_process_stock_data(df_amd)


#%% Intel

df_intel = read_stock_data("INTC.csv")
pre_process_stock_data(df_intel)


#%% merge

df_amd = df_amd[["yyyymm", "ret_next_1", "direction_up_next_1", "volume_next_1"]]
df_intel = df_intel[["yyyymm", "ret_next_1", "direction_up_next_1", "volume_next_1"]]

df = df_amd.merge(df_intel, on = "yyyymm", suffixes = ["_amd", "_intel"])

df.to_csv(FEATURE_PATH_CSV + "stock_data.csv")







