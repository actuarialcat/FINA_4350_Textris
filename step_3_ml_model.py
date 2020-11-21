#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training ML model

Created on Sat Nov 21 23:13:24 2020

@author: Jackson
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics



###################################################
# Global Param

FEATURE_PATH_CSV =  "features/"

MAX_DATE = 202009
MIN_DATE = 201801


###################################################
# Load data

def load_all_features():
    """Load and merge all features in to dataframe"""
    
    # Stock data
    df_stock = pd.read_csv(FEATURE_PATH_CSV + "stock_data.csv", index_col = 0)
    df_stock = df_stock[df_stock["yyyymm"] >= MIN_DATE]
    df_stock = df_stock[df_stock["yyyymm"] <= MAX_DATE]
    
    # Summary
    intel_summary = pd.read_csv(FEATURE_PATH_CSV + "intel_summary.csv", index_col = 0)
    intel_summary = intel_summary[df_stock["yyyymm"] >= MIN_DATE]
    intel_summary = intel_summary[df_stock["yyyymm"] <= MAX_DATE]
    
    amd_summary = pd.read_csv(FEATURE_PATH_CSV + "amd_summary.csv", index_col = 0)
    amd_summary = amd_summary[df_stock["yyyymm"] >= MIN_DATE]
    amd_summary = amd_summary[df_stock["yyyymm"] <= MAX_DATE]
    
    df_summary = intel_summary.merge(amd_summary, on ="yyyymm", suffixes = ["_intel", "_amd"])
    
# =============================================================================
#     # Sentiment data
#     intel_sentiment = pd.read_csv(FEATURE_PATH_CSV + "intel_sentiment.csv", index_col = 0)
#     intel_sentiment = intel_sentiment[df_stock["yyyymm"] >= MIN_DATE]
#     intel_sentiment = intel_sentiment[df_stock["yyyymm"] <= MAX_DATE]
#     
#     amd_sentiment = pd.read_csv(FEATURE_PATH_CSV + "amd_sentiment.csv", index_col = 0)
#     amd_sentiment = amd_sentiment[df_stock["yyyymm"] >= MIN_DATE]
#     amd_sentiment = amd_sentiment[df_stock["yyyymm"] <= MAX_DATE]
#     
#     df_sentiment = intel_sentiment.merge(amd_sentiment, on ="yyyymm", suffixes = ["_intel", "_amd"])
# =============================================================================
    

    # merge all
    df = df_summary.merge(df_stock, on ="yyyymm")
    
    return df
    
    

###################################################
# Naive Bayes
    



#%%###################################################
# Load
    
df = load_all_features()


