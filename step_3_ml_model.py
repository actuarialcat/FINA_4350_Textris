#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training ML model

Created on Sat Nov 21 23:13:24 2020

@author: Jackson
"""

import pandas as pd 
import numpy as np

from plotnine import *
import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics




###################################################
# Global Param

FEATURE_PATH_CSV =  "features/"

MAX_DATE = 202009
MIN_DATE = 201703



###################################################
# Date functions

def as_date(inp):
    """convert yyyymm as date object, last day of the month"""
    
    out = datetime.datetime.strptime(str(inp), "%Y%m")
    out = out.replace(day = 28) + datetime.timedelta(days=4)
    
    return  out - datetime.timedelta(days = out.day)
    
    


###################################################
#%% Load

# Stock data
df_stock = pd.read_csv(FEATURE_PATH_CSV + "stock_data.csv", index_col = 0)
df_stock = df_stock[df_stock["yyyymm"] >= MIN_DATE]
df_stock = df_stock[df_stock["yyyymm"] <= MAX_DATE]


# Reddit Summary
intel_summary = pd.read_csv(FEATURE_PATH_CSV + "Reddit_Summary_Intel_Result.csv", index_col = 0)
intel_summary = intel_summary[intel_summary["yyyymm"] >= MIN_DATE]
intel_summary = intel_summary[intel_summary["yyyymm"] <= MAX_DATE]

amd_summary = pd.read_csv(FEATURE_PATH_CSV + "Reddit_Summary_AMD_Result.csv", index_col = 0)
amd_summary = amd_summary[amd_summary["yyyymm"] >= MIN_DATE]
amd_summary = amd_summary[amd_summary["yyyymm"] <= MAX_DATE]

df_summary = intel_summary.merge(amd_summary, on ="yyyymm", suffixes = ["_intel", "_amd"])


# Reddit Sentiment data
intel_sentiment = pd.read_csv(FEATURE_PATH_CSV + "Reddit_Sentiment_intel_Result.csv")
intel_sentiment = intel_sentiment[intel_sentiment["yyyymm"] >= MIN_DATE]
intel_sentiment = intel_sentiment[intel_sentiment["yyyymm"] <= MAX_DATE]

amd_sentiment = pd.read_csv(FEATURE_PATH_CSV + "Reddit_Sentiment_AMD_Result.csv")
amd_sentiment = amd_sentiment[amd_sentiment["yyyymm"] >= MIN_DATE]
amd_sentiment = amd_sentiment[amd_sentiment["yyyymm"] <= MAX_DATE]

df_sentiment = intel_sentiment.merge(amd_sentiment, on ="yyyymm", suffixes = ["_intel", "_amd"])


# Reddit BoW
intel_bow = pd.read_csv(FEATURE_PATH_CSV + "Reddit_BoW_Intel_Result.csv")
intel_bow = intel_bow.rename(columns = {'name': 'yyyymm'})
intel_bow = intel_bow[intel_bow["yyyymm"] >= MIN_DATE]
intel_bow = intel_bow[intel_bow["yyyymm"] <= MAX_DATE]

amd_bow = pd.read_csv(FEATURE_PATH_CSV + "Reddit_BoW_AMD_Result.csv")
amd_bow = amd_bow.rename(columns = {'name': 'yyyymm'})
amd_bow = amd_bow[amd_bow["yyyymm"] >= MIN_DATE]
amd_bow = amd_bow[amd_bow["yyyymm"] <= MAX_DATE]


# Amazon Sentiment data
amazon_intel_sentiment = pd.read_csv(FEATURE_PATH_CSV + "Amazon_Sentiment_Intel_Result.csv")
amazon_intel_sentiment = amazon_intel_sentiment.rename(columns = {'Month': 'yyyymm'})
amazon_intel_sentiment = amazon_intel_sentiment[amazon_intel_sentiment["yyyymm"] >= MIN_DATE]
amazon_intel_sentiment = amazon_intel_sentiment[amazon_intel_sentiment["yyyymm"] <= MAX_DATE]

amazon_amd_sentiment = pd.read_csv(FEATURE_PATH_CSV + "Amazon_Sentiment_AMD_Result.csv")
amazon_amd_sentiment = amazon_amd_sentiment.rename(columns = {'Month': 'yyyymm'})
amazon_amd_sentiment = amazon_amd_sentiment[amazon_amd_sentiment["yyyymm"] >= MIN_DATE]
amazon_amd_sentiment = amazon_amd_sentiment[amazon_amd_sentiment["yyyymm"] <= MAX_DATE]


# Amazon BoW
amazon_intel_bow = pd.read_csv(FEATURE_PATH_CSV + "Amazon_BoW_Intel_Result.csv")
amazon_intel_bow = amazon_intel_bow.rename(columns = {'name': 'yyyymm'})
amazon_intel_bow = amazon_intel_bow[amazon_intel_bow["yyyymm"] >= MIN_DATE]
amazon_intel_bow = amazon_intel_bow[amazon_intel_bow["yyyymm"] <= MAX_DATE]

amazon_amd_bow = pd.read_csv(FEATURE_PATH_CSV + "Amazon_BoW_AMD_Result.csv")
amazon_amd_bow = amazon_amd_bow.rename(columns = {'name': 'yyyymm'})
amazon_amd_bow = amazon_amd_bow[amazon_amd_bow["yyyymm"] >= MIN_DATE]
amazon_amd_bow = amazon_amd_bow[amazon_amd_bow["yyyymm"] <= MAX_DATE]



# merge all
# df = df_summary.merge(df_sentiment, on ="yyyymm")
# df = df.merge(df_stock, on ="yyyymm")
    


#%% Naive Bayes






#%% 
    


    



#%% Plot Data Summery

plot_data = pd.melt(df_summary[["yyyymm", "word_count_intel", "word_count_amd"]], id_vars = "yyyymm")
plot_data["yyyymm"] = plot_data["yyyymm"].apply(as_date)

print(
     ggplot(plot_data)
     + aes(x = "yyyymm", y = "value", color = "variable") 
     + ggtitle("Word count of reddit post")
     + xlab("Date")
     + ylab("Word count")
     + geom_line() 
)



