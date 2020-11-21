#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 02:53:28 2020

Anaylsis Reddit data

@author: Textrix
"""

import pandas as pd 
from pathlib import Path

#import re

#from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
#from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#from collections import Counter
#from sklearn.feature_extraction.text import CountVectorizer

###################################################
# Global Param

OUTPUT_PATH_CSV = "clean_data/"
OUTPUT_FILENAME_PREFIX = "web_data_"


###################################################
# Load file

def load_cvs(filename):
    """load dataframe from csv file"""
    
    full_filename = Path(OUTPUT_PATH_CSV + filename) 
    return pd.read_csv(full_filename, index_col = 0)



###################################################
# data cleaning functions
    
def clean_whitespace(txt):
    """Remove all excess whitespace and change all letters to lowercase"""
    
    #txt = re.sub(r'\n', ' ', txt)
    
    txt = " ".join(txt.split())
    txt = str(txt).lower()
    
    return txt


#def remove_stopwords(txt):
#    """Remove stopwords using NLTK"""
#    pass


def convert_to_monthly(inp_df):
    """Clean and group text data by month"""

    # Concat submission and comments
    inp_df["text"] = inp_df[["sub_text", "comm_text"]].fillna("").apply(lambda x: "\n".join(x), axis = 1)    
    
    # Keep useful fields
    df = inp_df[["last_mod", "text"]]
    df["yyyymm"] = df["last_mod"].map(lambda x: str(x)[:6])
    
    # clean whitespace
    df["clean_text"] = df["text"].map(clean_whitespace)
    
    # Group
    df = df[["yyyymm", "clean_text"]].groupby(["yyyymm"])["clean_text"].apply(lambda x: " ".join(x)).reset_index()
    
    return df



###################################################
# Sentiment
    
def text_sentiment_NLTK(inp, sid):
    """Extract sentiment score from text using NLTK Vader"""
    
    if (isinstance(inp, str)):
        x = sid.polarity_scores(inp)
        return x["compound"]
    
    else:
        return 0        # for nan data



#%%###################################################
# Load file

red_intel = load_cvs("reddit_intel.csv").reset_index()


#%% Format

df = convert_to_monthly(red_intel)


#%% Sentiment

sid = SentimentIntensityAnalyzer()

df["sentiment"] = df["clean_text"].map(lambda x: text_sentiment_NLTK(x, sid))