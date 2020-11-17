#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 02:53:28 2020

Anaylsis Reddit data

@author: Textrix
"""

import pandas as pd 
from pathlib import Path

import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

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


def remove_stopwords(txt):
    """Remove stopwords using NLTK"""
    
    pass


#%%###################################################
# Load file

red_intel = load_cvs("reddit_intel.csv")
red_intel["text"] = red_intel[["sub_text", "comm_text"]].fillna("").apply(lambda x: "\n".join(x), axis = 1)


#%%

df = red_intel[["last_mod", "text"]]

df["clean_text"] = df["text"].map(clean_whitespace)


#%%
v = CountVectorizer(stop_words = "english")

bow = v.fit_transform(df["clean_text"])

v.get_feature_names()
v.transform(["Something completely new."]).toarray()

