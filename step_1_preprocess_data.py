#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 12:31:58 2020

Convert JSON Reddit data into dataframe

@author: Textrix
"""

import json

import datetime
import pandas as pd 
from pathlib import Path

###################################################
# Global Param

OUTPUT_PATH_JSON = "data/"
OUTPUT_PATH_CSV = "clean_data/"
OUTPUT_FILENAME_PREFIX = "web_data_"


###################################################
# file functions

def load_json(filename):
    """Load json file"""
    global file_data
    
    full_filename = OUTPUT_PATH_JSON + filename
    file = open(full_filename, "r")
    file_data = file.read()
    
    return json.loads(file_data)



def output_cvs(df, filename):
    """output dataframe into csv file"""
    
    full_filename = Path(OUTPUT_PATH_CSV + filename) 
    df.to_csv(full_filename)
    
    print("Outputed file: " + filename)



###################################################
# data cleaning functions



    
    


#%% ###################################################
# Main
    
subreddit = "intel"
year = 2019
month = 1

filename = "{0}{1}_{2}{3}.json".format(
    OUTPUT_FILENAME_PREFIX, subreddit, str(year), str(month).zfill(2))

data = load_json(filename)


