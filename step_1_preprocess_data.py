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
# data formatting functions
    
def format_comments(comments):
    """Normalize comments into tables"""
    
    if len(comments) == 0:
        body = ""
        last_utc = 0
    else:
        comm_raw_df = pd.json_normalize(comments)
        
        body_arr = comm_raw_df["body"]
        body = "\n".join(body_arr)
        
        last_utc = max(comm_raw_df["created_utc"])
    
    return body, last_utc
    


def format_submission(data):
    """Normalize data into tables"""

    if len(data) == 0:
        return pd.DataFrame()

    raw_df = pd.json_normalize(data)
    
    # comment text
    if "comments" in raw_df.columns:
        raw_df["comm_text"], raw_df["last_utc"] = zip(*raw_df["comments"].map(format_comments))
    else:
        raw_df["comm_text"] = ""
        raw_df["last_utc"] = 0
    
    # submission text
    raw_df["sub_text"] = raw_df[["title", "selftext"]].fillna('').apply(lambda x: "\n".join(x), 1)
    
    # time
    raw_df["last_utc"] = raw_df[["created_utc", "last_utc"]].apply(lambda x: max(x), 1)
    raw_df["created"] = raw_df["created_utc"].map(lambda x: datetime.datetime.fromtimestamp(x).strftime("%Y%m%d"))
    raw_df["last_mod"] = raw_df["last_utc"].map(lambda x: datetime.datetime.fromtimestamp(x).strftime("%Y%m%d"))
    
    df = raw_df[["created","last_mod","sub_text","comm_text","id","full_link"]]
    
    return df


def format_all_year(subreddit, st_year, st_month, end_year, end_month):
    """Format all data"""
    
    df = pd.DataFrame()
    
    while st_year < end_year or (st_year == end_year and st_month <= end_month):
        filename = "{0}{1}_{2}{3}.json".format(
            OUTPUT_FILENAME_PREFIX, subreddit, str(st_year), str(st_month).zfill(2))
        
        data = load_json(filename)
        new_df = format_submission(data)
        
        df = pd.concat([df, new_df])
        print("Completed {0}{1}".format(str(st_year), str(st_month).zfill(2)))
        
        if (st_month == 12):
            st_month = 1
            st_year = st_year + 1
        else:
            st_month = st_month + 1
    
    return df



#%% ###################################################
# Intel
    
subreddit = "intel"
OUTPUT_PATH_JSON = "intel_raw_data/"

df = format_all_year(subreddit, 2011, 1, 2020, 9)

output_cvs(df, "reddit_intel.csv")



#%% AMD

subreddit = "amd"
OUTPUT_PATH_JSON = "AMD_no_comments_data/"


df = format_all_year(subreddit, 2011, 1, 2020, 9)

output_cvs(df, "reddit_amd.csv")