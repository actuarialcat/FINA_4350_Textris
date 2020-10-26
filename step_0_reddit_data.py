#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 12:31:58 2020

Extract reddit data from Pushshift

@author: Textrix
"""

import datetime
import requests

import json


###################################################
# Global Param

OUTPUT_PATH = "data/"
OUTPUT_FILENAME_PREFIX = "web_data_"

REQUEST_LIMIT = 100

###################################################
# Pushshift web API functions

def get_posts_limit(subreddit, start, end):
    """Get submission data inside time range, limited to 100 data due to API"""
    
    url = "https://apiv2.pushshift.io/reddit/submission/search/" \
               "?subreddit={0}" \
               "&limit={1}" \
               "&after={2}" \
               "&before={3}".format(subreddit, REQUEST_LIMIT, start, end)
    
    success = False
    while (not success):
        success = True
        try:
            response = requests.get(url, timeout = 10)
        except requests.ReadTimeout:
            success = False
            print("Timeout: Retrying")
            
    resp_json = response.json()
    return resp_json['data']



def get_posts_all(subreddit, start, end = int(datetime.datetime.now().timestamp())):
    """Get all submission data inside time range"""

    data = get_posts_limit(subreddit, start, end)
    all_data = data
    
    # Get data exceeded the limit
    while len(data) >= REQUEST_LIMIT:
        last_one = data[REQUEST_LIMIT - 1]
        start = last_one['created_utc'] + 1
        print("Completed: {0}".format(datetime.datetime.fromtimestamp(start).strftime("%Y%m%d")))
        
        data = get_posts_limit(subreddit, start, end)
        all_data.extend(data)
        
    print("Completed: {0}".format(datetime.datetime.fromtimestamp(end).strftime("%Y%m%d")))
        
    return all_data


def get_comments_limit(post_id, start = 0):
    """Get all comments data under a submission, limited to 100 data due to API"""

    url = "https://apiv2.pushshift.io/reddit/comment/search/" \
               "?link_id={0}" \
               "&limit={1}" \
               "&after={2}".format(post_id, REQUEST_LIMIT, start)
         
    success = False
    while (not success):
        success = True
        try:
            response = requests.get(url, timeout = 10)
        except requests.ReadTimeout:
            success = False
            print("Timeout: Retrying")
        except:
            print("No result")
            return []               # Invald URL, return empty string
            
    resp_json = response.json()
    return resp_json['data']



def get_comments_all(post_id):
    """Get all comments data"""

    data = get_comments_limit(post_id, 0)
    all_data = data
    
    # Get data exceeded the limit
    while len(data) >= REQUEST_LIMIT:
        last_one = data[REQUEST_LIMIT - 1]
        start = last_one["created_utc"] + 1
        #print("Completed: {0}".format(datetime.datetime.fromtimestamp(start).strftime("%Y%m%d")))
        
        data = get_comments_limit(subreddit, start)
        all_data.extend(data)
        
    #end = all_data[len(all_data)-1]["created_utc"]
    #print("Completed: {0}".format(datetime.datetime.fromtimestamp(end).strftime("%Y%m%d")))
        
    return all_data



def extract_comments(post):
    """Extract all comments in all post"""
    
    i = 0               # For console output
    total = len(post)   # For console output
    print("Starting extact comments for {0} posts".format(total))
    
    for record in post:
        record["comments"] = get_comments_all(record["id"])
        
        # For console output
        i = i + 1
        if(i % 10 == 0):
            print("Completed {0} of {1} posts".format(i, total))



###################################################
# Control Functions
    

def extract_month(subreddit, year, month, with_comments = False):
    """Extract data within a month, and output csv"""
    global post
    
    # Parameters
    start = int(datetime.datetime(year = year, month = month, day = 1).timestamp())
    if (month == 12):
        end = int(datetime.datetime(year = year + 1, month = month, day = 1).timestamp()) - 1
    else:
        end = int(datetime.datetime(year = year, month = month + 1, day = 1).timestamp()) - 1
    
    # Retrieve data
    post = get_posts_all(subreddit, start, end)
    if (with_comments):
        extract_comments(post)
    
    # Output JSON file
    filename = "{0}{1}_{2}{3}.json".format(
        OUTPUT_FILENAME_PREFIX, subreddit, str(year), str(month).zfill(2))
    
    output_json(post, filename)
    
    
def extract_year(subreddit, st_year, st_month, end_year, end_month, with_comments = False):
    while st_year < end_year or st_month <= end_month:
        extract_month(subreddit, st_year, st_month, with_comments)
        
        if (st_month == 12):
            st_month = 1
            st_year = st_year + 1
        else:
            st_month = st_month + 1
            
            
    
###################################################
# Output Functions
    
def output_json(data, filename):
    """output data into json file"""
    
    full_filename = OUTPUT_PATH + filename
    
    with open(full_filename, "w+") as outfile:
        json.dump(data, outfile)
    
    print("Outputed file: " + filename)




#%% ###################################################
# Main
    
subreddit = "intel"     #starting from 2011-01

extract_year(subreddit, 2011, 1, 2011, 1, False)


#%%

# =============================================================================
# start = int(datetime.datetime(year = 2019, month = 8, day = 1).timestamp())
# end = int(datetime.datetime(year = 2019, month = 9, day = 1).timestamp()) - 1
# =============================================================================



#%% Testing

# =============================================================================
# OUTPUT_PATH = "data_test/"
# extract_year(subreddit, 2020, 1, 2020, 1, False)
# 
# 
# =============================================================================
