#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remove outlier

Created on Mon Nov 23 23:42:58 2020

@author: vm
"""

import pandas as pd 

reddit_id = "bwixzg"    # problematic post


df = pd.read_csv("clean_data/reddit_intel.csv", index_col = 0)

x = df[df["id"] == reddit_id]
new_df = df[df["id"] != reddit_id]

new_df.to_csv("clean_data/reddit_intel.csv")



