#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 13:52:18 2020

@author: jackson
"""


import pandas as pd 
import textract
import re



INPUT_PDF_PATH = "Data/"



def extract_text(file_name):
    """Extract text in single file"""
    
    file_path = INPUT_PDF_PATH + file_name
    text = textract.process(file_path, check_extractable=False)

    text_string = str(text).lower()
    
    return text_string


def remove_white_space_tags(inp):
    """Remove raw tags in text data"""
    
    clean_txt = re.sub(r'\\x..', '', inp)
    clean_txt = re.sub(r'\\.', ' ', clean_txt)
    
    return (clean_txt)


def word_count(str):
    """Count occurrences of each word"""
    counts = dict()
    text_ary = str.split()

    for text in text_ary:
            if text in counts:
                counts[text] += 1
            else:
                counts[text] = 1
    
    sorted_counts = sorted(counts.items(), key=lambda x:x[1], reverse=True)
    
    return sorted_counts
    

#%% Extract text from pdf #####################################

x = extract_text("aia_2019.pdf")


#%% Clean text #####################################

raw_text = remove_white_space_tags(x)

#%% Word count #####################################

counts = word_count(raw_text)

#%% Output #####################################

#clean_text.to_json("clean_text")
























