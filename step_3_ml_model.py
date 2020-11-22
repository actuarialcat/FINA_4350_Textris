#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training ML model

Created on Sat Nov 21 23:13:24 2020

@author: Jackson
"""

import pandas as pd 
from pandas.plotting import parallel_coordinates

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from plotnine import *
import datetime

from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm



###################################################
# Global Param

FEATURE_PATH_CSV =  "features/"

MAX_DATE = 202009
MIN_DATE = 201703

TRAIN_DATE = 201910           # 43 months total, 12 test month (28% test set)

###################################################
# Date functions

def as_date(inp):
    """convert yyyymm as date object, last day of the month"""
    
    out = datetime.datetime.strptime(str(inp), "%Y%m")
    out = out.replace(day = 28) + datetime.timedelta(days=4)
    
    return  out - datetime.timedelta(days = out.day)
    


###################################################
#%% Load all data

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
intel_bow = pd.read_csv(FEATURE_PATH_CSV + "Reddit_BoW_Intel_Result.csv", index_col = 0)
intel_bow = intel_bow.rename(columns = {'name': 'yyyymm'})
intel_bow = intel_bow[intel_bow["yyyymm"] >= MIN_DATE]
intel_bow = intel_bow[intel_bow["yyyymm"] <= MAX_DATE]

amd_bow = pd.read_csv(FEATURE_PATH_CSV + "Reddit_BoW_AMD_Result.csv", index_col = 0)
amd_bow = amd_bow.rename(columns = {'name': 'yyyymm'})
amd_bow = amd_bow[amd_bow["yyyymm"] >= MIN_DATE]
amd_bow = amd_bow[amd_bow["yyyymm"] <= MAX_DATE]


# Amazon Sentiment data
amazon_intel_sentiment = pd.read_csv(FEATURE_PATH_CSV + "Amazon_Sentiment_Intel_Result.csv", index_col = 0)
amazon_intel_sentiment = amazon_intel_sentiment.rename(columns = {'Month': 'yyyymm'})
amazon_intel_sentiment = amazon_intel_sentiment[amazon_intel_sentiment["yyyymm"] >= MIN_DATE]
amazon_intel_sentiment = amazon_intel_sentiment[amazon_intel_sentiment["yyyymm"] <= MAX_DATE]

amazon_amd_sentiment = pd.read_csv(FEATURE_PATH_CSV + "Amazon_Sentiment_AMD_Result.csv", index_col = 0)
amazon_amd_sentiment = amazon_amd_sentiment.rename(columns = {'Month': 'yyyymm'})
amazon_amd_sentiment = amazon_amd_sentiment[amazon_amd_sentiment["yyyymm"] >= MIN_DATE]
amazon_amd_sentiment = amazon_amd_sentiment[amazon_amd_sentiment["yyyymm"] <= MAX_DATE]


# Amazon BoW
amazon_intel_bow = pd.read_csv(FEATURE_PATH_CSV + "Amazon_BoW_Intel_Result.csv", index_col = 0)
amazon_intel_bow = amazon_intel_bow.rename(columns = {'name': 'yyyymm'})
amazon_intel_bow = amazon_intel_bow[amazon_intel_bow["yyyymm"] >= MIN_DATE]
amazon_intel_bow = amazon_intel_bow[amazon_intel_bow["yyyymm"] <= MAX_DATE]

amazon_amd_bow = pd.read_csv(FEATURE_PATH_CSV + "Amazon_BoW_AMD_Result.csv", index_col = 0)
amazon_amd_bow = amazon_amd_bow.rename(columns = {'name': 'yyyymm'})
amazon_amd_bow = amazon_amd_bow[amazon_amd_bow["yyyymm"] >= MIN_DATE]
amazon_amd_bow = amazon_amd_bow[amazon_amd_bow["yyyymm"] <= MAX_DATE]



# merge all
# df = df_summary.merge(df_sentiment, on ="yyyymm")
# df = df.merge(df_stock, on ="yyyymm")
    


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

# =============================================================================
# plt.figure(figsize = (10,7))
# plt.plot(df_summary["yyyymm"].apply(as_date), df_summary["word_count_intel"], label = "Intel")
# plt.plot(df_summary["yyyymm"].apply(as_date), df_summary["word_count_amd"], label = "AMD")
# plt.legend(fontsize = 15)
# plt.xlabel('Date', fontsize = 15)
# plt.ylabel('Word count', fontsize = 15)
# plt.set_major_formatter(ScalarFormatter())
# plt.ticklabel_format(useOffset = False)
# plt.yticks(fontsize = 15)
# plt.xticks(fontsize = 15)
# 
# =============================================================================


###################################################
#%% KNN clustering

def knn_model(inp, n_clusters, print_ind = True):
    """implement knn clusters"""
    
    X = inp.iloc[:,1:]
    kmeans = KMeans(n_clusters).fit(X)
    
    labels = inp.iloc[:,0:1]
    labels.insert(len(labels.columns), "labels", kmeans.labels_)
    
    #kmeans.cluster_centers_
    
    if (print_ind):
        print(labels)
        
    return kmeans, labels



def parallel_plot(x_all, labels, is_BoW = True):
    """parallel coordinates plot to visualise knn clusters"""
    
    plot_data = x_all.merge(labels, on = "yyyymm")
    
    plt.figure(figsize = (15,10))
    parallel_coordinates(plot_data.drop("yyyymm", axis = 1), "labels", color = ["green", "red", "blue"])

    if is_BoW:
        plt.xlabel('Words', fontsize = 15)
        plt.ylabel('Word occurrence', fontsize = 15)
        plt.xticks(rotation = 90)
    else:
        plt.xlabel('Value', fontsize = 15)
        plt.ylabel('Features', fontsize = 15)
        plt.xticks(fontsize = 15)
    
    plt.legend(loc = 1, prop={'size': 15}, frameon = True)
    plt.show()
    
    

def scatter_plot(x_all, labels):
    """scatter plot to visualise knn clusters"""
    
    color = labels.iloc[:,1].apply(lambda x: "green" if x == 0 else "red")
    
    plt.figure(figsize = (15,10))
    plt.scatter(x_all.iloc[:,1], x_all.iloc[:,2], c = color)
    plt.xlabel('Title Polarity', fontsize = 15)
    plt.ylabel('Content Polarity', fontsize = 15)
    


#%% Intel reddit sentiment 

x_all = intel_sentiment.iloc[:,[0,1,2,4,5]]      # Netural is fully correlated with pos and neg, thus not included
kmeans, labels = knn_model(x_all, 2)
parallel_plot(intel_sentiment, labels, False)


#%% AMD reddit sentiment 

x_all = amd_sentiment.iloc[:,[0,1,2,4,5]]      # Netural is fully correlated with pos and neg, thus not included
kmeans, labels = knn_model(x_all, 2)
parallel_plot(amd_sentiment, labels, False)


#%% Intel Reddit bag of words

x_all = intel_bow
kmeans, labels = knn_model(x_all, 3)
parallel_plot(x_all, labels)


#%% AMD Reddit bag of words

x_all = amd_bow
kmeans, labels = knn_model(x_all, 2)
parallel_plot(x_all, labels)


#%% Intel Amazon sentiment

x_all = amazon_intel_sentiment
kmeans, labels = knn_model(x_all, 2)
scatter_plot(x_all, labels)
#parallel_plot(x_all, labels, False)


#%% AMS Amazon sentiment

x_all = amazon_amd_sentiment
kmeans, labels = knn_model(x_all, 2)
scatter_plot(x_all, labels)
#parallel_plot(x_all, labels, False)


#%% Intel Amazon bag of words

x_all = amazon_intel_bow
kmeans, labels = knn_model(x_all, 2)
parallel_plot(x_all, labels)


#%% AMD Amazon bag of words

x_all = amazon_amd_bow
kmeans, labels = knn_model(x_all, 2)
parallel_plot(x_all, labels)



###################################################
#%% Naive Bayes

def nb_model(x_all, y_all, print_ind = True):
    """Fit a navie bayes model"""
    
    x_train = x_all[x_all["yyyymm"] < TRAIN_DATE].iloc[:,1:]
    x_test = x_all[x_all["yyyymm"] >= TRAIN_DATE].iloc[:,1:]
    y_train = y_all[y_all["yyyymm"] < TRAIN_DATE].iloc[:,1].to_numpy()
    y_test = y_all[y_all["yyyymm"] >= TRAIN_DATE].iloc[:,1].to_numpy()

    # Model
    nb = MultinomialNB()
    nb.fit(x_train, y_train)

    # Test accuracy
    pred = nb.predict(x_test)
    test_acc = metrics.accuracy_score(y_test, pred)
    conf_matrix = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
    
    if (print_ind):
        print("Test accuarcy: " + str(test_acc))
        print(conf_matrix)
        print()
    
    return nb, test_acc, conf_matrix



#%% Intel reddit sentiment vs Stock Direction

x_all = intel_sentiment.iloc[:,[0,1,2,4,5]]      # Netural is fully correlated with pos and neg, thus not included
y_all = df_stock[["yyyymm", "direction_up_next_1_intel"]]
nb, test_acc, conf_matrix = nb_model(x_all, y_all)

#nb.coef_[0]


#%% AMD reddit sentiment vs Stock Direction

x_all = amd_sentiment.iloc[:,[0,1,2,4,5]]      # Netural is fully correlated with pos and neg, thus not included
y_all = df_stock[["yyyymm", "direction_up_next_1_amd"]]
nb, test_acc, conf_matrix = nb_model(x_all, y_all)


#%% Intel reddit bag of words vs Stock Direction

x_all = intel_bow
y_all = df_stock[["yyyymm", "direction_up_next_1_intel"]]
nb, test_acc, conf_matrix = nb_model(x_all, y_all)
    

#%% AMD reddit bag of words vs Stock Direction

x_all = amd_bow
y_all = df_stock[["yyyymm", "direction_up_next_1_amd"]]
nb, test_acc, conf_matrix = nb_model(x_all, y_all)


#%% Intel Amazon sentiment vs Stock Direction

x_all = amazon_intel_sentiment
y_all = df_stock[["yyyymm", "direction_up_next_1_intel"]]
nb, test_acc, conf_matrix = nb_model(x_all, y_all)


#%% AMD Amazon sentiment vs Stock Direction

x_all = amazon_amd_sentiment
y_all = df_stock[["yyyymm", "direction_up_next_1_amd"]]
nb, test_acc, conf_matrix = nb_model(x_all, y_all)


#%% Intel Amazon bag of words vs Stock Direction

x_all = amazon_intel_bow
y_all = df_stock[["yyyymm", "direction_up_next_1_intel"]]
nb, test_acc, conf_matrix = nb_model(x_all, y_all)
    

#%% AMD Amazon bag of words vs Stock Direction

x_all = amazon_amd_bow
y_all = df_stock[["yyyymm", "direction_up_next_1_amd"]]
nb, test_acc, conf_matrix = nb_model(x_all, y_all)



###################################################
#%% Intel reddit sentiment vs Trade volumn indicator

x_all = intel_sentiment.iloc[:,[0,1,2,4,5]]      # Netural is fully correlated with pos and neg, thus not included
y_all = df_stock[["yyyymm", "direction_up_next_1_intel"]]
nb, test_acc, conf_matrix = nb_model(x_all, y_all)

#nb.coef_[0]


#%% AMD reddit sentiment vs Trade volumn indicator

x_all = amd_sentiment.iloc[:,[0,1,2,4,5]]      # Netural is fully correlated with pos and neg, thus not included
y_all = df_stock[["yyyymm", "direction_up_next_1_amd"]]
nb, test_acc, conf_matrix = nb_model(x_all, y_all)


#%% Intel reddit bag of words vs Trade volumn indicator

x_all = intel_bow
y_all = df_stock[["yyyymm", "direction_up_next_1_intel"]]
nb, test_acc, conf_matrix = nb_model(x_all, y_all)
    

#%% AMD reddit bag of words vs Trade volumn indicator

x_all = amd_bow
y_all = df_stock[["yyyymm", "direction_up_next_1_amd"]]
nb, test_acc, conf_matrix = nb_model(x_all, y_all)


#%% Intel Amazon sentiment vs Trade volumn indicator

x_all = amazon_intel_sentiment
y_all = df_stock[["yyyymm", "direction_up_next_1_intel"]]
nb, test_acc, conf_matrix = nb_model(x_all, y_all)


#%% AMD Amazon sentiment vs Trade volumn indicator

x_all = amazon_amd_sentiment
y_all = df_stock[["yyyymm", "direction_up_next_1_amd"]]
nb, test_acc, conf_matrix = nb_model(x_all, y_all)


#%% Intel Amazon bag of words vs Trade volumn indicator

x_all = amazon_intel_bow
y_all = df_stock[["yyyymm", "direction_up_next_1_intel"]]
nb, test_acc, conf_matrix = nb_model(x_all, y_all)
    

#%% AMD Amazon bag of words vs Trade volumn indicator

x_all = amazon_amd_bow
y_all = df_stock[["yyyymm", "direction_up_next_1_amd"]]
nb, test_acc, conf_matrix = nb_model(x_all, y_all)



###################################################
#%% Linear Regression

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')



def lm_model(x_all, y_all, print_ind = True, plot_ind = True):
    """Fit a linear regression model"""
    
    # data
    x_train = x_all[x_all["yyyymm"] < TRAIN_DATE].iloc[:,1:].to_numpy()
    x_test = x_all[x_all["yyyymm"] >= TRAIN_DATE].iloc[:,1:].to_numpy()
    y_train = y_all[y_all["yyyymm"] < TRAIN_DATE].iloc[:,1].to_numpy()
    y_test = y_all[y_all["yyyymm"] >= TRAIN_DATE].iloc[:,1].to_numpy()
    
    # Fit model
    X = sm.add_constant(x_train)
    lm = sm.OLS(y_train, X).fit()
    
    if print_ind:
        print(lm.summary())
        print()
    
    # test
    X_test = sm.add_constant(x_test)
    ypred = lm.predict(X_test)
    MSE = np.sum((ypred - y_test)**2) / len(y_test)

    # plot
    if plot_ind:
        # Residual plot
        plt.figure(figsize = (10,7))
        plt.scatter(lm.fittedvalues, lm.resid)
        plt.xlabel('Fitted Value', fontsize = 15)
        plt.ylabel('Residual', fontsize = 15)
        plt.title("Residual plot",  fontsize = 25)
        plt.axhline(y = 0)
        plt.show()
        
        # QQ plot
        plt.figure(figsize = (10,7))
        sm.ProbPlot(lm.resid).qqplot()
        plt.title("qq-plot",  fontsize = 25)
        abline(1,0)
        plt.show()
    
    # print
    if print_ind:
        print("MSE as % of square mean")
        print("Test MSE: {:.6f}".format(MSE / (np.mean(y_test) ** 2)))
    
    
    return lm, MSE



def null_model_MSE(y_all):
    """Test MSE for null model"""
    
    y_train = y_all[y_all["yyyymm"] < TRAIN_DATE].iloc[:,1].to_numpy()
    y_test = y_all[y_all["yyyymm"] >= TRAIN_DATE].iloc[:,1].to_numpy()
    
    MSE = np.sum((np.mean(y_train) - y_test)**2) / len(y_test)
    print("Null MSE: {:.6f}    (i.e. Predict with mean)".format(MSE / (np.mean(y_test) ** 2)))
    


#%% Intel reddit sentiment vs Return

x_all = intel_sentiment.iloc[:,[0,1,2,4,5]]      # Netural is fully correlated with pos and neg, thus not included
y_all = df_stock[["yyyymm", "ret_next_1_intel"]]
lm, MSE = lm_model(x_all, y_all)
null_model_MSE(y_all)


#%% AMD reddit sentiment vs Return

x_all = amd_sentiment.iloc[:,[0,1,2,4,5]]      # Netural is fully correlated with pos and neg, thus not included
y_all = df_stock[["yyyymm", "ret_next_1_amd"]]
lm, MSE = lm_model(x_all, y_all)
null_model_MSE(y_all)


#%% Intel Amazon sentiment vs Return

x_all = amazon_intel_sentiment
y_all = df_stock[["yyyymm", "ret_next_1_intel"]]
lm, MSE = lm_model(x_all, y_all)
null_model_MSE(y_all)


#%% AMD Amazon sentiment vs Return

x_all = amazon_amd_sentiment
y_all = df_stock[["yyyymm", "ret_next_1_amd"]]
lm, MSE = lm_model(x_all, y_all)
null_model_MSE(y_all)


###################################################
#%% Intel reddit sentiment vs Trade volume

x_all = intel_sentiment.iloc[:,[0,1,2,4,5]]      # Netural is fully correlated with pos and neg, thus not included
y_all = df_stock[["yyyymm", "volume_next_1_intel"]]
y_all["volume_next_1_intel"] = y_all["volume_next_1_intel"].apply(lambda x: np.log(x))

lm, MSE = lm_model(x_all, y_all)
null_model_MSE(y_all)


#%% AMD reddit sentiment vs Trade volume

x_all = amd_sentiment.iloc[:,[0,1,2,4,5]]      # Netural is fully correlated with pos and neg, thus not included
y_all = df_stock[["yyyymm", "volume_next_1_amd"]]
y_all["volume_next_1_amd"] = y_all["volume_next_1_amd"].apply(lambda x: np.log(x))

lm, MSE = lm_model(x_all, y_all)
null_model_MSE(y_all)


#%% AMD reddit sentiment vs Trade volume (Reduced model)

x_all = amd_sentiment.iloc[:,[0,5]]
y_all = df_stock[["yyyymm", "volume_next_1_amd"]]
y_all["volume_next_1_amd"] = y_all["volume_next_1_amd"].apply(lambda x: np.log(x))

lm, MSE = lm_model(x_all, y_all)
null_model_MSE(y_all)


#%% Intel Amazon sentiment vs Trade volume

x_all = amazon_intel_sentiment
y_all = df_stock[["yyyymm", "volume_next_1_intel"]]
y_all["volume_next_1_intel"] = y_all["volume_next_1_intel"].apply(lambda x: np.log(x))

lm, MSE = lm_model(x_all, y_all)
null_model_MSE(y_all)


#%% AMD Amazon sentiment vs Trade volume

x_all = amazon_amd_sentiment
y_all = df_stock[["yyyymm", "volume_next_1_amd"]]
y_all["volume_next_1_amd"] = y_all["volume_next_1_amd"].apply(lambda x: np.log(x))

lm, MSE = lm_model(x_all, y_all)
null_model_MSE(y_all)







