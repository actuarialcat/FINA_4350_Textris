import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import datetime
import os

def calKey(year, month):
    datetime_object = datetime.datetime.strptime(month, "%B")
    return year + str(datetime_object.strftime("%m"))

sid = SentimentIntensityAnalyzer()
brands = ["Intel", "AMD"]
for brand in brands:
    directory = "Product review data_{0}/".format(brand)
    review_files = os.listdir(directory)
    
    result = pd.DataFrame(columns=["Month", "Title Polarity", "Content Polarity"])
    title_polarity_sum = {}
    title_polarity_count = {}
    content_polarity_sum = {}
    content_polarity_count = {}

    for review_file in review_files:
        data = pd.read_csv(directory + review_file)
        print(review_file)
        title_total_polarity = []
        content_total_polarity = []
        for i in data.index:
            title = str(data["title"][i]).strip()
            content = str(data["content"][i]).strip()
            year = str(data["date"][i]).split(" ")[-1]
            month = str(data["date"][i]).split(" ")[-3]
            
            index = calKey(year, month)
            title_polarity = sid.polarity_scores(title)['compound']
            content_polarity = sid.polarity_scores(content)['compound']

            if index in title_polarity_sum:
                title_polarity_sum[index] += title_polarity
                title_polarity_count[index] += 1
                content_polarity_sum[index] += content_polarity
                content_polarity_count[index] += 1
            else:
                title_polarity_sum[index] = title_polarity
                title_polarity_count[index] = 1
                content_polarity_sum[index] = content_polarity
                content_polarity_count[index] = 1

    for key in sorted(title_polarity_sum):
        title_polarity_avg = title_polarity_sum[key] / title_polarity_count[key]
        content_polarity_avg = content_polarity_sum[key] / content_polarity_count[key]
        result = result.append({'Month': key, 'Title Polarity': title_polarity_avg, 'Content Polarity': content_polarity_avg}, ignore_index=True)

    result.to_csv("Amazon_Sentiment_{0}_Result.csv".format(brand))
