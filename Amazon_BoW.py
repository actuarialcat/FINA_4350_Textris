import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from collections import defaultdict
import re
import datetime
import os

def calKey(year, month):
    datetime_object = datetime.datetime.strptime(month, "%B")
    return year + str(datetime_object.strftime("%m"))

stop_words = set(stopwords.words("english"))
wnl = WordNetLemmatizer()

brands = ["Intel", "AMD"]
for brand in brands:
    directory = "Product review data_{0}/".format(brand)
    review_files = os.listdir(directory)

    unigram_all_tokens = []
    bigram_all_tokens = []
    trigram_all_tokens = []
    unigram = {}
    bigram = {}
    trigram = {}
    unigram_count = {}
    bigram_count = {}
    trigram_count = {}

    for review_file in review_files:
        data = pd.read_csv(directory + review_file)
        title_total_polarity = []
        content_total_polarity = []
        for i in data.index:
            title = str(data["title"][i]).strip()
            content = str(data["content"][i]).strip()
            year = str(data["date"][i]).split(" ")[-1]
            month = str(data["date"][i]).split(" ")[-3]
            
            index = calKey(year, month)

            title_tokens = [w for w in word_tokenize(title.lower()) if w.isalpha()]
            title_filtered_tokens = [w for w in title_tokens if not w in stop_words]
            title_lemmatized_tokens = [wnl.lemmatize(w) for w in title_filtered_tokens]
            title_bigram_token = [' '.join(ng) for ng in ngrams(title_filtered_tokens, 2)]
            title_trigram_token = [' '.join(ng) for ng in ngrams(title_filtered_tokens, 3)]

            content_tokens = [w for w in word_tokenize(content.lower()) if w.isalpha()]
            content_filtered_tokens = [w for w in title_tokens if not w in stop_words]
            content_lemmatized_tokens = [wnl.lemmatize(w) for w in content_filtered_tokens]
            content_bigram_token = [' '.join(ng) for ng in ngrams(content_filtered_tokens, 2)]
            content_trigram_token = [' '.join(ng) for ng in ngrams(content_filtered_tokens, 3)]

            unigram_all_tokens += title_lemmatized_tokens + content_lemmatized_tokens
            bigram_all_tokens += title_bigram_token + content_bigram_token
            trigram_all_tokens += title_trigram_token + content_trigram_token

            if index in unigram:
                unigram[index] += title_lemmatized_tokens + content_lemmatized_tokens
                bigram[index] += title_bigram_token + content_bigram_token
                trigram[index] += title_trigram_token + content_trigram_token
                unigram_count[index] += len(title_lemmatized_tokens + content_lemmatized_tokens)
                bigram_count[index] += len(title_bigram_token + content_bigram_token)
                trigram_count[index] += len(title_trigram_token + content_trigram_token)
            else:
                unigram[index] = title_lemmatized_tokens + content_lemmatized_tokens
                bigram[index] = title_bigram_token + content_bigram_token
                trigram[index] = title_trigram_token + content_trigram_token
                unigram_count[index] = len(title_lemmatized_tokens + content_lemmatized_tokens)
                bigram_count[index] = len(title_bigram_token + content_bigram_token)
                trigram_count[index] = len(title_trigram_token + content_trigram_token)

    """
    Finding top30/20/10 most used token of unigram/bigram/trigram 
    """
    unigram_target = defaultdict(int)
    for tk in unigram_all_tokens:
        unigram_target[tk] += 1
    unigram_target = [key for key, value in sorted(unigram_target.items(), key=lambda kv: kv[1], reverse=True)[:30]]

    bigram_target = defaultdict(int)
    for tk in bigram_all_tokens:
        bigram_target[tk] += 1
    bigram_target = [key for key, value in sorted(bigram_target.items(), key=lambda kv: kv[1], reverse=True)[:20]]

    trigram_target = defaultdict(int)
    for tk in trigram_all_tokens:
        trigram_target[tk] += 1
    trigram_target = [key for key, value in sorted(trigram_target.items(), key=lambda kv: kv[1], reverse=True)[:10]]
    columns = ["name"]+unigram_target+bigram_target+trigram_target
    result = pd.DataFrame(columns=columns)

    """
    check the occurrence of most used token in each month  
    """
    for index in sorted(unigram):
        unigram_bow = defaultdict(int)
        bigram_bow = defaultdict(int)
        trigram_bow = defaultdict(int)
        resultRow = [index]
        for tk in unigram[index]:
            unigram_bow[tk] += 1
        for tk in bigram[index]:
            bigram_bow[tk] += 1
        for tk in trigram[index]:
            trigram_bow[tk] += 1
        for target in unigram_target:
            temp = 0
            try:
                temp = unigram_bow[target]/unigram_count[index]
            except:
                temp = 0
            resultRow.append(temp)
        for target in bigram_target:
            temp = 0
            try:
                temp = bigram_bow[target]/bigram_count[index]
            except:
                temp = 0
            resultRow.append(temp)
        for target in trigram_target:
            temp = 0
            try:
                temp = trigram_bow[target]/trigram_count[index]
            except:
                temp = 0
            resultRow.append(temp)

        result = result.append(pd.Series(resultRow, index=columns), ignore_index=True)

    result.to_csv("Amazon_{0}_BoW_Result.csv".format(brand))