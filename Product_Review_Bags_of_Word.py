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

    result = pd.DataFrame(columns=["Month", 'Unigram Top 30', 'Bigram Top 20', 'Trigram Top 10'])
    unigram = {}
    bigram = {}
    trigram = {}

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

            if index in unigram:
                unigram[index] += title_lemmatized_tokens + content_lemmatized_tokens
                bigram[index] += title_bigram_token + content_bigram_token
                trigram[index] += title_trigram_token + content_trigram_token
            else:
                unigram[index] = title_lemmatized_tokens + content_lemmatized_tokens
                bigram[index] = title_bigram_token + content_bigram_token
                trigram[index] = title_trigram_token + content_trigram_token

    for key in sorted(unigram):
        unigram_bow = defaultdict(int)
        bigram_bow = defaultdict(int)
        trigram_bow = defaultdict(int)
        for tk in unigram[key]:
            unigram_bow[tk] += 1
        for tk in bigram[key]:
            bigram_bow[tk] += 1
        for tk in trigram[key]:
            trigram_bow[tk] += 1
        unigram_bow = sorted(unigram_bow.items(), key=lambda kv: kv[1], reverse=True)[:30]
        bigram_bow = sorted(bigram_bow.items(), key=lambda kv: kv[1], reverse=True)[:20]
        trigram_bow = sorted(trigram_bow.items(), key=lambda kv: kv[1], reverse=True)[:10]
        result = result.append({'Month': key, 'Unigram Top 30': unigram_bow, 'Bigram Top 20': bigram_bow, 'Trigram Top 1': trigram_bow}, ignore_index=True)

    result.to_csv("BoW_Result_{0}.csv".format(brand))
