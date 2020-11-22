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

data = pd.read_csv("reddit_intel.csv")

unigram_all_tokens = []
bigram_all_tokens = []
trigram_all_tokens = []
unigram = {}
bigram = {}
trigram = {}
unigram_count = {}
bigram_count = {}
trigram_count = {}

for i in data.index:
    sub_text = re.sub("(?P<url>http[^\s]+)", "", str(data["sub_text"][i]))
    sub_text = re.sub("&amp;#x200B;", "", sub_text)
    comment_text = re.sub("(?P<url>http[^\s]+)", "", str(data["comm_text"][i]))
    comment_text = re.sub("&amp;#x200B;", "", comment_text)
    index = str(data["created"][i])[:6]

    #For sub text
    sub_text_tokens = [w for w in word_tokenize(sub_text.lower()) if w.isalpha()]
    sub_text_filtered_tokens = [w for w in sub_text_tokens if not w in stop_words]
    sub_text_lemmatized_tokens = [wnl.lemmatize(w) for w in sub_text_filtered_tokens]
    sub_text_bigram_token = [' '.join(ng) for ng in ngrams(sub_text_filtered_tokens, 2)]
    sub_text_trigram_token = [' '.join(ng) for ng in ngrams(sub_text_filtered_tokens, 3)]

    #For comment text
    comment_text_tokens = [w for w in word_tokenize(comment_text.lower()) if w.isalpha()]
    comment_text_filtered_tokens = [w for w in comment_text_tokens if not w in stop_words]
    comment_text_lemmatized_tokens = [wnl.lemmatize(w) for w in comment_text_filtered_tokens]
    comment_text_bigram_token = [' '.join(ng) for ng in ngrams(comment_text_filtered_tokens, 2)]
    comment_text_trigram_token = [' '.join(ng) for ng in ngrams(comment_text_filtered_tokens, 3)]

    #Storing all tokens regardless of date
    unigram_all_tokens += sub_text_lemmatized_tokens + comment_text_lemmatized_tokens
    bigram_all_tokens += sub_text_bigram_token + comment_text_bigram_token
    trigram_all_tokens += sub_text_trigram_token + comment_text_trigram_token

    if index in unigram:
        unigram[index] += sub_text_lemmatized_tokens + comment_text_lemmatized_tokens
        bigram[index] += sub_text_bigram_token + comment_text_bigram_token
        trigram[index] += sub_text_trigram_token + comment_text_trigram_token
        unigram_count[index] += len(sub_text_lemmatized_tokens + comment_text_lemmatized_tokens)
        bigram_count[index] += len(sub_text_bigram_token + comment_text_bigram_token)
        trigram_count[index] += len(sub_text_trigram_token + comment_text_trigram_token)
    else:
        unigram[index] = sub_text_lemmatized_tokens + comment_text_lemmatized_tokens
        bigram[index] = sub_text_bigram_token + comment_text_bigram_token
        trigram[index] = sub_text_trigram_token + comment_text_trigram_token
        unigram_count[index] = len(sub_text_lemmatized_tokens + comment_text_lemmatized_tokens)
        bigram_count[index] = len(sub_text_bigram_token + comment_text_bigram_token)
        trigram_count[index] = len(sub_text_trigram_token + comment_text_trigram_token)

"""
Finding top30/20/10 most used token of unigram/bigram/trigram 
"""
unigram_target = defaultdict(int)
for tk in unigram_all_tokens:
    unigram_target[tk] += 1
unigram_target = [key for key, value in sorted(unigram_target.items(), key=lambda kv: kv[1], reverse=True)[:50]]

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
result.to_csv("Reddit_BoW_Intel_Result.csv")