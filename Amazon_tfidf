import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from collections import defaultdict
import re
import datetime
import os
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel

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
        for i in data.index:
            title = str(data["title"][i]).strip()
            content = str(data["content"][i]).strip()
            year = str(data["date"][i]).split(" ")[-1]
            month = str(data["date"][i]).split(" ")[-3]
            
            index = calKey(year, month)
            unigram_first_month = ''

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
            unigram_first_month = title_lemmatized_tokens + content_lemmatized_tokens

            if index in unigram:
                unigram[index].append(unigram_first_month)
            else:
                unigram[index] = unigram_first_month
            '''
            if index in unigram:
                #unigram[index] += title_lemmatized_tokens + content_lemmatized_tokens
                bigram[index] += title_bigram_token + content_bigram_token
                trigram[index] += title_trigram_token + content_trigram_token
                unigram_count[index] += len(title_lemmatized_tokens + content_lemmatized_tokens)
                bigram_count[index] += len(title_bigram_token + content_bigram_token)
                trigram_count[index] += len(title_trigram_token + content_trigram_token)
            else:
                #unigram[index] = title_lemmatized_tokens + content_lemmatized_tokens
                bigram[index] = title_bigram_token + content_bigram_token
                trigram[index] = title_trigram_token + content_trigram_token
                unigram_count[index] = len(title_lemmatized_tokens + content_lemmatized_tokens)
                bigram_count[index] = len(title_bigram_token + content_bigram_token)
                trigram_count[index] = len(title_trigram_token + content_trigram_token)
                '''
    print(unigram['201102'])
    if brand == 'Intel':
        tokenized_corpus = [word_tokenize( str(unigram['201102']))]
        #tokenized_corpus = [word_tokenize(doc.lower()) for doc in unigram['201102']]
        d = Dictionary(tokenized_corpus)
        bowcorpus = [d.doc2bow(doc) for doc in tokenized_corpus]
        print(bowcorpus)

        tfidf = TfidfModel(bowcorpus)
        tfidf_weights = tfidf[bowcorpus[0]] 
        tfidf_weights[:5]                   
        sorted_tfidf_weights = \
            sorted(
                tfidf_weights,
                key=lambda x: x[1],
                reverse=True)
        for term_id, weight in sorted_tfidf_weights[:5]:
            print(d.get(term_id), weight)
