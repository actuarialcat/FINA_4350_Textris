import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from collections import Counter
import re

data = pd.read_csv("reddit_intel.csv")
stop_words = set(stopwords.words("english"))
wnl = WordNetLemmatizer()

unigram = []
bigram = []
trigram = []
for i in data.index:
    sub_text = re.sub("(?P<url>http[^\s]+)", "", str(data["sub_text"][i]))
    comment_text = re.sub("(?P<url>http[^\s]+)", "", str(data["comm_text"][i]))

    sub_text_tokens = [w for w in word_tokenize(sub_text.lower()) if w.isalpha() and not w.startswith("http")]
    sub_text_filtered_tokens = [w for w in sub_text_tokens if not w in stop_words]
    sub_text_lemmatized_tokens = [wnl.lemmatize(w) for w in sub_text_filtered_tokens]

    comment_text_tokens = [w for w in word_tokenize(comment_text.lower()) if w.isalpha()]
    comment_text_filtered_tokens = [w for w in comment_text_tokens if not w in stop_words]
    comment_text_lemmatized_tokens = [wnl.lemmatize(w) for w in comment_text_filtered_tokens]

    unigram += sub_text_lemmatized_tokens + comment_text_lemmatized_tokens
    bigram += [' '.join(ng) for ng in ngrams(sub_text_filtered_tokens, 2)] + [' '.join(ng) for ng in ngrams(comment_text_filtered_tokens, 2)]
    trigram += [' '.join(ng) for ng in ngrams(sub_text_filtered_tokens, 3)] + [' '.join(ng) for ng in ngrams(comment_text_filtered_tokens, 3)]

bow_uni = Counter(unigram)
bow_bi = Counter(bigram)
bow_tri = Counter(trigram)

f = open("BoW_Unigram for Reddit_Intel.txt", "a")
for word, count in sorted(list(bow_uni.items()), key=lambda tup: tup[1], reverse=True):
    if count > 10:
        f.write("{0} {1}\n".format(word, str(count)))
f.close()

f = open("BoW_Bigram for Reddit_Intel.txt", "a")
for word, count in sorted(list(bow_bi.items()), key=lambda tup: tup[1], reverse=True):
    if count > 10:
        f.write("{0} {1}\n".format(word, str(count)))
f.close()

f = open("BoW_Trigram for Reddit_Intel.txt", "a")
for word, count in sorted(list(bow_tri.items()), key=lambda tup: tup[1], reverse=True):
    if count > 10:
        f.write("{0} {1}\n".format(word, str(count)))
f.close()