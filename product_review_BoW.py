import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from collections import Counter
import re
import os

directory = "Product review data/"
review_files = os.listdir(directory)

stop_words = set(stopwords.words("english"))
wnl = WordNetLemmatizer()

for review_file in review_files:
    data = pd.read_csv(directory + review_file)
    print(review_file)
    unigram = []
    bigram = []
    trigram = []
    for i in data.index:
        title = str(data["title"][i]).strip()
        content = str(data["content"][i]).strip()
        title_tokens = [w for w in word_tokenize(title.lower()) if w.isalpha()]
        title_filtered_tokens = [w for w in title_tokens if not w in stop_words]
        title_lemmatized_tokens = [wnl.lemmatize(w) for w in title_filtered_tokens]

        content_tokens = [w for w in word_tokenize(content.lower()) if w.isalpha()]
        content_filtered_tokens = [w for w in title_tokens if not w in stop_words]
        content_lemmatized_tokens = [wnl.lemmatize(w) for w in content_filtered_tokens]
        
        unigram += title_lemmatized_tokens + content_lemmatized_tokens
        bigram += [' '.join(ng) for ng in ngrams(title_filtered_tokens, 2)] + [' '.join(ng) for ng in ngrams(content_filtered_tokens, 2)]
        trigram += [' '.join(ng) for ng in ngrams(title_filtered_tokens, 3)] + [' '.join(ng) for ng in ngrams(content_filtered_tokens, 3)]

    bow_uni = Counter(unigram)
    bow_bi = Counter(bigram)
    bow_tri = Counter(trigram)

    product = re.search('for (.+?).csv', review_file).group(1)
    f = open("BoW for {0}.txt".format(product), "a")
    f.write("Unigram:\n")
    for word, count in sorted(list(bow_uni.items()), key=lambda tup: tup[1], reverse=True):
        if count > 10:
            f.write("{0} {1}\n".format(word, str(count)))

    f.write("\nBigram:\n")
    for word, count in sorted(list(bow_bi.items()), key=lambda tup: tup[1], reverse=True):
        if count > 2:
            f.write("{0} {1}\n".format(word, str(count)))

    f.write("\nTrigram:\n")  
    for word, count in sorted(list(bow_tri.items()), key=lambda tup: tup[1], reverse=True):
        if count > 2:
            f.write("{0} {1}\n".format(word, str(count)))
