#!/usr/local/bin/python3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.util import ngrams
import collections
import operator
import numpy as np
import pandas as pd
import re

#stopWords
stopWords = set(stopwords.words('english'))
#----------------------------------------------------------------------------------------------------#
def tokens(doc):
    doc = "".join(re.sub(r'[^\w\s]','',doc.lower()))
    doc = ''.join(re.sub('[0-9]','',doc))
    tokens = word_tokenize(doc)
    #print(tokens)
    return tokens

#----------------------------------------------------------------------------------------------------#
def extractKeys(train, test):

    print("extract keyphrases initialized..\n")

    vec = TfidfVectorizer(use_idf=True, ngram_range=(1,3), stop_words="english", tokenizer=tokens)

    joined = train + test

    trainvec = vec.fit(joined)
    testvec = vec.transform(test)

    print("obtaining tfidf score array..\n")

    # sum tfidf frequency of each term through documents
    sums = testvec.sum(axis=0)

    feature_names = np.asarray(vec.get_feature_names())

    print(len(feature_names))

    print("Printing feature_names values: \n")

    #Scoring algorithm: score = tfidf score * (number of characters or number of words)
    print("Applying scoring algorithm.. \n")
    # connecting term to its [(sums frequency) * (number of words)]
    d = {}


    for col, term in enumerate(feature_names):
    	words = term.split()
    	n_words = len(words)
    	res = sums[0,col] * n_words
    	d[term] = res


    #sorted algorithm
    print("Sorting the dictionary...\n")
    top5 = dict(sorted(d.items(), key=operator.itemgetter(1), reverse=True)[:5])

    #return best of 5
    return top5

#----------------------------------------------------------------------------------------------------#
#Train collection
train = fetch_20newsgroups(subset='train')
#usamos um excerto do subset "train" do 20newsgroups
train_data = train.data[:100]
#Document read
document = open('input.txt')
doc = [document.read()]
#run script
best_of = extractKeys(train_data, doc)
print("Printing results.. \n")
#print results
print(best_of)
