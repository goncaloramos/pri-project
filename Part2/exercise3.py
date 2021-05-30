import json
from glob import glob
from math import log
from xml.dom.minidom import parse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import *
import re
import operator
import networkx as nx
from datetime import datetime
from scipy.stats import rankdata
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler

startTime = datetime.now()

filesTestList = glob('..\\dataset\\500N-KPCrowd\\test\\*.xml')
filesTrainList = glob('..\\dataset\\500N-KPCrowd\\train\\*.xml')

#stopWords
stopWords = set(stopwords.words('english'))

#----------------------------------------------------------------------------------------------------#
def parseXml(filesList):
    docsList = []

    for file in filesList:
        sent = ""
        dom = parse(file)

        sentences = dom.getElementsByTagName("sentence")
        for sentence in sentences:
            tokens = sentence.getElementsByTagName("token")
            for token in tokens:
                word = token.getElementsByTagName("word")[0]
                if word.firstChild.data not in stopWords:
                    sent += word.firstChild.data + " "

        #tratar a pontuacao
        sent_final = sent.replace(" .", ".")
        sent_final = sent_final.replace(" ,", ",")
        sent_final = "".join(re.sub(r'[^\w\s]', '', sent_final.lower()))
        sent_final = ''.join(re.sub('[0-9]', '', sent_final))

        #adicionar a lista
        docsList.append(sent_final)

    return docsList



def parseTestNames(filesList):
	fileNames = [s.replace('..\\dataset\\500N-KPCrowd\\test\\', '') for s in filesList]
	fileNames = [s.replace('.xml', '') for s in fileNames]
	return fileNames

def parseTrainNames(filesList):
	fileNames = [s.replace('..\\dataset\\500N-KPCrowd\\train\\', '') for s in filesList]
	fileNames = [s.replace('.xml', '') for s in fileNames]
	return fileNames


# -----------------------------------------------------------------------------------------------
# Creates test features

def create_test_data(docsList, vectorizer, docvec):
    fileNames = parseTestNames(filesTestList)
    documents = {}
    termkeys = {}
    i = 0
    ficheiro = 0
    doc_len = 0

    with open('../dataset/500N-KPCrowd/references/test.reader.json') as json_file:
        data = json.load(json_file)

    for file in fileNames:
        candidates = {}

        # cria um dicionario em que associa a cada feature name o seu idf score
        dictionary = dict(zip(vectorizer.get_feature_names(), docvec.toarray()[ficheiro]))
        terms = [key for key, value in dictionary.items() if value != 0.0]
        len(terms)

        graph = checkPairs(docsList[i], terms)
        pn = getPageRank(graph)

        # avgdl
        doc_len += len(docsList[i])
        avgdl = doc_len / len(docsList)

        expertKeys = [item for sublist in data[file] for item in sublist]

        for term in terms:
            features = []
            # tf
            features.append(docsList[i].count(term) / len(docsList[i]))
            tf_score = features[0]

            # idf
            nt = sum(1 for doc in docsList if term in doc)
            features.append(idf_calc(len(docsList), nt))
            idf_score = features[1]

            # tf-idf
            features.append(tf_score * idf_score)
            tf_idf_score = features[2]

            # bm25_score
            features.append(bm25_calc(idf_score, tf_score, len(docsList[i]), avgdl))
            bm25_score = features[3]

            #candidate word count
            features.append(candidateWordCount(term))
            wordCount = features[4]

            # page rank
            features.append(pn[term])
            page_rank = features[5]

            candidates[term] = features

        documents[file] = candidates
        termkeys[file] = expertKeys

    return documents, termkeys

# -----------------------------------------------------------------------------------------------
# Creates train features

def create_train_data(docsList, vectorizer, docvec):
    fileNames = parseTrainNames(filesTrainList)
    documents = []
    i = 0
    ficheiro = 0
    doc_len = 0

    with open('../dataset/500N-KPCrowd/references/train.reader.json') as json_file:
        data = json.load(json_file)

    for file in fileNames:
        candidates = []

        # cria um dicionario em que associa a cada feature name o seu idf score
        dictionary = dict(zip(vectorizer.get_feature_names(), docvec.toarray()[ficheiro]))
        terms = [key for key, value in dictionary.items() if value != 0.0]
        len(terms)

        # avgdl
        doc_len += len(docsList[i])
        avgdl = doc_len / len(docsList)

        graph = checkPairs(docsList[i], terms)
        pn = getPageRank(graph)

        expertKeys = [item for sublist in data[file] for item in sublist]

        for term in terms:
            features = []
            # tf
            features.append(docsList[i].count(term) / len(docsList[i]))
            tf_score = features[0]

            # idf
            nt = sum(1 for doc in docsList if term in doc)
            features.append(idf_calc(len(docsList), nt))
            idf_score = features[1]

            # tf-idf
            features.append(tf_score * idf_score)
            tf_idf_score = features[2]

            # bm25_score
            features.append(bm25_calc(idf_score, tf_score, len(docsList[i]), avgdl))
            bm25_score = features[3]

            # candidate word count
            features.append(candidateWordCount(term))
            wordCount = features[4]

            # page rank
            features.append(pn[term])
            page_rank = features[5]

            # keyphrase ou nao
            if term in expertKeys:
                features.append(1)
            else:
                features.append(0)

            candidates.append(features)

        documents.append(candidates)

    return documents

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Ranks features
# rankings[n_feature][candidate] = feature_rank
# Devolve um lista de dicionários onde o primeiro indíce corresponde a uma feature,
# a segunda chave a um candidato e contem o rank dessa feature do candidato relativamente aos outros candidatos

def rank_features(docs, n_features):

    doc_rankings = {}

    for candidates in docs.keys():
        
        rankings = []
        i = 0
        while i < n_features:

            # Primeiro, reunir os valores de todos os candidatos referentes a uma feature
            pre_rank = {}
            for candidate in docs[candidates].keys():
                pre_rank[candidate] = docs[candidates][candidate][i]

            # Segundo, fazemos o ranking, se um ou mais candidatos tiverem o mesmo valor, eles vao obter o mesmo rank
            rank_dict = dict(zip(pre_rank.keys(), rankdata([-j for j in pre_rank.values()], method='min')))
            rankings.append(rank_dict)
            i += 1

        doc_rankings[candidates] = rankings

    return doc_rankings

# -----------------------------------------------------------------------------------------------
# Calculates Reciprocal Rank Fusion approach

def RRFScore(docs, rankings, l_expertKeys):

    doc_top_keys = {}
    dic_avg_precisions = {}

    for candidates in docs.keys():

        expertKeys = l_expertKeys[candidates]

        rrf_score = {}

        for candidate in docs[candidates].keys():
            rrf_candidate = 0

            for feature in rankings[candidates]:
                rrf_candidate += 1/(50 + feature[candidate])

            rrf_score[candidate] = rrf_candidate

        top_keys = dict(sorted(rrf_score.items(), key=operator.itemgetter(1), reverse=True)[:5])

        doc_top_keys[candidates] = top_keys

        print(top_keys)

        precisions_at_k = calc_precision_at_k(expertKeys, list(top_keys.keys()))

        avg_precisions = calc_average_precision(precisions_at_k, expertKeys, list(top_keys.keys()))

        dic_avg_precisions[candidates] = avg_precisions

    mean_avg_precision = calc_mean_avg_precision(dic_avg_precisions)

    print(mean_avg_precision)

#----------------------------------------------------------------------------------------------------#
#Perceptron Classifiers

# Treina o classificador e testamos a precisao com a mean average precision

def train_classifier(train_features, docs_features, l_expertKeys):

    dic_avg_precisions = {}

    for doc_features in docs_features.keys():

        expertKeys = l_expertKeys[doc_features]

        test_features = docs_features[doc_features]

        # Primeiro, tratamos dos dados para que possam ser processados pelo perceptron
        test_candidates = list(test_features.keys())
        candidates_features = list(test_features.values())

        X_train, y_train = [], []
        for docs in train_features:
            for data in docs:
                X_train.append(data[:-1])
                y_train.append(data[-1])


        # Train the scaler, which standarizes all the features to have mean=0 and unit variance
        sc = StandardScaler()
        sc.fit(X_train)

        # Apply the scaler to the X training data
        X_train_std = sc.transform(X_train)

        # Apply the SAME scaler to the X test data
        X_test_std = sc.transform(candidates_features)

        # Create a perceptron object with the parameters that increase its accuracy
        ppn = Perceptron()

        # Train the perceptron
        ppn.fit(X_train_std, y_train)

        # Apply the trained perceptron on the X data to make predicts for the y test data
        y_pred = ppn.decision_function(X_test_std)

        pred_candidates = {}

        for i in range(0, len(test_candidates)):
            pred_candidates[test_candidates[i]] = y_pred[i]

        top_keys = dict(sorted(pred_candidates.items(), key=operator.itemgetter(1), reverse=True)[:5])

        print(top_keys)

        precisions_at_k = calc_precision_at_k(expertKeys, list(top_keys.keys()))

        avg_precisions = calc_average_precision(precisions_at_k, expertKeys, list(top_keys.keys()))

        dic_avg_precisions[doc_features] = avg_precisions

    mean_avg_precision = calc_mean_avg_precision(dic_avg_precisions)

    print(mean_avg_precision)


#----------------------------------------------------------------------------------------------------#
#Features Calculation

#candidate word count
def candidateWordCount(candidate):
	count = len(re.findall(r'\w+', candidate))
	return count

#----------------------------------------------------------------------------------------------------#
#IDF formula
#N = total number of docs in a background collection
#nt = number of docs, containing the term t
def idf_calc(N, nt):
    idf_value = log((N-nt+0.5)/(nt+0.5))
    return idf_value

#----------------------------------------------------------------------------------------------------#
#BM25 formula
#tf
# doc length
#avg doc length in a background text collection
def bm25_calc(idf, tf, d, avgdl):
    k=1.2
    b=0.75
    second_half = (tf * (k+1))/(tf + k * (1 - b + b * (d/avgdl)))
    final_bm25 = idf * second_half
    return final_bm25

#----------------------------------------------------------------------------------------------------#

def tokens(doc):
    sentences = sent_tokenize(doc)
    return sentences

def findWholeWord(term, sentence):
    return re.compile(r'\b({0})\b'.format(term), flags=re.IGNORECASE).search(sentence)

def checkPairs(doc, terms):
    terms_Sent = []
    sentences = tokens(doc.lower())
    for sent in sentences:
        termos = []
        for term in terms:
            if findWholeWord(term, sent):
                termos.append(term)

        terms_Sent.append(termos)

    graph = {}
    for sent in terms_Sent:
        for term in sent:
            sente = [x for x in sent if x != term]
            if term in graph:
                graph[term] += sente
            else:
                graph[term] = sente

    # adding to the graph the terms that had no links because they are also candidates
    for term in terms:
        if term not in graph:
            graph[term] = []

    return graph

def getPageRank(graph):
    g = nx.DiGraph(graph) #direct graph
    pr = nx.pagerank(g, alpha=0.15, weight=None, max_iter=50)

    return pr


#----------------------------------------------------------------------------------------------------#
#Metrics
def calc_precision_at_k(expertKeys, predictedKeys):
    precision_at_k = {}
    tp = 0
    k = 0
    for i in range(0, len(expertKeys)):
        k += 1
        if i <= len(predictedKeys) - 1:
            if predictedKeys[i] in expertKeys:
                tp += 1

            precision_at_k['precision@' + str(k)] = tp / k
        else:
            break

    return precision_at_k


def calc_average_precision(precisions_at_k, expectedKeys, predictedKeys):
    precisions = [precision for precision in precisions_at_k.values()]
    somatorio = 0
    ri = 1

    for i in range(0, len(predictedKeys)):
        if predictedKeys[i] in expectedKeys:
            somatorio += precisions[i] * ri

    avg_prec = somatorio / len(expectedKeys)
    return avg_prec

def calc_mean_avg_precision(mean_avg_precisions):
	mean_avg_precision = sum(mean_avg_precisions.values()) / len(mean_avg_precisions)
	return mean_avg_precision

#----------------------------------------------------------------------------------------------------#
#Execution
testList = parseXml(filesTestList)
vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1, 3))
docvec = vectorizer.fit_transform(testList)


doc_test_features, test_keys = create_test_data(testList, vectorizer, docvec)
ranking = rank_features(doc_test_features, 6)
print(ranking)
RRFScore(doc_test_features, ranking, test_keys)

trainList = parseXml(filesTrainList)
docvec2 = vectorizer.fit_transform(trainList)
doc_train_features = create_train_data(trainList, vectorizer, docvec2)
train_classifier(doc_train_features, doc_test_features, test_keys)

print(datetime.now() - startTime)
