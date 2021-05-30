#!/usr/local/bin/python3
import itertools
import json
import string
from xml.dom.minidom import parse
from collections import Counter
from nltk.tokenize import *
import io
from nltk.util import ngrams
import operator
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from math import *
from glob import glob
from nltk.corpus import stopwords
from datetime import datetime
startTime = datetime.now()

filesList = glob('..\\dataset\\500N-KPCrowd\\test\\*.xml')
stopWords = set(stopwords.words('english'))
vecFile = "wiki-news-300d-1M.vec"


# ----------------------------------------------------------------------------------------------------#
def parseXml():
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

        # tratar a pontuacao
        sent_final = sent.replace(" .", ".")
        sent_final = sent_final.replace(" ,", ",")
        #sent_final = "".join(re.sub(r'[^\w\s]', '', sent_final.lower()))
        sent_final = ''.join(re.sub('[0-9]', '', sent_final))

        # adicionar a lista
        docsList.append(sent_final)

    return docsList

#----------------------------------------------------------------------------------------------------#
def getKeyphrases(docsList, terms, docvec, iterations):
    avg_precisions_tf_co_occurr = {}
    avg_precisions_bm_co_occurr = {}
    avg_precisions_len_pos_co_occurr = {}
    avg_precisions_WCount_pos_co_occurr = {}
    avg_precisions_tf_sim = {}
    avg_precisions_bm_sim = {}
    avg_precisions_len_pos_sim = {}
    avg_precisions_WCount_pos_sim = {}

    with open('../dataset/500N-KPCrowd/references/test.reader.json') as json_file:
        data = json.load(json_file)

    docNumber = 0
    doc_len = 0
    fileNames = parseFileNames(filesList)

    averages = load_vectors(vecFile, terms)
    weights_sim = calc_weights_sim(averages, terms)

    for file in fileNames:
        # avgdl
        doc_len += len(docsList[docNumber])
        avgdl = doc_len / len(docsList)

        expertKeys = [item for sublist in data[file] for item in sublist]

        #get priors (in this case by tf-idf scores)
        priors_tf = calc_priors_tf_idf(terms, docvec, docNumber)

        # #get priors (in this case by bm25 scores)
        priors_bm = calc_priors_bm25(terms, docsList, docNumber, avgdl)

        #get priors (based on combo between length and position of the candidate)
        priors_len_pos = calc_priors_len_pos(docsList[docNumber], terms)

        # #get priors (based on combo between word count and position of the candidate)
        priors_WCount_pos = calc_priors_wCount_pos(docsList[docNumber], terms)

        weights = calc_weights_cooccurr(docsList[docNumber], terms)

        predicted_tf_co_occurr = calc_page_rank(terms, weights, priors_tf, iterations)
        predicted_bm_co_occurr = calc_page_rank(terms, weights, priors_bm, iterations)
        predicted_len_pos_co_occurr = calc_page_rank(terms, weights, priors_len_pos, iterations)
        predicted_WCount_pos_co_occurr = calc_page_rank(terms, weights, priors_WCount_pos, iterations)

        predicted_tf_sim = calc_page_rank(terms, weights_sim, priors_tf, iterations)
        predicted_bm_sim = calc_page_rank(terms, weights_sim, priors_bm, iterations)
        predicted_len_pos_sim = calc_page_rank(terms, weights_sim, priors_len_pos, iterations)
        predicted_WCount_pos_sim = calc_page_rank(terms, weights_sim, priors_WCount_pos, iterations)

        predictedKeys_tf_co_occurr = [item for item in predicted_tf_co_occurr.keys()]
        predictedKeys_bm_co_occurr = [item for item in predicted_bm_co_occurr.keys()]
        predictedKeys_len_pos_co_occurr = [item for item in predicted_len_pos_co_occurr.keys()]
        predictedKeys_WCount_pos_co_occurr = [item for item in predicted_WCount_pos_co_occurr.keys()]

        predictedKeys_tf_sim = [item for item in predicted_tf_sim.keys()]
        predictedKeys_bm_sim = [item for item in predicted_bm_sim.keys()]
        predictedKeys_len_pos_sim = [item for item in predicted_len_pos_sim.keys()]
        predictedKeys_WCount_pos_sim = [item for item in predicted_WCount_pos_sim.keys()]


        # # calcula precision@k para cada documento (nao identifica o documento)
        precisions_at_k_tf_co_occurr = calc_precision_at_k(expertKeys, predictedKeys_tf_co_occurr)
        precisions_at_k_bm_co_occurr = calc_precision_at_k(expertKeys, predictedKeys_bm_co_occurr)
        precisions_at_k_len_pos_co_occurr = calc_precision_at_k(expertKeys, predictedKeys_len_pos_co_occurr)
        precisions_at_k_WCount_pos_co_occurr = calc_precision_at_k(expertKeys, predictedKeys_WCount_pos_co_occurr)

        precisions_at_k_tf_sim = calc_precision_at_k(expertKeys, predictedKeys_tf_sim)
        precisions_at_k_bm_sim = calc_precision_at_k(expertKeys, predictedKeys_bm_sim)
        precisions_at_k_len_pos_sim = calc_precision_at_k(expertKeys, predictedKeys_len_pos_sim)
        precisions_at_k_WCount_pos_sim = calc_precision_at_k(expertKeys, predictedKeys_WCount_pos_sim)

        # # calcula a average precision para cada documento

        avg_precisions_tf_co_occurr[file] = calc_average_precision(precisions_at_k_tf_co_occurr, expertKeys, predictedKeys_tf_co_occurr)
        avg_precisions_bm_co_occurr[file] = calc_average_precision(precisions_at_k_bm_co_occurr, expertKeys, predictedKeys_bm_co_occurr)
        avg_precisions_len_pos_co_occurr[file] = calc_average_precision(precisions_at_k_len_pos_co_occurr, expertKeys, predictedKeys_len_pos_co_occurr)
        avg_precisions_WCount_pos_co_occurr[file] = calc_average_precision(precisions_at_k_WCount_pos_co_occurr, expertKeys, predictedKeys_WCount_pos_co_occurr)

        avg_precisions_tf_sim[file] = calc_average_precision(precisions_at_k_tf_sim, expertKeys, predictedKeys_tf_sim)
        avg_precisions_bm_sim[file] = calc_average_precision(precisions_at_k_bm_sim, expertKeys, predictedKeys_bm_sim)
        avg_precisions_len_pos_sim[file] = calc_average_precision(precisions_at_k_len_pos_sim, expertKeys, predictedKeys_len_pos_sim)
        avg_precisions_WCount_pos_sim[file] = calc_average_precision(precisions_at_k_WCount_pos_sim, expertKeys, predictedKeys_WCount_pos_sim)

        print("\n")
        docNumber += 1

    mean_avg_precision_tf_co_occurr = calc_mean_avg_precision(avg_precisions_tf_co_occurr)
    mean_avg_precision_bm_co_occurr = calc_mean_avg_precision(avg_precisions_bm_co_occurr)
    mean_avg_precision_len_pos_co_occurr = calc_mean_avg_precision(avg_precisions_len_pos_co_occurr)
    mean_avg_precision_WCount_pos_co_occurr = calc_mean_avg_precision(avg_precisions_WCount_pos_co_occurr)

    mean_avg_precision_tf_sim = calc_mean_avg_precision(avg_precisions_tf_sim)
    mean_avg_precision_bm_sim = calc_mean_avg_precision(avg_precisions_bm_sim)
    mean_avg_precision_len_pos_sim = calc_mean_avg_precision(avg_precisions_len_pos_sim)
    mean_avg_precision_WCount_pos_sim = calc_mean_avg_precision(avg_precisions_WCount_pos_sim)

    print("\n")
    print("----------------------")
    print("Mean Average Precision using tf-idf and co_occurr: ", mean_avg_precision_tf_co_occurr)
    print("Mean Average Precision using bm25 and co-occur: ", mean_avg_precision_bm_co_occurr)
    print("Mean Average Precision using len_pos and co-occur ", mean_avg_precision_len_pos_co_occurr)
    print("Mean Average Precision using WCount_pos and co-occurr: ", mean_avg_precision_WCount_pos_co_occurr)
    print("Mean Average Precision using tf-idf and sim: ", mean_avg_precision_tf_sim)
    print("Mean Average Precision using bm25 and sim: ", mean_avg_precision_bm_sim)
    print("Mean Average Precision using len_pos and sim: ", mean_avg_precision_len_pos_sim)
    print("Mean Average Precision using WCount_pos and sim: ", mean_avg_precision_WCount_pos_sim)



def tokens(doc):
    sentences = sent_tokenize(doc)
    return sentences


def parseFileNames(filesList):
    fileNames = [s.replace('..\\dataset\\500N-KPCrowd\\test\\', '') for s in filesList]
    fileNames = [s.replace('.xml', '') for s in fileNames]
    return fileNames


def findWholeWord(term, sentence):
    return re.compile(r'\b({0})\b'.format(term), flags=re.IGNORECASE).search(sentence)


def getGraph(terms_Sent):
    graph = {}
    for sent in terms_Sent:
        for term in sent:
            sente = [x for x in sent if x != term]
            if term in graph:
                graph[term] += sente
            else:
                graph[term] = sente

    return graph


def checkPairs(doc, terms):
    terms_Sent = []
    sentences = tokens(doc.lower())

    for sent in sentences:
        termos = []
        for term in terms:
            if findWholeWord(term, sent):
                termos.append(term)

        terms_Sent.append(termos)

    return terms_Sent


def idf_calc(N, nt):
    idf_value = log((N - nt + 0.5) / (nt + 0.5))
    return idf_value


def bm25_calc(idf, tf, d, avgdl):
    k = 1.2
    b = 0.75
    second_half = (tf * (k + 1)) / (tf + k * (1 - b + b * (d / avgdl)))
    final_bm25 = idf * second_half
    return final_bm25


def getBm25(docsList):
    i = 0
    doc_len = 0
    fileNames = parseFileNames(filesList)

    for file in fileNames:
        idf_scores = {}
        tf_scores = {}
        bm25_dict = {}

        # avgdl
        doc_len += len(docsList[i])
        avgdl = doc_len / len(docsList)

        for f in file:
            nt = sum(1 for doc in docsList if f in doc)
            idf_scores[f] = idf_calc(len(docsList), nt)
            tf_scores[f] = docsList[i].count(f) / len(docsList[i])
            bm25_dict[f] = bm25_calc(idf_scores[f], tf_scores[f], len(docsList[i]), avgdl)


# ----------------------------------------------------------------------------------------------------#
# Prior
# can be calculated based on the length or position of the candidate in the document,
# under the intuition that longer candidates in the first sentences are more likely to be good keyphrases

# can be calculated based on TF-IDF or BM25 scores for the candidates (i.e., computing TF-IDF or BM25 scores,
# similarly to what was done on the first part of the project)

def calc_priors_tf_idf(terms, docvec, docNumber):
    # obtain tf_idf_scores
    return dict(zip(terms, docvec.toarray()[docNumber]))


def calc_priors_bm25(terms, docsList, docNumber, avgdl):
    idf_scores = {}
    tf_scores = {}
    bm25_scores = {}

    for term in terms:
        nt = sum(1 for doc in docsList if term in doc)
        idf_scores[term] = idf_calc(len(docsList), nt)
        tf_scores[term] = docsList[docNumber].count(term) / len(docsList[docNumber])
        bm25_scores[term] = bm25_calc(idf_scores[term], tf_scores[term], len(docsList[docNumber]), avgdl)

    return bm25_scores

#length of each term
def calc_priors_length(terms):
    priors = {}
    max_len = len(max(terms, key=len))
    for term in terms:
        priors[term] = len(term) / max_len

    return priors

#word count of each term
def calc_priors_wCount(terms):
    priors = {}
    for term in terms:
        priors[term] = len(re.findall(r'\w+', term))

    return priors

#document position for each term (1st sentence, 2nd sentence, etc)
def calc_priors_position(doc, terms):
    priors = {}
    terms_Sent = checkPairs(doc, terms)
    weight = len(terms_Sent)

    for sent in terms_Sent:
        for term in terms:
            if term in sent:
                priors[term] = weight
        weight -= 1

    for term in terms:
        if term not in priors:
            priors[term] = 0

    return priors


def calc_priors_len_pos(doc, terms):
    priors = {}
    priors_norm = {}
    i = 0
    positions = calc_priors_position(doc, terms)

    lengths = calc_priors_length(terms)

    for term in terms:
        priors[term] = positions[term] * lengths[term]

    input_array = np.array([prior for prior in priors.values()])

    if np.isnan(np.min(input_array)):
        input_array = np.array([0 for prior in priors])
        result_array = input_array
    else:
        result_array = (input_array - np.min(input_array)) / np.ptp(input_array)

    for term in terms:
        priors_norm[term] = result_array[i]
        i += 1

    return priors_norm


def calc_priors_wCount_pos(doc, terms):
    priors = {}
    priors_norm = {}
    i = 0
    word_count = calc_priors_wCount(terms)
    positions = calc_priors_position(doc, terms)

    for term in terms:
        priors[term] = positions[term] * word_count[term]

    input_array = np.array([prior for prior in priors.values()])

    if np.isnan(np.min(input_array)):
        input_array = np.array([0 for prior in priors])
        result_array = input_array
    else:
        result_array = (input_array - np.min(input_array)) / np.ptp(input_array)

    for term in terms:
        priors_norm[term] = result_array[i]
        i += 1

    return priors_norm


def drawGraph2(graph):
    g = nx.DiGraph(graph) #direct graph
    nx.draw(g, with_labels=True)
    plt.draw()
    plt.show()


# ----------------------------------------------------------------------------------------------------#
# Weight Functions
# can be calculated based on the number of co-ocurrences involving the candidates, computed over a background collection

def checkPairs2(doc, terms):
    terms_Sent = []
    sentences = tokens(doc.lower())

    # get the terms that appear in each document
    for sent in sentences:
        termos = []
        for term in terms:
            if findWholeWord(term, sent):
                termos.append(term)

        terms_Sent.append(termos)

    return terms_Sent


def getGraph2(terms_Sent, terms):
    graph = {}
    # creates a dictionary wwith terms as keys and a list of terms as values, corresponding to the link to each key
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

def load_vectors(fname, terms):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    i=0
    data = {}
    averages = {}
    terms_final = []

    for term in terms:
        if len(re.findall(r'\w+', term)) == 1:
            terms_final.append(term)

    for line in fin:
        tokens = line.rstrip().split(' ')
        if tokens[0].lower() in terms_final:
            data[tokens[0].lower()] = map(float, tokens[1:])
            info = list(data[tokens[0].lower()])
            averages[tokens[0].lower()] = sum(info) / len(info)

        if i % 100000 == 0 and i != 0:
            print("lines: ", i)

        i += 1

    return averages


# returns an dictionary of dictionaries where each key corresponds to a candidate and each value corresponds
# to a dictionary that contains the number of co-occurrences for each term it is linked to
def calc_weights_cooccurr(doc, terms):
    occurrences = {}
    terms_Sent = checkPairs2(doc, terms)
    graph = getGraph2(terms_Sent, terms)

    for term in terms:
        occurrences[term] = dict(Counter(graph[term]))

    return occurrences


def calc_weights_sim(averages, terms):

    weights = {}

    for term in terms:
        weights[term] = {}
        for word in averages:
            if findWholeWord(word, term):
                weights[term][word] = averages[word]

    return weights


# ---------------------------------------------------------------------------------------------
# Page Rank
# parcels separated by the sum operator

def calc_page_rank(terms, weights, priors, iterations):
    # first parcel
    fparcel = {}
    # second parcel
    sparcel = {}

    # second parcel final value
    sparcel_final = 0

    # temporary page rank
    page_rank_temp = {}

    # final page rank
    page_rank = {}

    # sum of priors (used in first parcel)
    sum_priors = sum(priors.values())

    # dumping factor
    d = 0.15

    # define initial page ranks
    for term in priors.keys():
        page_rank_temp[term] = 1 / len(priors)

    for i in range(1, iterations + 1):
        # term corresponds to pi
        for term in terms:
            # key corresponds to pj
            for key in weights[term]:
                # weights[term][key] corresponds to the weight between pi and pj, pj being the terms who link to pi - Weight(pi,pj)
                sparcel1 = page_rank_temp[key] * weights[term][key]
                # sparcel2 corresponds to the sum of the weights between pj and pk, pk being the terms who link to pj - Weight(pj,pk)
                sparcel2 = sum(weights[key].values())
                # second parcel of the formula
                sparcel_final += sparcel1 / sparcel2

            # first parcel of the formula
            fparcel[term] = d * (priors[term] / sum_priors)
            sparcel[term] = (1 - d) * sparcel_final

            # page_rank by the formula
            page_rank[term] = fparcel[term] + sparcel[term]
            sparcel_final = 0

        page_rank_temp = page_rank

    page_rank = dict(sorted(page_rank.items(), key=operator.itemgetter(1), reverse=True)[:5])

    return page_rank

# ---------------------------------------------------------------------------------------------
# Metrics
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


docsList = parseXml()
vec = TfidfVectorizer(use_idf=True, ngram_range=(1, 3))
docvec = vec.fit_transform(docsList)
terms = vec.get_feature_names()
getKeyphrases(docsList, terms, docvec, 1)
print(datetime.now() - startTime)



