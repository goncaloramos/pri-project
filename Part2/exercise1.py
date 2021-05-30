#!/usr/local/bin/python3
from nltk.tokenize import *
import operator
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import networkx as nx

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

    print(terms_Sent)

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

    print("graph: ")
    print(graph)
    return graph


def drawGraph(graph):

    g = nx.DiGraph() #direct graph
    g.add_nodes_from(graph.keys())
    for k,v in graph.items():
        g.add_edges_from(([(k,t) for t in v]))

    nx.draw(g,with_labels=True)
    plt.draw()
    plt.show()

def drawGraph2(graph):
    g = nx.DiGraph(graph) #direct graph
    nx.draw(g, with_labels=True)
    plt.draw()
    plt.show()

def getPageRank(graph):
    g = nx.DiGraph(graph) #direct graph
    pr = nx.pagerank(g, alpha=0.15, weight=None, max_iter=50)
    print("pagerank: ")
    print(pr)
    best_keys = dict(sorted(pr.items(), key=operator.itemgetter(1), reverse=True)[:5])
    return best_keys



vec = TfidfVectorizer(use_idf=True, ngram_range=(1,3), stop_words="english")
document = open('input.txt')
read = document.read()
doc = [read]
vect = vec.fit(doc)
terms = vec.get_feature_names()
graph = checkPairs(read, terms)
top5keys = getPageRank(graph)
print("-----------------")
print("Best Keys - Top 5")
print(top5keys)
#drawGraph(graph)
drawGraph2(graph)
