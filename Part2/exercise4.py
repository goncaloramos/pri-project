#!/usr/local/bin/python3
from nltk.tokenize import *
from sklearn.feature_extraction.text import TfidfVectorizer
import feedparser as fp
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import os
import operator
from itertools import islice
from write_html import write_message
import webbrowser

#===================================================================================
#   To test more feeds:     Add more feed links to the data/links.txt file
#
#   Additional features:    Script stores feeds and reads from database
#===================================================================================

#===================================================================================
#	Feed Functions
#===================================================================================

def readLinks(f):
    links = []
    input = open(f, 'r')

    for line in input:
        links.append(u""+line.lower())

    input.close()
    return links

def getFeed(webData):
    feed = fp.parse(webData)
    return feed

def getListRSS(linkList):
    listRSS = []

    for link in linkList:
        feed = getFeed(link)
        for entry in feed['entries']:
            listRSS.append(entry)

    return listRSS

def storeOnDatabase(path, dictDBNews):
    pickle.dump(dictDBNews, open(path,'wb'))

def readDatabase(path):
    database = pickle.load(open(path,'rb'))
    return database

#===================================================================================
#	Extract Titles and Descriptions from articles in the Feed Functions
#===================================================================================

def getContentNews(linkList):
    contentList = []

    for link in linkList:
        sent = ""
        feed = getFeed(link)

        print("\n\n Newsletter \n\n")
        for entry in feed['entries']:
            print ("\tNew: ", entry['link'])
            print ("\t\tTitle : ", entry['title'])
            print ("\t\tDescription : ", entry['description'])
            print ("\n\n")

            #sentence processing
            sent = entry['title'] + ' ' + entry['description']
            sent = sent.replace(" .", ".")
            sent = sent.replace(" ,", ",")
            sent = "".join(re.sub(r'[^\w\s]', '', sent.lower()))
            sent = ''.join(re.sub('[0-9]', '', sent))

            contentList.append(sent)

    return contentList

def getContentNewsFromDB(db):
    contentList = []

    for entry in db:
        print ("\tNew: ", entry['link'])
        print ("\t\tTitle : ", entry['title'])
        print ("\t\tDescription : ", entry['description'])
        print ("\n\n")

        #sentence processing
        sent = entry['title'] + ' ' + entry['description']
        sent = sent.replace(" .", ".")
        sent = sent.replace(" ,", ",")
        sent = "".join(re.sub(r'[^\w\s]', '', sent.lower()))
        sent = ''.join(re.sub('[0-9]', '', sent))

        contentList.append(sent)

    return contentList

#===================================================================================
#	Keyphrases Functions
#===================================================================================

def tokens(doc):
    sentences = sent_tokenize(doc)
    return sentences

def findWholeWord(term, sentence):
    return re.compile(r'\b({0})\b'.format(term), flags=re.IGNORECASE).search(sentence)

def getTerms(contentList):
    vec = TfidfVectorizer(use_idf=True, ngram_range=(1,3), stop_words="english")
    vect = vec.fit(contentList)
    terms = vec.get_feature_names()

    return terms

def buildGraph(contentList, terms):
    graph = {}
    terms_Sent = []

    for sent in contentList:
        termos = []
        for term in terms:
            if findWholeWord(term, sent):
                termos.append(term)

        terms_Sent.append(termos)

    for sent in terms_Sent:
        for term in sent:
            sente = [x for x in sent if x != term]
            if term in graph:
                graph[term] += sente
            else:
                graph[term] = sente

    for term in terms:
    	if term not in graph:
    		graph[term] = []

    return graph

def getBestPageRank(graph):
    g = nx.DiGraph(graph) #direct graph
    pr = nx.pagerank(g, alpha=0.15, weight=None, max_iter=50)
    bestRanked = dict(sorted(pr.items(), key=operator.itemgetter(1), reverse=True)[:50])

    #print("\n\n Keyphrases Best Ranked \n")
    #for k, v in bestRanked.items():
    #    print("\t\t",k, v)

    return bestRanked

def getNewsClusterFromDB(keys10, db):
    cluster = {}

    for key in keys10:

        temp = ''.join(key)
        cluster[key]= []
        #cluster[key] = ""
        titulo = "Keyword - " + temp + ":" + "\n\n\n"
        cluster[temp].append(titulo)
        for entry in db:

            sent = '' + entry['description']
            sent = sent.replace(" .", ".")
            sent = sent.replace(" ,", ",")
            sent = "".join(re.sub(r'[^\w\s]', '', sent.lower()))
            sent = ''.join(re.sub('[0-9]', '', sent))

            titulo = '' + entry['title']
            titulo = sent.replace(" .", ".")
            titulo = sent.replace(" ,", ",")
            titulo = "".join(re.sub(r'[^\w\s]', '', titulo.lower()))
            titulo = ''.join(re.sub('[0-9]', '', titulo))
            titulo = titulo + "\n\n"


            if temp in sent:
                cluster[temp].append(titulo)
                #cluster[temp] = titulo

            #cluster[temp] = "\n".join(cluster[temp])


    return cluster

def getDocCountForKeywords(cluster):
    lst = []

    for term in cluster:
        lst.append(len(cluster[term]) - 1)

    return lst

#===================================================================================
#	Script Execution
#===================================================================================

#===================================================================================
#	Global Variables
#===================================================================================
f = "data/links.txt"
path = "db/entries.p"

#===================================================================================
#	Obter lista de feeds
#===================================================================================

listFeeds = readLinks(f)
print("Numero de feeds: ", len(listFeeds))
for line in listFeeds:
    print(line)

#===================================================================================
#	Obter lista RSS
#===================================================================================

lstRss = getListRSS(listFeeds)
#test = getFeed("feed:https://rss.nytimes.com/services/xml/rss/nyt/Space.xml")

#===================================================================================
#	Save feed data
#===================================================================================

storeOnDatabase(path, lstRss)
db = readDatabase(path)

#===================================================================================
#	Additional Function
#===================================================================================

def take(n, iterable):
    #Return first n items of the iterable as a list
    return list(islice(iterable, n))

#===================================================================================
#	Get Keyphrases
#===================================================================================

#newsContentList = getContentNews(listFeeds)
newsContentList = getContentNewsFromDB(db)
terms = getTerms(newsContentList)
graph = buildGraph(newsContentList, terms)
keyphrases50 = getBestPageRank(graph)

word_set = []
score_set = []

for key, value in keyphrases50.items():
    word_set.append(key)
    score_set.append(str(value))

str_words = ','.join(word_set)

print("\n\n word set listing \n")
print(word_set)
print("\n\n score set listing \n")
print(score_set)
print("\n\n word string listing \n")
print(str_words)


print("\n\n Keyphrases Best Ranked \n")
n_items = take(10, keyphrases50.items())
for k, v in n_items:
    print("\t\t",k, v)
print("\n")

toCluster = word_set[:10]
print("\n\n Keyphrases to Cluster \n")
print(toCluster)

clusteredNews = getNewsClusterFromDB(toCluster, db)
#print("\n\n Top 10 Clustered Keyphrases Results \n")
#print(clusteredNews)

cluster_set =[]

for key, value in clusteredNews.items():
    cluster_set.append(str(value))

count = getDocCountForKeywords(clusteredNews)

#===================================================================================
#	Get and Open HTML
#===================================================================================

html = open('keyphrases.html','w')
message = write_message(word_set, score_set, str_words, cluster_set,count)
html.write(message)
html.close()



chrome_path="C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s"
fname = 'file:///'+ os.getcwd()+'/' + 'keyphrases.html'
print("Opening file: \t\t", fname+'/keyphrases.html')
webbrowser.get(chrome_path).open_new_tab(fname)

#	End of Script
#===================================================================================

print("\n\nEnd of Script")
