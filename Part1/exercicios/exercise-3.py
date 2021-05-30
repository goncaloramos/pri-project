#!/usr/local/bin/python3
import json
import numpy as np
from nltk.corpus import stopwords
import operator
from glob import glob
import re
import nltk
from math import *
from xml.dom.minidom import parse
from datetime import datetime
startTime = datetime.now()

filesList = glob('..\\dataset\\DUC-2001\\test\\*.xml')

#stopWords
stopWords = set(stopwords.words('english'))
#----------------------------------------------------------------------------------------------------#
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

        #tratar a pontuacao
        sent_final = sent.replace(" .", ".")
        sent_final = sent_final.replace(" ,", ",")
        sent_final = "".join(re.sub(r'[^\w\s]', '', sent_final))
        sent_final = ''.join(re.sub('[0-9]', '', sent_final))

        #adicionar a lista
        docsList.append(sent_final)

    return docsList


def parseFileNames(fileList):
	fileNames = [s.replace('..\\dataset\\DUC-2001\\test\\', '') for s in filesList]
	fileNames = [s.replace('.xml', '') for s in fileNames]
	return fileNames


#----------------------------------------------------------------------------------------------------#
def getKeyphrases(docLists, method):
    avg_precisions = {}
    with open('../dataset/DUC-2001/references/test.reader.json') as json_file:
        data = json.load(json_file)
    i=0
    doc_len = 0
    fileNames = parseFileNames(filesList)

    for file in fileNames:
        idf_scores = {}
        tf_scores = {}
        bm25_dict = {}
        bm25_comb_dict = {}

        #chunks de um documento
        if method == "ne":
            chunks = posNE(docLists[i])
        else:
            docLists[i] = docLists[i].lower()
            chunks = posTags(docLists[i])

        print("chunks: ", chunks)

        # avgdl
        doc_len += len(docLists[i])
        avgdl = doc_len / len(docLists)

        print(len(docLists[i]))

        for c in chunks:
            nt = sum(1 for doc in docLists if c in doc)
            idf_scores[c] = idf_calc(len(docLists), nt)
            tf_scores[c] = docLists[i].count(c) / len(docLists[i])
            bm25_dict[c] = bm25_calc(idf_scores[c], tf_scores[c], len(docLists[i]), avgdl)
            bm25_comb_dict[c] = (bm25_calc(idf_scores[c], tf_scores[c], len(docLists[i]), avgdl)) * len(c)


        print("\n")
        print("Printing idf scores \n")
        print(idf_scores)

        print("\n")
        print("printing tf scores \n")
        print(tf_scores)

        # best of 10 computation
        top10_bm25 = dict(sorted(bm25_dict.items(), key=operator.itemgetter(1), reverse=True)[:10])
        top10Keys_bm25 = [item for item in top10_bm25.keys()]

        top10_bm25_comb = dict(sorted(bm25_comb_dict.items(), key=operator.itemgetter(1), reverse=True)[:10])

        print("\n")
        print("----------------------")
        print("Printing best of bm25.. \n")
        print("\n".join("{}\t{}".format(k, v) for k, v in top10_bm25.items()))

        print("\n")
        print("----------------------")
        print("Printing best of bm25comb.. \n")
        print("\n".join("{}\t{}".format(k, v) for k, v in top10_bm25_comb.items()))

        expertKeys = [item for sublist in data[file] for item in sublist]

        print("chunk len: ", len(chunks))

        # calcula precision@k para cada documento (nao identifica o documento)
        precisions_at_k = calc_precision_at_k(expertKeys, top10Keys_bm25)

        # calcula recalls@k para cada documento(nao identifica o documento)
        recalls_at_k = calc_recall_at_k(expertKeys, top10Keys_bm25)

        # calcula a average precision para cada documento
        avg_precisions[file] = calc_average_precision(precisions_at_k, recalls_at_k)

        i += 1

    mean_avg_precision = calc_mean_avg_precision(avg_precisions)
    print("\n")
    print("----------------------")
    print("Mean Average Precision using BM25: ", mean_avg_precision)

#----------------------------------------------------------------------------------------------------#
#Tagging phase
def posTags(document):
    chunks = []
    tokenized = nltk.word_tokenize(document)
    tagged = nltk.pos_tag(tokenized)
    chunkGram = r'Chunk: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'
    chunkParser = nltk.RegexpParser(chunkGram)

    chunked = chunkParser.parse(tagged)

    for subtree in chunked.subtrees():
        cand = ''
        if subtree.label() == 'Chunk':
            for leave in subtree.leaves():
                (expression, tag) = leave
                cand += expression + " "
            cand = cand.rstrip()
            chunks.append(cand)

    return chunks

#Named Entities
def posNE(document):
    chunks = []
    tokenized = nltk.word_tokenize(document)
    tagged = nltk.pos_tag(tokenized)
    chunked = nltk.ne_chunk(tagged, binary=True)

    print(chunked)

    for subtree in chunked.subtrees():
        cand = ''
        if subtree.label() == 'NE':
            for leave in subtree.leaves():
                (expression, tag) = leave
                cand += expression + " "
            cand = cand.rstrip()
            chunks.append(cand)

    return chunks

#----------------------------------------------------------------------------------------------------#
#Metrics
def calc_precision_at_k(expertKeys, predictedKeys):
	precision_at_k = {}
	tp = 0
	k = 0
	for i in range(0, len(predictedKeys)):
		k += 1
		if predictedKeys[i] in expertKeys:
			tp += 1

		precision_at_k['precision@' + str(k)] = tp / k

	return precision_at_k

def calc_recall_at_k(expertKeys, predictedKeys):
	tp_at_k = {}
	recall_at_k = {}
	tp = 0
	k = 0
	j = 0
	total = 0
	for i in range(0, len(predictedKeys)):
		j += 1
		if predictedKeys[i] in expertKeys:
			tp += 1
			total += 1

		tp_at_k['recall@' + str(j)] = tp

	for tp in tp_at_k.values():
		k += 1
		if total == 0:
			recall_at_k['recall@' + str(k)] = 0
		else:
			recall_at_k['recall@' + str(k)] = tp / total

	return recall_at_k


def calc_average_precision(precisions_at_k, recalls_at_k):
	recalls = [recall for recall in recalls_at_k.values()]
	precisions = [precision for precision in precisions_at_k.values()]
	prec_at_rec = []

	for recall_level in np.linspace(0.0, 1.0, 11):
		prec = []
		try:
			prec = [precisions[i] for i in range(0, len(precisions)) if recalls[i] >= recall_level]
			max_prec = max(prec)
		except:
			max_prec = 0.0
		prec_at_rec.append(max_prec)
	avg_prec = np.mean(prec_at_rec, dtype=np.float64)
	return avg_prec

def calc_mean_avg_precision(mean_avg_precisions):
	mean_avg_precision = sum(mean_avg_precisions.values()) / len(mean_avg_precisions)
	return mean_avg_precision

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
#execution
docList = parseXml()
getKeyphrases(docList, "exp") #write "exp" or "ne" to choose how to select the candidates
print(datetime.now() - startTime)