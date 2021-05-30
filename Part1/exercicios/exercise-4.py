import json
from glob import glob
from math import log
from xml.dom.minidom import parse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from nltk.corpus import stopwords
import re
import operator
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
        sent_final = "".join(re.sub(r'[^\w\s]', '', sent_final.lower()))
        sent_final = ''.join(re.sub('[0-9]', '', sent_final))

        #adicionar a lista
        docsList.append(sent_final)

    return docsList

def parseFileNames(fileList):
	fileNames = [s.replace('..\\dataset\\DUC-2001\\test\\', '') for s in filesList]
	fileNames = [s.replace('.xml', '') for s in fileNames]
	return fileNames


# -----------------------------------------------------------------------------------------------
# Creates data to train and test the classifier
''' X = [[feature_1, feature_2, feature_3, feature_4, feature_5, feature_6], ...]
	where each index of X corresponds to a candidate
    y [target_1, ...]
    where each index of y corresponds to a candidate
'''
'''
		X[i][0] = Position of the candidate in the document
		X[i][1] = Length of the candidate
		X[i][2] = Term frequency
		X[i][3] = inverse document frequency
		X[i][4] = TF-IDF
		X[i][5] = BM25 scores for the candidate
		X[i][6] = Word count of the candidate
		y = 1 - if its a keyphrase
			0 - if it isn't
'''
def create_train_data(docsList, vectorizer, docvec):
	fileNames = parseFileNames(filesList)
	fileNames = fileNames
	avg_precisions = {}
	termkeys = {}
	i = 0
	ficheiro = 0
	doc_len = 0

	with open('../dataset/DUC-2001/references/test.reader.json') as json_file:
		data = json.load(json_file)

	for file in fileNames:
		X = []
		y = []
		topKeys = {}

		# cria um dicionario em que associa a cada feature name o seu idf score
		dictionary = dict(zip(vectorizer.get_feature_names(), docvec.toarray()[ficheiro]))
		terms = [key for key, value in dictionary.items() if value != 0.0]
		len(terms)

		# avgdl
		doc_len += len(docsList[i])
		avgdl = doc_len / len(docsList)

		expertKeys = [item for sublist in data[file] for item in sublist]

		for term in terms:
			features = []
			# position
			features.append(candidatePosition(docsList[ficheiro], term))
			position = features[0]

			# length
			features.append(candidateLength(term))
			length = features[1]

			# tf
			features.append(docsList[i].count(term) / len(docsList[i]))
			tf_score = features[2]

			# idf
			nt = sum(1 for doc in docsList if term in doc)
			features.append(idf_calc(len(docsList), nt))
			idf_score = features[3]

			# tf-idf
			features.append(tf_score * idf_score)
			tf_idf_score = features[4]

			# bm25_score
			features.append(bm25_calc(idf_score, tf_score, len(docsList[i]), avgdl))
			bm25_score = features[5]

			#candidate word count
			features.append(candidateWordCount(term))
			wordCount = features[6]

			if term in expertKeys:
				y.append(1)
			else:
				y.append(0)

			X.append(features)

			termkeys[term] = features

		docKeys = [term for term in termkeys.keys()]

		ppn = create_final_classifier(X, y)
		confidences = ppn.decision_function(X)

		termo = 0
		for conf in confidences:
			if conf > 0:
				topKeys[docKeys[termo]] = conf

			termo += 1

		print("\n")
		print("----------------------")
		print("Printing top keys... \n")

		topKeys = dict(sorted(topKeys.items(), key=operator.itemgetter(1), reverse=True)[:5])
		print(topKeys)

		print("\n")
		print("----------------------")

		top5Keys = [key for key in topKeys.keys()]

		expertKeys = [item for sublist in data[file] for item in sublist]

		# calcula precision@k para cada documento (nao identifica o documento)
		precisions_at_k = calc_precision_at_k(expertKeys, top5Keys)

		# calcula recalls@k para cada documento(nao identifica o documento)
		recalls_at_k = calc_recall_at_k(expertKeys, top5Keys)

		if bool(precisions_at_k) is False or bool(recalls_at_k) is False:
			avg_precisions[file] = 0
		else:
			# calcula a average precision para cada documento
			avg_precisions[file] = calc_average_precision(precisions_at_k, recalls_at_k)

		ficheiro += 1

	mean_average_precision = calc_mean_avg_precision(avg_precisions)
	print("Mean Average Precision: ", mean_average_precision)


#----------------------------------------------------------------------------------------------------#
#Features Calculation

#candidate position
def candidatePosition(document, candidate):
	c_index = document.find(candidate)
	return c_index

#----------------------------------------------------------------------------------------------------#
#candidate length
def candidateLength(candidate):
	c_len = len(candidate)
	return c_len
#----------------------------------------------------------------------------------------------------#
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
#Metrics
def calc_precision_at_k(expertKeys, predictedKeys):
	precision_at_k = {}
	tp = 0
	k = 0

	if len(predictedKeys) == 0:
		return precision_at_k

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

	if len(predictedKeys) == 0:
		return recall_at_k

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
#Perceptron Classifiers

# Trains classifier with features from X and what our classifier will predict from y
def train_classifier(X, y):

	# Split the data into 70% training data and 30% test data
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

	# Train the scaler, which standarizes all the features to have mean=0 and unit variance
	sc = StandardScaler()
	sc.fit(X_train)

	# Apply the scaler to the X training data
	X_train_std = sc.transform(X_train)

	# Apply the SAME scaler to the X test data
	X_test_std = sc.transform(X_test)

	# Create a perceptron object with the parameters that increase its accuracy
	ppn = Perceptron()

	# Train the perceptron
	ppn.fit(X_train_std, y_train)

	# Apply the trained perceptron on the X data to make predicts for the y test data
	y_pred = ppn.predict(X_test_std)

	# View the accuracy of the model, which is: 1 - (observations predicted wrong / total observations)
	print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
	print("predicted: ", y_pred)

#----------------------------------------------------------------------------------------------------#
# Creates the final classifier after finding the best parameters for the classifier
def create_final_classifier(X, y):

	# Train the scaler, which standarizes all the features to have mean=0 and unit variance
	sc = StandardScaler()
	sc.fit(X)

	# Apply the scaler to the X training data
	X_std = sc.transform(X)

	# Create a perceptron object with the parameters: 40 iterations (epochs) over the data, and a learning rate of 0.1
	ppn = Perceptron()

	# Train the perceptron
	ppn.fit(X_std, y)

	return ppn

#----------------------------------------------------------------------------------------------------#
#Execution
docsList = parseXml()
vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1, 3))
docvec = vectorizer.fit_transform(docsList)
create_train_data(docsList, vectorizer, docvec)
print(datetime.now() - startTime)
