import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from xml.dom.minidom import parse
from glob import glob
import operator
import json
from datetime import datetime
startTime = datetime.now()

filesList = glob('..\\dataset\\DUC-2001\\test\\*.xml')

def parseXml():
	docLists = []

	for file in filesList:
		sent = ""
		dom = parse(file)

		sentences = dom.getElementsByTagName("sentence")
		for sentence in sentences:
			tokens = sentence.getElementsByTagName("token")
			for token in tokens:
				word = token.getElementsByTagName("word")[0]
				sent += word.firstChild.data + " "
		#tratar a pontuacao
		sent1 = sent.replace(" ,", ",")
		sent_final = sent1.replace(" .", ".")
		#adicionar a lista
		docLists.append(sent_final)

	return docLists

def parseFileNames(fileList):
	fileNames = [s.replace('..\\dataset\\DUC-2001\\test\\', '') for s in filesList]
	fileNames = [s.replace('.xml', '') for s in fileNames]
	return fileNames

def parseJson(vectorizer, docvec):
	precisions = {}
	recalls = {}
	f1scores = {}
	precisions_at_5 = {}
	precisions_at_k = {}
	recalls_at_k = {}
	avg_precisions = {}

	with open('../dataset/DUC-2001/references/test.reader.json') as json_file:
		data = json.load(json_file)

	#faz o parse da lista de ficheiros xml para poder percorrer pelos fileNames no ficheiro .json
	fileNames = parseFileNames(filesList)

	ficheiro = 0
	for file in fileNames:
		#cria um dicionario em que associa a cada feature name o seu idf score
		dictionary = dict(zip(vectorizer.get_feature_names(), docvec.toarray()[ficheiro]))

		#ordena o dicionario por ordem decrescente de idf score e mantem apenas o top 10 das mais relevantes para o classifier
		top10Vec = dict(sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True)[:10])

		#guarda o top 10 das keys mais relevantes para o classifier numa lista
		predictedKeys = [key for key in top10Vec.keys()]

		#contém as keyphrases mais relevantes calculadas pelo expert
		expertKeys = [item for sublist in data[file] for item in sublist]

		#guarda a precision para cada documento num dicionario
		precisions[file] = calc_precision(expertKeys, predictedKeys)

		#guarda o recall para cada documento num dicionario
		recalls[file] = calc_recall(expertKeys, predictedKeys)

		#guarda o f1 score para cada documento num dicionario
		f1scores[file] = calc_f1(precisions[file], recalls[file])

		#guarda a Precision@5 para cada documento num dicionario
		precisions_at_5[file] = calc_precision_at_5(expertKeys, predictedKeys)

		#calcula precision@k para cada documento (nao identifica o documento)
		precisions_at_k = calc_precision_at_k(expertKeys, predictedKeys)

		#calcula recalls@k para cada documento(nao identifica o documento)
		recalls_at_k = calc_recall_at_k(expertKeys, predictedKeys)

		#calcula a average precision para cada documento
		avg_precisions[file] = calc_average_precision(precisions_at_k, recalls_at_k)

		ficheiro += 1

	print("Precisions: ", precisions)
	print("Recalls: ", recalls)
	print("F1scores: ", f1scores)
	print("Precisions@5: ", precisions_at_5)
	print("Average Precisions: ", avg_precisions)

	average_recall = calc_avg_recall(recalls)
	average_f1 = calc_avg_f1(f1scores)

	mean_precision_at_5 = calc_mean_precision_at_5(precisions_at_5)
	mean_average_precision = calc_mean_avg_precision(avg_precisions)

	print("Average Recall: ", average_recall)
	print("Average F1: ", average_f1)
	print("Mean Precision@5: ", mean_precision_at_5)
	print("Mean Average Precision: ", mean_average_precision)


def calc_precision(expertKeys, predictedKeys):
	tp = 0
	for key in expertKeys:
		if key in predictedKeys:
			tp += 1
	precision = tp / len(predictedKeys)
	return precision

def calc_recall(expertKeys, predictedKeys):
	tp = 0
	for key in expertKeys:
		if key in predictedKeys:
			tp += 1
	recall = tp / len(expertKeys)
	return recall

def calc_f1(precision, recall):
	f1 = 0
	if(precision != 0 and recall != 0):
		f1 = 2 * ((precision * recall) / (precision + recall))
	return f1


def calc_precision_at_k(expertKeys, predictedKeys):
	precision_at_k = {}
	tp = 0
	k = 0
	for i in range(0, 10):
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
	for i in range(0, 10):
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
			prec = [precisions[i] for i in range(0,10) if recalls[i] >= recall_level]
			max_prec = max(prec)
		except:
			max_prec = 0.0
		prec_at_rec.append(max_prec)
	avg_prec = np.mean(prec_at_rec, dtype=np.float64)
	return avg_prec

def calc_avg_recall(recalls):
	avg_recall = sum(recalls.values()) / len(recalls)
	return avg_recall

def calc_avg_f1(f1scores):
	avg_f1 = sum(f1scores.values()) / len(f1scores)
	return avg_f1

def calc_precision_at_5(expertKeys, predictedKeys):
	# numero de items recomendados @k que sao relevantes
	recRel = 0

	for i in range(0, 5):
		if predictedKeys[i] in expertKeys:
			recRel += 1
	precision_at_5 = recRel / 5

	return precision_at_5

def calc_mean_precision_at_5(precisions_at_5):
	mean_precision_at_5 = sum(precisions_at_5.values()) / len(precisions_at_5)
	return mean_precision_at_5

def calc_mean_avg_precision(mean_avg_precisions):
	mean_avg_precision = sum(mean_avg_precisions.values()) / len(mean_avg_precisions)
	return mean_avg_precision

docList = parseXml()
vectorizer = TfidfVectorizer(use_idf=True, stop_words="english", ngram_range=(1, 3))
docvec = vectorizer.fit_transform(docList)
parseJson(vectorizer, docvec)
print(datetime.now() - startTime)



