import csv
import pandas as pd
import numpy as np
from sklearn import neighbors
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import time
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

def reviewToWords(review):
	cleanReview = BeautifulSoup(review).get_text()
	cleanReview = re.sub('[^a-zA-Z]', ' ', cleanReview)
	words = cleanReview.lower().split()
	stops = set(stopwords.words('english'))
	finalWords = [w for w in words if not w in stops]
	return (" ".join(finalWords))
	
def initData(trainData, unLabelData, testData):
	newTrainData = []
	newTestData = []
	for i in range(0, trainData['review'].size):
		newTrainData.append(reviewToWords(trainData['review'][i]))
	for i in range(0, testData['review'].size):
		newTestData.append(reviewToWords(testData['review'][i]))
	
	return [newTrainData, newTestData]

#time : 1518.925058
def RandomForest(train_features,  trainData, test_features, testData):
	t = time.time()
	rfr = RandomForestClassifier(n_estimators = 1000)
	X = train_features
	Y = trainData['sentiment']
	rfr.fit(X, Y)
	result = rfr.predict(test_features)
	finalRes = pd.DataFrame({'id':testData['id'], 'sentiment':result})
	finalRes.to_csv('result.csv', index = False)
	print('time : %f'%(time.time() - t))

if __name__ == '__main__':
	trainData = pd.read_csv('./labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
	unLabelData = pd.read_csv('./unlabeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
	testData = pd.read_csv('./testData.tsv', header=0, delimiter="\t", quoting=3)
	
	[newTrainData, newTestData] = initData(trainData, unLabelData, testData)
	vectorizer = CountVectorizer(analyzer="word", tokenizer=None, stop_words=None, max_features=5000)
	train_features = vectorizer.fit_transform(newTrainData).toarray()
	test_features = vectorizer.fit_transform(newTestData)
	RandomForest(train_features, trainData, test_features, testData)
	