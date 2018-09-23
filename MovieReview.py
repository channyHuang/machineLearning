import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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

def initData(trainData, testData):
	newTrainData = []
	newTestData = []
	for i in range(0, trainData['Phrase'].size):
		newTrainData.append(reviewToWords(trainData['Phrase'][i]))
	for i in range(0, testData['Phrase'].size):
		newTestData.append(reviewToWords(testData['Phrase'][i]))
	return [newTrainData, newTestData]

def RandomForest(train_features, trainData, test_features, testData):
	t = time.time()
	rfr = RandomForestClassifier(n_estimators = 100)
	X = train_features
	y = trainData['Sentiment']
	rfr.fit(X, y)
	result = rfr.predict(test_features)
	finalRes = pd.DataFrame({'PhraseId':testData['PhraseId'], 'Sentiment':result})
	finalRes.to_csv('MovieReview_result.csv', index=False)
	print('time : %f'%(time.time() - t))

if __name__ == '__main__':
	trainData = pd.read_csv('./train.tsv', delimiter="\t", quoting=3)
	testData = pd.read_csv('./test.tsv', delimiter="\t", quoting=3)
	
	[newTrainData, newTestData] = initData(trainData, testData)
	vectorizer = CountVectorizer(analyzer="word", tokenizer=None, stop_words=None, max_features=5000)
	train_features = vectorizer.fit_transform(newTrainData).toarray()
	test_features = vectorizer.fit_transform(newTestData)
	
	RandomForest(train_features, trainData, test_features, testData)