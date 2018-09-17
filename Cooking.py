import pandas as pd
import numpy as np
import time
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

def readJson(filename, isTrain = False):
	with open(filename) as f:
		data = json.load(f)
	
	data_id = []
	data_cuisine = []
	data_ingredients = []
	for i in range(0, len(data)):
		data_id.append(data[i]['id'])
		if (isTrain):
			data_cuisine.append(data[i]['cuisine'])
		else:
			data_cuisine.append('')
		data_ingredients.append(' '.join(data[i]['ingredients']))
	inputData = pd.DataFrame({'id':data_id, 'cuisine':data_cuisine, 'ingredients':data_ingredients})
	return inputData

def initData(trainData, testData):
	return [trainData, testData]

def RandomForest(train_features,  trainData, test_features, testData):
	t = time.time()
	rfr = RandomForestClassifier(n_estimators = 100)
	X = train_features
	Y = trainData['cuisine']
	rfr.fit(X, Y)
	result = rfr.predict(test_features)
	finalRes = pd.DataFrame({'id':testData['id'], 'cuisine':result})
	finalRes = finalRes[['id', 'cuisine']]
	finalRes.to_csv('Cooking_result.csv', index = False)
	print('time : %f'%(time.time() - t))
	
if __name__ == '__main__':
	trainData = readJson('./train.json', True)
	testData = readJson('./test.json')
	
	newTrainData = trainData['ingredients']
	newTestData = testData['ingredients']
	vectorizer = CountVectorizer(analyzer="word", tokenizer=None, stop_words=None, max_features=500)
	train_features = vectorizer.fit_transform(newTrainData).toarray()
	test_features = vectorizer.transform(newTestData).toarray()
	print train_features.shape, test_features.shape
	RandomForest(train_features, trainData, test_features, testData)
