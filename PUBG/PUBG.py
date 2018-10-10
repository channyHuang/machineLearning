#PC is too slow to read train data......

import numpy as np
import pandas as pd
import csv
from sklearn.ensemble import RandomForestClassifier
import time

def init(X_train, y_train, testData):
	trainNa = trainData.isnull().sum().values * 100.0 / trainData.shape[0] 
	dataNa = pd.DataFrame(trainNa, index=trainData.columns, columns = ['Count'])
	dataNa = dataNa.sort_values(by=['Count'], ascending=False)
	dataNa.head()
	'''
	for i in range(0, X_train.shape[1]):
		X_train[i] = X_train[i].fillna(X_train[i].median())
	for i in range(0, testData.shape[1]):
		testData[i] = testData[i].fillna(testData[i].median())
	'''
	return [X_train, y_train, testData]

def RandomForest(X_train,  y_train, testData):
	t = time.time()
	rfr = RandomForestClassifier(n_estimators = 1000)
	rfr.fit(X_train, y_train)
	result = rfr.predict(testData.drop('Id', axis = 1))
	index = range(1, len(result) + 1)
	finalRes = pd.DataFrame({'Id':testData['Id'], 'winPlacePerc':result})
	finalRes.to_csv('result.csv', index = False)
	print('time : %f'%(time.time() - t))

if __name__ == '__main__':
	trainData = pd.read_csv('train.csv')
	testData = pd.read_csv('test.csv')
	X_train = trainData.drop(['Id', 'winPlacePerc'], axis = 1)
	y_train = trainData['winPlacePerc']

	#trainData.info()
	[X_train, y_train, testData] = init(X_train, y_train, testData)
	
	#RandomForest(X_train, y_train, testData)
