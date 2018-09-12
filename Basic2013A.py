import csv
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time

def init(X_train, y_train, testData):
	for i in range(0, X_train.shape[1]):
		X_train[i] = X_train[i].fillna(X_train[i].median())
	for i in range(0, testData.shape[1]):
		testData[i] = testData[i].fillna(testData[i].median())
	return [X_train, y_train, testData]

def RandomForest(X_train,  y_train, testData):
	t = time.time()
	rfr = RandomForestClassifier(n_estimators = 1000)
	rfr.fit(X_train, y_train)
	result = rfr.predict(testData)
	index = range(1, len(result) + 1)
	finalRes = pd.DataFrame({'':result})
	finalRes.to_csv('result.csv', index = False)
	print('time : %f'%(time.time() - t))
	
if __name__ == '__main__':
	X_train = pd.read_csv('./train.csv', header=None)
	y_train = pd.read_csv('./train_labels.csv', header=None)
	testData = pd.read_csv('./test.csv', header=None)
	
	[X_train, y_train, testData] = init(X_train, y_train, testData)
	
	RandomForest(X_train, y_train, testData)