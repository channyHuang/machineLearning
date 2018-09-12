import csv
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier

def RandomForest(X_train, y_train, testData):
	t = time.time()
	rfr = RandomForestClassifier(n_estimators = 100)
	rfr.fit(X_train, y_train)
	result = rfr.predict(testData)
	index = range(1, len(result) + 1)
	finalRes = pd.DataFrame({'Id':index, 'Solution':result})
	finalRes.to_csv('result.csv', index = False)
	print('time : %f'%(time.time() - t))

if __name__ == '__main__':
	X_train = pd.read_csv('train.csv', header = None)
	y_train = pd.read_csv('trainLabels.csv', header = None)
	testData = pd.read_csv('test.csv', header = None)
	
	RandomForest(X_train, y_train, testData)
	
	