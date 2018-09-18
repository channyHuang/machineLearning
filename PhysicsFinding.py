import csv
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

#data is so perfect that don't need to do any pre-operation
def initData():
	t = time.time()
	trainData = pd.read_csv('./training.csv')
	trainData = trainData.drop(['id', 'production', 'mass', 'min_ANNmuon'], axis = 1)
	testData = pd.read_csv('./test.csv')
	#testData = testData.drop('id', axis = 1)
	print trainData.info()
	print testData.info()
	print('read data time : %f'%(time.time() - t))
	return [trainData, testData]

def RandomForest(trainData, testData):
	t = time.time()
	rfr = RandomForestRegressor(n_estimators = 250)
	X = trainData.drop('signal', axis = 1)
	Y = trainData['signal']
	rfr.fit(X, Y)
	result = rfr.predict(testData.drop('id', axis = 1))
	finalRes = pd.DataFrame({'id':testData['id'], 'prediction':result})
	finalRes.to_csv('PhysicsFinding_result.csv', index = False)
	print('time : %f'%(time.time() - t))

if __name__ == '__main__':
	[trainData, testData] = initData()
	RandomForest(trainData, testData)