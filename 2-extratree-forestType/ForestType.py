import csv
import pandas as pd
import numpy as np
import time
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pylab

#also, without null data, seems good
def initData():
	t = time.time()
	trainData = pd.read_csv('./train.csv')
	trainData = trainData.drop(['Id'], axis = 1)
	testData = pd.read_csv('./test.csv')
	#testData = testData.drop('id', axis = 1)
	print trainData.info()
	print testData.info()
	print('read data time : %f'%(time.time() - t))
	return [trainData, testData]

#Failed
#Evaluation Exception: The value 0.146483902518097 is above the agreement threshold of 0.09.
def RandomForest(trainData, testData):
	t = time.time()
	rfr = RandomForestClassifier(n_estimators = 250)
	X = trainData.drop('Cover_Type', axis = 1)
	Y = trainData['Cover_Type']
	rfr.fit(X, Y)
	result = rfr.predict(testData.drop('Id', axis = 1))
	finalRes = pd.DataFrame({'Id':testData['Id'], 'Cover_Type':result})
	finalRes.to_csv('ForestType_result.csv', index = False)
	print('time : %f'%(time.time() - t))
	
if __name__ == '__main__':
	[trainData, testData] = initData()
	#RandomForest(trainData, testData)
	plt.figure(figsize=(12,5))
	plt.title("")
	#ax = sns.distplot(trainData['Cover_Type'])
	#sns.FacetGrid(trainData, hue='Cover_Type', size=10).map(plt.scatter, 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology').add_legend()
	#pylab.show()
	
	etc = ExtraTreesClassifier(n_estimators = 350)
	etc.fit(trainData.drop('Cover_Type', axis = 1), trainData['Cover_Type'])
	result = etc.predict(testData.drop('Id', axis = 1))
	finalRes = pd.DataFrame({'Id':testData['Id'], 'Cover_Type':result})
	finalRes = finalRes[['Id', 'Cover_Type']]
	finalRes.to_csv('ForestType_result.csv', index = False)