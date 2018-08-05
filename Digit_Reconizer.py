import csv
import pandas as pd
import numpy as np
from sklearn import neighbors
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.linear_model.logistic import LogisticRegression
from sklearn import svm

def initData(trainData, testData):
	return [trainData, testData]

def knn(trainData, testData):
	t = time.time()
	k = 5
	weights = ['uniform', 'distance']
	clf = neighbors.KNeighborsClassifier(k, 'distance')
	clf.fit(trainData.values[:, 1:], np.ravel(trainData.values[:, 0]))
	result = clf.predict(testData)
	index = range(1, len(result) + 1)
	finalRes = pd.DataFrame({'ImageId':index, 'Label':result})
	finalRes.to_csv('result.csv', index = False)
	print('time : %f'%(time.time() - t))

def Decision_Tree(trainData, testData):
	t = time.time()
	dt = tree.DecisionTreeClassifier()
	dt = dt.fit(trainData.values[:, 1:], trainData['label'])
	result = dt.predict(testData)
	index = range(1, len(result) + 1)
	finalRes = pd.DataFrame({'ImageId':index, 'Label':result})
	finalRes.to_csv('result.csv', index = False)
	print('time : %f'%(time.time() - t))

def RandomForest(trainData, testData):
	t = time.time()
	rfr = RandomForestClassifier(n_estimators = 1000)
	X = trainData.values[:, 1:]
	Y = trainData.values[:, 0]
	rfr.fit(X, Y)
	result = rfr.predict(testData)
	index = range(1, len(result) + 1)
	finalRes = pd.DataFrame({'ImageId':index, 'Label':result})
	finalRes.to_csv('result.csv', index = False)
	print('time : %f'%(time.time() - t))

def calcPassP():
	myResult = pd.read_csv('result.csv')
	rightResult = pd.read_csv('./kaggleData/Digit_recognizer/score_1.0_submission.csv')
	rightResult = rightResult['Label'].astype(int)
	myResult = myResult['Label'].astype(int)
	res = 0.0
	for i, j in zip(rightResult, myResult):
		res += (i + j) % 2
	print(1 - res/len(myResult) * 1.0)
			
def Logic_regression():
	t = time.time()
	cf = LogisticRegression()
	X = trainData.values[:, 1:]
	Y = trainData.values[:, 0]
	cf.fit(X, Y)
	result = rfr.predict(testData)
	index = range(1, len(result) + 1)
	finalRes = pd.DataFrame({'ImageId':index, 'Label':result})
	finalRes.to_csv('result.csv', index = False)
	print('time : %f'%(time.time() - t))
	
def svmClassify(trainData, testData):
	t = time.time()
	X = trainData.values[:, 1:]
	Y = trainData.values[:, 0]
	svc = svm.SVC(kernel = 'rbf', C = 10)
	svc.fit(X, Y)
	result = svc.predict(testData)
	index = range(1, len(result) + 1)
	finalRes = pd.DataFrame({'ImageId':index, 'Label':result})
	finalRes.to_csv('result.csv', index = False)
	print('time : %f'%(time.time() - t))
	
if __name__ == '__main__':
	trainData = pd.read_csv('./kaggleData/Digit_recognizer/train.csv')
	testData = pd.read_csv('./kaggleData/Digit_recognizer/test.csv')
	
	[trainData, testData] = initData(trainData, testData)
	trainData.info()
	'''
	Decision_Tree(trainData, testData)
	calcPassP()
	
	#take a long long ~~~ time
	knn(trainData, testData)
	calcPassP()
	
	#take more time than knn
	svmClassify(trainData, testData)
	calcPassP()
	'''
	RandomForest(trainData, testData)
	calcPassP()
	