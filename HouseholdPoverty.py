import csv
import pandas as pd
import numpy as np
import time
import lightgbm as lgb
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
import pylab
import matplotlib as plt
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score

def initData():
	t = time.time()
	trainData = pd.read_csv('./train.csv')
	trainData = trainData.drop(['Id', 'idhogar'], axis = 1)
	testData = pd.read_csv('./test.csv')
	originTest = testData
	#testData = testData.drop('id', axis = 1)
	trainNa = trainData.isnull().sum().values * 100.0 / trainData.shape[0] 
	dataNa = pd.DataFrame(trainNa, index=trainData.columns, columns = ['Count'])
	dataNa = dataNa.sort_values(by=['Count'], ascending=False)
	trainData = fillZero(trainData, ['rez_esc', 'meaneduc'])
	cols = ['edjefe', 'edjefa', 'dependency']
	trainData[cols] = trainData[cols].replace({'no':0, 'yes':1}).astype(float)
	testData = testData.drop('idhogar', axis = 1)
	testData[cols] = testData[cols].replace({'no':0, 'yes':1}).astype(float)
	print trainData.info()
	print testData.info()
	trainData = fillWithRegressor(trainData)
	testData = fillWithRegressor(testData.drop('Id', axis = 1))
	print('read data time : %f'%(time.time() - t))
	return [trainData, testData, originTest]

def fillZero(data, col):
	for c in col:
		data[c] = data[c].fillna(0)
	return data

def fillMiddle(data, col):
	for c in col:
		data[c] = data[c].fillna(data[c].median())
	return data

#all are digits
def fillWithRegressor(data):
	dataNa = data.isnull().sum().values
	idxOfData = [x for i,x in enumerate(data.columns) if dataNa[i] != 0]
	#print 'idxOfData', idxOfData
	df = data.drop(idxOfData, axis = 1)
	for idx in idxOfData:
		#print 'idx = ', idx
		df_na = df.loc[(data[idx].notnull())]
		if (len(df_na) == 0):
			continue;
		df_isa = df.loc[(data[idx].isnull())]
		X = df.loc[(data[idx].notnull())]
		Y = data[idx].loc[(data[idx].notnull())]
		rfr = RandomForestRegressor(n_estimators = 10, n_jobs = -1)
		rfr.fit(X, Y)
		fillValue = rfr.predict(df_isa)
		data.loc[(data[idx].isnull()), idx] = fillValue[0]
	return data

def showStatistic(data, col):
	print trainData.info()
	#trainData.select_dtypes(np.int64).nunique().value_counts().sort_index().plot.bar(color='red', figsize=(8,6), edgecolor='k', linewidth = 2)
	#pylab.show()
	print trainData.select_dtypes('object').head()
	print data[col].value_counts(normalize=True)
	
def cv_model(X, y, model, name, model_results = None):
	scorer = make_scorer(f1_score, greater_is_better=True, average = 'macro')
	cv_score = cross_val_score(model, X, y, cv=10, scoring=scorer)
	if model_results is not None:
		model_results = model_results.append(pd.DataFrame({'model':name, 'cv_mean':cv_score.mean(), 'cv_std':cv_score.std()}, index = [0]), ignore_index=True)
	print 'cv_score', cv_score.mean(), cv_score.std()
	return model_results
	
if __name__ == '__main__':
	[trainData, testData, originTest] = initData()
	#showStatistic(trainData, 'Target')
	
	model = RandomForestClassifier(n_estimators = 10, random_state=10, n_jobs=-1)
	X = trainData.drop('Target', axis = 1)
	y = trainData['Target']
	print 'X', X.head(5)
	'''
	model_results = pd.DataFrame(columns = ['model', 'cv_mean', 'cv_std'])
	model_results = cv_model(X, y, LinearSVC(), 'LSVC', model_results)
	model_results = cv_model(X, y, GaussianNB(), 'GNB', model_results)
	model_results = cv_model(X, y, MLPClassifier(hidden_layer_sizes=(32, 64, 128, 64, 32)), 'MLP', model_results)
	model_results = cv_model(X, y, LinearDiscriminantAnalysis(), 'LDA', model_results)
	model_results = cv_model(X, y, RidgeClassifierCV(), 'RIDGE', model_results)
	model_results = cv_model(X, y, KNeighborsClassifier(n_neighbors = 10), 'knn', model_results)
	model_results = cv_model(X, y, ExtraTreesClassifier(n_estimators = 10), 'EXT', model_results)
	model_results = cv_model(X, y, RandomForestClassifier(100), 'RF', model_results)
	
	model_results.set_index('model', inplace = True)
	model_results['cv_mean'].plot.bar(color = 'orange', figsize = (8, 6),
									  yerr = list(model_results['cv_std']),
									  edgecolor = 'k', linewidth = 2)
	model_results.reset_index(inplace = True)
	pylab.show()
	'''
	clf = GaussianNB()
	clf.fit(X, y)
	result = clf.predict(testData)
	finalRes = pd.DataFrame({'Id':originTest['Id'], 'Target':result})
	finalRes.to_csv('HouseholdPoverty_result.csv', index = False)