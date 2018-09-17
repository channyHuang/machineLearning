import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

def initData(trainData, testData):
	trainData['date'] = pd.to_datetime(trainData['date'], format='%d.%m.%Y')
	trainData['month'] = trainData['date'].dt.month
	trainData['year'] = trainData['date'].dt.year
	trainData['day'] = trainData['date'].dt.day
	trainData = trainData.groupby([c for c in trainData.columns if c not in ['item_cnt_day']], as_index=False)[['item_cnt_day']].sum()
	trainData = trainData.rename(columns={'item_cnt_day':'item_cnt_month'})
	return [trainData, testData]

def drawFeature(feature, data):
	plt.hist(data[feature], bins=10, range=(data[feature].min(), data[feature].max()))
	plt.title(feature)
	plt.xlabel(feature)
	plt.ylabel('data')
	plt.show()

#need large memory to run...
def RandomForest(trainData, testData):
	t = time.time()
	rfr = RandomForestClassifier(n_estimators = 100)
	X = trainData.drop(['item_cnt_month', 'date'], axis=1)
	y = trainData['item_cnt_month']
	rfr.fit(X, y)
	result = rfr.predict(testData)
	finalRes = pd.DataFrame({'ID':testData['ID'], 'item_cnt_month':result})
	finalRes.to_csv('FutureSales_result.csv', index=False)
	print('time : %f'%(time.time() - t))

if __name__ == '__main__':
	trainData = pd.read_csv('./sales_train_v2.csv')
	testData = pd.read_csv('./test.csv')
	[trainData, testData] = initData(trainData, testData)
	#trainData.info()
	#drawFeature('month', trainData)
	
	RandomForest(trainData, testData)
	
	