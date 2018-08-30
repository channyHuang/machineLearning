import csv
import pandas as pd
import numpy as np

def StringToInt(data, idxOfData):
	for idx in idxOfData:
		data[idx] = data[idx].fillna('N')
		uniData = set(data[idx])
		for i, d in zip(range(0, len(uniData)), uniData):
			data[idx] = data[idx].replace(d, i)
		data[idx] = data[idx].astype(int)
	return data

#all are digits
def fillWithRegressor(data):
	idxOfData = data.select_dtypes(include=[np.number]).columns
	print idxOfData
	df = data.drop(idxOfData)
	for idx in idxOfData:		
		df_na = df.loc[(data[idx].notnull())]
		if (len(df_na) == 0):
			continue;
		df_isa = df.loc[(data[idx].isnull())]
		X = df.valuse[:, :]
		Y = data.valuse[idx]
		rfr = RandomForestRegressor(n_estimators = 1000, n_jobs = -1)
		rfr.fit(X, Y)
		fillValue = rfr.predict(df.values[:, :])
		data.loc[(data[idx].isnull()), idx] = fillValue
	return data

def initData(trainData, testData):
	trainData = fillWithRegressor(trainData)
	trainData = StringToInt(trainData, ['MSZoning', 'Street', 'Alley', 'LotShape', \
			'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', \
			'BsmtFinType2', 'Electrical', 'FireplaceQu', 'GarageType', 'GarageFinish', \
		    'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature'])
	
	trainData['MasVnrArea'] = trainData['MasVnrArea'].fillna(trainData['MasVnrArea'].median())
	trainData['GarageYrBlt'] = trainData['GarageYrBlt'].fillna(trainData['GarageYrBlt'].median())
	trainData['LotFrontage'] = trainData['LotFrontage'].fillna(trainData['LotFrontage'].median())
	
	return [trainData, testData]

if __name__ == '__main__':
	trainData = pd.read_csv('./kaggleData/House_prices/train.csv')
	testData = pd.read_csv('./kaggleData/House_prices/test.csv')
	
	#trainData.info()
	[trainData, testData] = initData(trainData, testData)
	
	#print('-'*40)
	#trainData.info()
	