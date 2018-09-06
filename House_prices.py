import csv
import pandas as pd
import numpy as np
import sklearn.tree as tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

def StringToInt(data, idxOfData):
	for idx in idxOfData:
		data[idx] = data[idx].fillna('N')
		uniData = set(data[idx])
		for i, d in zip(range(0, len(uniData)), uniData):
			data[idx] = data[idx].replace(d, i)
		data[idx] = data[idx].astype(int)
	return data

def fillMiddle(data):
	idxOfData = data.select_dtypes(exclude=[np.object]).columns
	for idx in idxOfData:
		data[idx] = data[idx].fillna(data[idx].median())
	return data

#all are digits
def fillWithRegressor(data):
	idxOfData = data.select_dtypes(include=[np.int64, np.float32]).columns
	print 'idx = ', idxOfData
	df = data.drop(idxOfData, axis = 1)
	for idx in idxOfData:		
		df_na = df.loc[(data[idx].notnull())]
		if (len(df_na) == 0):
			continue;
		df_isa = df.loc[(data[idx].isnull())]
		X = df.values[:, :]
		Y = data[idx]
		rfr = RandomForestRegressor(n_estimators = 1000, n_jobs = -1)
		rfr.fit(X, Y)
		fillValue = rfr.predict(df.values[:, :])
		data.loc[(data[idx].isnull()), idx] = fillValue
	print data.info
	return data

def initData(trainData, testData):
	trainData = StringToInt(trainData, ['MSZoning', 'Street', 'Alley', 'LotShape', \
			'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', \
			'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', \
			'Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', \
			'Heating', 'HeatingQC', 'CentralAir', 'KitchenQual', 'Functional', 'PavedDrive', \
			'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'SaleType', 'SaleCondition', \
			'BsmtFinType2', 'Electrical', 'FireplaceQu', 'GarageType', 'GarageFinish', \
		    'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature',])
	
	trainData['MasVnrArea'] = trainData['MasVnrArea'].fillna(trainData['MasVnrArea'].median())
	trainData['GarageYrBlt'] = trainData['GarageYrBlt'].fillna(trainData['GarageYrBlt'].median())
	trainData['LotFrontage'] = trainData['LotFrontage'].fillna(trainData['LotFrontage'].median())
	#trainData = fillWithRegressor(trainData)
	#trainData.info()

	testData = StringToInt(testData, ['MSZoning', 'Street', 'Alley', 'LotShape', \
			'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', \
			'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', \
			'Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', \
			'Heating', 'HeatingQC', 'CentralAir', 'KitchenQual', 'Functional', 'PavedDrive', \
			'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'SaleType', 'SaleCondition', \
			'BsmtFinType2', 'Electrical', 'FireplaceQu', 'GarageType', 'GarageFinish', \
		    'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature',])
	
	testData = fillMiddle(testData)
	#testData = fillWithRegressor(testData)
	#testData.info()
	
	return [trainData, testData]

def Decision_Tree(trainData, testData):
	[trainData, testData] = initData(trainData, testData)
	
	
	dt = tree.DecisionTreeClassifier()
	inputData = trainData.drop('SalePrice', axis = 1)
	dt = dt.fit(inputData, trainData['SalePrice'])
	
	result = dt.predict(testData)
	
	finalRes = pd.DataFrame({'Id':testData['Id'], 'SalePrice':result})

	finalRes.to_csv('result.csv', index = False)

def Ramdom_Forest(trainData, testData):
	[trainData, testData] = initData(trainData, testData)
	
	rfr = RandomForestClassifier(n_estimators = 1000)
	X = trainData.drop('SalePrice', axis = 1)
	Y = trainData['SalePrice']
	rfr.fit(X, Y)
	result = rfr.predict(testData.values[:, :])
	finalRes = pd.DataFrame({'Id':testData['Id'], 'SalePrice':result})
	finalRes.to_csv('result.csv', index = False)	
	
if __name__ == '__main__':
	trainData = pd.read_csv('./kaggleData/House_prices/train.csv')
	testData = pd.read_csv('./kaggleData/House_prices/test.csv')
	
	#trainData.info()
	[trainData, testData] = initData(trainData, testData)
	
	#Decision_Tree(trainData, testData)
	Ramdom_Forest(trainData, testData)
	