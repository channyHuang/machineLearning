import csv
import pandas as pd

def initData(trainData, testData):
	return [trainData, testData]

if __name__ == '__main__':
	trainData = pd.read_csv('./kaggleData/House_prices/train.csv')
	testData = pd.read_csv('./kaggleData/House_prices/test.csv')
	
	trainData.info()
	
	[trainData, testData] = initData(trainData, testData)