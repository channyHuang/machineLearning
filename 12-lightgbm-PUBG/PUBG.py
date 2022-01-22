import numpy as np
import pandas as pd
import csv
import time
import lightgbm as lgb

t = time.time()

def info(trainData, testData):
    # print data info, found 4 object columns
    #trainData.info()
    #testData.info() 
    
    trainData = trainData.drop(['Id', 'groupId', 'matchId'], axis = 1)
    testData = testData.drop(['Id', 'groupId', 'matchId'], axis = 1)
    # print types of matchType column
    matchTypeSets = trainData['matchType'].unique()
    # count matchType
    matchTypeSs = pd.Series(trainData['matchType'])
    #print(matchTypeSs.value_counts())
    # replace string to number
    matchTypeMaps = dict((v, i) for i, v in enumerate(matchTypeSets))
    #matchTypeMaps = map(matchTypeSets, range(1, matchTypeSets.size))
    trainData['matchType'] = trainData['matchType'].map(matchTypeMaps).astype(int)
    testData['matchType'] = testData['matchType'].map(matchTypeMaps).astype(int)
    return [trainData, testData]

def init(X_train, y_train, testData):
    # fill nan
    trainNa = X_train.isnull().sum().values * 100.0 / X_train.shape[0] 
    dataNa = pd.DataFrame(trainNa, index=X_train.columns, columns = ['Count'])
    dataNa = dataNa.sort_values(by=['Count'], ascending=False)
    print(dataNa.head())
    for i in X_train.columns:
        X_train[i] = X_train[i].fillna(X_train[i].median())
    for i in testData.columns:
        testData[i] = testData[i].fillna(testData[i].median())
    # replace type
     
    return [X_train, y_train, testData]

if __name__ == '__main__':
    trainData = pd.read_csv('./kaggleData/PUBG/train_V2.csv')
    testData = pd.read_csv('./kaggleData/PUBG/test_V2.csv')
    testGroup = testData[['Id', 'matchId', 'groupId']].copy()
    # data cleaning
    [trainData, testData] = info(trainData, testData)
    
    X_train = trainData.drop(['winPlacePerc'], axis = 1)
    y_train = trainData['winPlacePerc']
    [X_train, y_train, testData] = init(X_train, y_train, testData)
    # build model
    
    params={'learning_rate': 0.05,
        'objective':'mae',
        'metric':'mae',
        'num_leaves': 128,
        'verbose': 1,
        'random_state':42,
        'bagging_fraction': 0.7,
        'feature_fraction': 0.7
    }
    reg = lgb.LGBMRegressor(**params, n_estimators = 10000)
    reg.fit(X_train, y_train)
    pred = reg.predict(testData, num_iteration = reg.best_iteration_)
    
    testGroup['winPlacePerc'] = pred
    testData = pd.concat([testData, testGroup], axis = 1)
    result = testData
    result = result[['Id', 'winPlacePerc']]
    result.to_csv('submission.csv', index = False)
    print('time : %f'%(time.time() - t))
