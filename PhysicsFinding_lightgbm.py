#from https://www.kaggle.com/muonneutrino/lightgbm-starter-some-data-exploration

import csv
import pandas as pd
import numpy as np
import time
import lightgbm as lgb

t = time.time()

trainData = pd.read_csv('./training.csv')
testData = pd.read_csv('./test.csv')

feature_name = ['FlightDistance', 'LifeTime', 'pt', 'IP']
features = trainData[feature_name]
trainSet = lgb.Dataset(features, trainData['signal'])
params = {
	'task': 'train',
	'boosting_type': 'gbdt',
    'objective': 'binary',
    'num_leaves': 31,
    'metric': {'auc'},
    'learning_rate': 0.01,
}
cv_output = lgb.cv(
    params,
    trainSet,
    num_boost_round=400,
    nfold=10,
)
best_niter = np.argmax(cv_output['auc-mean'])
best_score = cv_output['auc-mean'][best_niter]
print('Best number of iterations: {}'.format(best_niter))
print('Best CV score: {}'.format(best_score))
model = lgb.train(params, trainSet, num_boost_round=best_niter)

result = model.predict(testData[feature_name])
finalRes = pd.DataFrame({'id':testData['id'], 'prediction':result})
finalRes.to_csv('PhysicsFinding_result_gbm.csv', index = False)

print('time : %f'%(time.time() - t))