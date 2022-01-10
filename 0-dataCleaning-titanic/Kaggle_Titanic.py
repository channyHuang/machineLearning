import csv
import pandas as pd
import sklearn.tree as tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

def initData(trainData, testData):
	trainData['Sex'] = trainData['Sex'].apply(lambda x:1 if x == 'male' else 0)
	testData['Sex'] = testData['Sex'].apply(lambda x:1 if x == 'male' else 0)
	
	#trainData['Cabin'] = trainData['Cabin'].apply(lambda x: 1
	trainData = trainData.drop(['PassengerId', 'Ticket', 'Cabin'], 1)
	testData = testData.drop(['Ticket', 'Cabin'], 1)
	
	trainData['Name'] = trainData.Name.str.extract(' ([A-Za-z]+)\.', expand = False)
	trainData['Name'] = trainData['Name'].replace(['Lady', 'Countess','Capt', 'Col',\
    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
	trainData['Name'] = trainData['Name'].replace(['Mlle', 'Ms'], 'Miss')
	trainData['Name'] = trainData['Name'].replace('Mme', 'Mrs')
	trainData['Name'] = trainData['Name'].map({'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}).astype(int)
	
	testData['Name'] = testData.Name.str.extract(' ([A-Za-z]+)\.', expand = False)
	testData['Name'] = testData['Name'].replace(['Lady', 'Countess','Capt', 'Col',\
    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
	testData['Name'] = testData['Name'].replace(['Mlle', 'Ms'], 'Miss')
	testData['Name'] = testData['Name'].replace('Mme', 'Mrs')
	testData['Name'] = testData['Name'].map({'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}).astype(int)

	testData['Fare'] = testData['Fare'].fillna(testData['Fare'].median())
	
	df = trainData[['Age', 'Sex', 'Fare', 'Parch', 'SibSp', 'Pclass', 'Name']]
	df_na = df.loc[(trainData.Age.notnull())]
	df_isa = df.loc[(trainData.Age.isnull())]
	X = df_na.values[:, 1:]
	Y = df_na.values[:, 0]
	rfr = RandomForestRegressor(n_estimators = 1000, n_jobs = -1)
	rfr.fit(X, Y)
	fillAge = rfr.predict(df_isa.values[:, 1:])
	trainData.loc[(trainData.Age.isnull()), 'Age'] = fillAge
	'''
	df = testData[['Age', 'Sex', 'Fare', 'Parch', 'SibSp', 'Pclass', 'Name']]
	df_na = df.loc[(testData.Age.notnull())]
	df_isa = df.loc[(testData.Age.isnull())]
	df_na.info()
	X = df_na.values[:, 1:]
	Y = df_na.values[:, 0]
	rfr = RandomForestRegressor(n_estimators = 100, n_jobs = -1)
	rfr.fit(X, Y)
	fillAge = rfr.predict(df_isa.values[:, 1:])
	testData.loc[(testData.Age.isnull()), 'Age'] = fillAge
	'''
	trainData['Embarked'] = trainData['Embarked'].fillna('N')
	trainData['Embarked'] = trainData['Embarked'].map({'S':0, 'C':1, 'Q':2, 'N':3}).astype(int)
	
	testData['Embarked'] = testData['Embarked'].fillna('N')
	testData['Embarked'] = testData['Embarked'].map({'S':0, 'C':1, 'Q':2, 'N':3}).astype(int)

	return [trainData, testData]
	
def Decision_Tree(trainData, testData):
	[trainData, testData] = initData(trainData, testData)
	#trainData['Age'] = trainData['Age'].fillna(trainData['Age'].median())
	testData['Age'] = testData['Age'].fillna(testData['Age'].median())
	
	feature = ['Age', 'Sex']
	
	dt = tree.DecisionTreeClassifier()
	dt = dt.fit(trainData[feature], trainData['Survived'])
	
	result = dt.predict(testData[feature])
	
	finalRes = pd.DataFrame({'PassengerId':testData['PassengerId'], 'Survived':result})

	finalRes.to_csv('result.csv', index = False)
	
def Ramdom_Forest(trainData, testData):
	[trainData, testData] = initData(trainData, testData)
	#trainData['Age'] = trainData['Age'].fillna(trainData['Age'].median())
	testData['Age'] = testData['Age'].fillna(testData['Age'].median())
	
	rfr = RandomForestClassifier(n_estimators = 1000)
	X = trainData.drop('Survived', axis = 1)
	Y = trainData['Survived']
	rfr.fit(X, Y)
	result = rfr.predict(testData.values[:, 1:])
	finalRes = pd.DataFrame({'PassengerId':testData['PassengerId'], 'Survived':result})
	finalRes.to_csv('result.csv', index = False)
	
def calcPassP():
	rightResult = pd.read_csv('./kaggleData/Titanic/final_prediction.csv')
	myResult = pd.read_csv('result.csv')
	rightResult = rightResult['Survived'].astype(int)
	myResult = myResult['Survived'].astype(int)
	res = 0.0
	for i, j in zip(rightResult, myResult):
		res += (i + j) % 2
	print(1 - res/len(myResult)*1.0)
	
if __name__ == '__main__':
	trainData = pd.read_csv('./kaggleData/Titanic/train.csv')
	testData = pd.read_csv('./kaggleData/Titanic/test.csv')
	'''
	trainData.info()
	print('-'*40)
	testData.info()
	print('-'*40)
	'''
	#Decision_Tree(trainData, testData)
	Ramdom_Forest(trainData, testData)
	calcPassP()