# machineLearning
A beginner, use kaggle data, just practice  
All data for training and testing are from [kaggle](https://www.kaggle.com/)

# 模型选择
初入门，选择简单的python语言，sklearn库  

# 入门
## Dogs vs. Cats
输入数据：图像 (Cat/Dog)     
输出数据：分类 (1: Dog；0: Cat)  
练习目标：图像数据读取
### 数据预处理
输入为单一的.jpg格式图像，且所有图像均有主体（Dog/Cat），无需其它处理操作 

# Last Story
Learned the basic algorithms of ML.
Learned how to pre-load data.
Learned how to analyse data and how to improve my model.
But...
My PC is too old to run large data. I have to give up ML.
Later, I will make my major on linux network programming.

# 后话
机器学习需要硬件支持

```python
from enum import Enum
# 有/无监督学习 (https://sklearn.apachecn.org/#/)

# 最近邻
from sklearn import neighbors
# 最近邻, 无监督学习 (balltree, kdtree, brute-force)
from sklearn.neighbors import NearestNeighbors
# 决策树, 无参监督学习 (ID3, C4.5, C5.0, CART)
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model.logistic import LogisticRegression
# 支持向量机，监督学习 (SVC, NuSVC, LinearSVC)
from sklearn import svm
# 朴素bayes，监督学习
from sklearn.naive_bayes import GaussianNB
# mlp 神经网络模型
from sklearn.neural_network import MLPClassifier

# -------------------------------------------------------------
# 最近邻, 无监督学习
def Nearest(X):
    nearest = NearestNeighbors(n_neighbors = 2, algorithm = 'ball_tree')
    nearest.fit(X)
    #result = nearest.predict(testData)
    #return result

# knn 最近邻
def Knn(X, Y, testData):
    k = 5
    knn = neighbors.KNeighborsClassifier(k, 'distance')
    knn.fit(X, Y)
    result = knn.predict(testData)
    return result

# ------------------------------------------------------------- 
# 逻辑回归
def LogicRegression(X, Y, testData):
    regression = LogicRegression()
    regression.fit(X, Y)
    result = regression.predict(testData)
    return result

class SVCType(Enum):
    SVC = 1,
    LINEARSVC = 2,
    SVR = 3

# svm 支持向量机
def Svm(X, Y, testData):
    if (svnType == SVCType.SVC):
        # SVC
        svc = svm.SVC(kernel = 'rbf', C = 10)
    else if (svnType == SVCType.SVR):
        # LinearSVC
        svc = svm.LinearSVC()
    else:
        svc = svm.SVR()
    svc.fit(X, Y)
    result = svc.predict(testData)
    return result

# 随机梯度下降
def SGD(X, Y, testData):
    sgd = SGDClassifier(loss = 'hinge', penalty = 'l2')
    sgd.fit(X, Y)
    result = sgd.predict(testData)
    return result

# 极度随机树
def ExtraTree(X, Y, testData):
    extraTree = ExtraTreeClassifer(n_estimators = 350)
    extraTree.fit(X, Y)
    result = extraTree.predict(testData)
    return result

# -------------------------------------------------------------
# 决策树
def DecisionTree(X, Y, testData):
    decisionTree = tree.DecisionTreeClassifer()
    decisionTree = decisionTree.fit(X, Y)
    result = decisionTree.predict(testData)
    return result

# 随机森林
def RandomForest(X, Y, testData):
    randomForest = RandomForestClassifier(n_estimators = 1000)
    randomForest.fit(X, Y)
    result = randomForest.predict(testData)
    return result

# 高斯朴素贝叶斯
def GaussianBayes(X, Y, testData):
    gnb = GaussianNB()
    gnb.fit(X, Y)
    result = gnb.predict(testData)
    return result

def AdaBoostAlg(X, Y, testData):
    adaboost = AdaBoostClassifier(n_estimators = 100)
    adaboost.fit(X, Y)
    result = adaboost.predict(testData)
    return result

# MLP 神经网络模型
def MLP(X, Y, testData):
    mlp = MLPClassifier(solver = 'lbfgs', alpha = 1e - 5, hidden_layer_sizes = (5, 2), random_state = 1)
    mlp.fit(X, Y)
    result = mlp.predict(testData)
    return result
```