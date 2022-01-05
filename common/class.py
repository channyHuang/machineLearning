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