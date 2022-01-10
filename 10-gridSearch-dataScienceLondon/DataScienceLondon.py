import csv
import pandas as pd
import numpy as np
import time

from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    X_train = pd.read_csv('./kaggleData/DataScienceLondon/train.csv', header = None)
    y_train = pd.read_csv('./kaggleData/DataScienceLondon/trainLabels.csv', header = None)
    testData = pd.read_csv('./kaggleData/DataScienceLondon/test.csv', header = None)
    
    svm_params = {"C": [10, 100, 1000, 1e4, 1e5]}
    nb_params = {"var_smoothing": [1e-8, 1e-9, 1e-10]}
    knn_params = {"n_neighbors": range(5, 10)}
    tree_params = {"max_depth": [5, 10, None], "min_samples_split":[2,3,4,5], "min_samples_leaf": [1,2,3,4]}

    svm_grid = GridSearchCV(SVC(), svm_params, n_jobs = 1, cv = 10)
    nb_grid = GridSearchCV(GaussianNB(), nb_params, n_jobs = 1, cv = 10)
    knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, n_jobs = 1, cv = 10)
    tree_grid = GridSearchCV(DecisionTreeClassifier(), tree_params, n_jobs = 1, cv = 10)

    svm_grid.fit(X_train, y_train)
    nb_grid.fit(X_train, y_train)
    knn_grid.fit(X_train, y_train)
    tree_grid.fit(X_train, y_train)

    print(svm_grid.best_score_, nb_grid.best_score_, knn_grid.best_score_, tree_grid.best_score_)

    bic = []
    lowest_bic = np.inf
    best_gm = None
    X = np.r_[X_train, testData]
    for i in range(1, 7):
        gm = GaussianMixture(n_components = i)
        gm.fit(X)
        bic.append(gm.aic(X))
        if (bic[-1] < lowest_bic):
            lowest_bic = bic[-1]
            best_gm = gm
    print(best_gm)
    best_gm.fit(X)
    gm_train = best_gm.predict_proba(X_train)
    gm_test = best_gm.predict_proba(testData)

    svm_grid.fit(gm_train, y_train)
    nb_grid.fit(gm_train, y_train)
    knn_grid.fit(gm_train, y_train)
    tree_grid.fit(gm_train, y_train)
    
    print(svm_grid.best_score_, nb_grid.best_score_, knn_grid.best_score_, tree_grid.best_score_)

    pred = knn_grid.predict(gm_test)

    pred_df = pd.DataFrame(pred, columns=['Solution'], index = np.arange(1, 9001))
    pred_df.index.name = 'Id'
    pred_df.reset_index(drop = False, inplace = True)
    pred_df.to_csv('submission.csv', index = False)
