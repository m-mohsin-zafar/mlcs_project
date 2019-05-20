#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:47:17 2019

@author: mohsin
"""

import pandas as pd
import numpy as np
import os
import pickle
from paths import *
from pycm import ConfusionMatrix
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from asm_csv_manipulation import load_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier


def encode_targets(Y_in):
    #    lb = LabelBinarizer()
    #    lb.fit(Y_in)
    le = LabelEncoder()
    return le.fit_transform(Y_in)


# A General Function to Evaluate Metrics
def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(model)
    cm = ConfusionMatrix(actual_vector=y_test, predict_vector=y_pred)

    return cm


def perform_kfcv(model, X, Y, folds=10):
    skf = StratifiedKFold(n_splits=folds)
    scores = []
    # Applying K-Cross Validation
    for train_index, test_index in skf.split(X, Y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        scores.append(get_score(model, X_train, X_test, y_train, y_test))

    return scores


def save_object(obj, fname):
    with open(fname, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def main():
    # Loading the Data
    X, Xts, Y, Yts = load_data(proc_train_csv_path, proc_test_csv_path)

    X = X.values
    Xts = Xts.values

    # Encoding Targets
    Y_encoded = encode_targets(Y.values)
    Yts_encoded = encode_targets(Yts.values)

    # Setting Folds for Cross Validation
    folds = 10

    # Setting up Value of 'k'
    k_vals = range(2, 51)

    # Setting up Value of 'p'
    p_vals = range(2, 6)

    # Initializing a One Vs All KNN Model
    # nn_clf = OneVsRestClassifier(KNeighborsClassifier(n_jobs=-1))
    # nn_clf = KNeighborsClassifier()

    # Confusion Matrix List
    clf_conf_list = []

    for p in p_vals:
        for k in k_vals:
            nn_clf = KNeighborsClassifier(n_neighbors=k, p=p)
            results = perform_kfcv(nn_clf, X, Y_encoded, folds)
            clf_conf_list.append((p, k, results))

    # Saving resultant list object
    save_object(clf_conf_list, 'pycm_all_cm_results_list_1_obj.pkl')


if __name__ == "__main__":
    main()
