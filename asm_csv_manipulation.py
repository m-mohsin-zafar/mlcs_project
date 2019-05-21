#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 11:11:26 2019

@author: mohsin
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from paths import *


def generate_train_test_files():
    train_asm_df = pd.read_csv(asm_csv_path, index_col='filename')
    train_labels_df = pd.read_csv(labels_csv_path, index_col='Id')

    # Merged Both CSVs into one based on Filenames (index) 
    merged_df = pd.merge(train_asm_df, train_labels_df, right_index=True, left_index=True)

    # Group Merged DataFrame By 'Class' i.e Label
    grouped_df = merged_df.groupby('Class')

    # We have a total of 9 different Classes, so 9 groups will be made
    class_df = []
    for i in range(0, 9):
        class_df.append(grouped_df.get_group(i + 1))

        csv_path = os.path.join(root_path, 'data', 'per_class_distribution', ('class_' + (str(i + 1)) + '.csv'))
        class_df[i].to_csv(csv_path, index_label='filename')

    Y_data = np.array(merged_df.values)[:, -1]
    feature_array = np.array(merged_df)[:, :-1]
    f_feature_array = np.array(merged_df.index)

    f_feature_array = np.reshape(f_feature_array, (10868, 1))
    # f_feature_array = f_feature_array.astype(str)

    X_data = np.append(f_feature_array, feature_array, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, stratify=Y_data, test_size=0.4)

    cols = np.append(['filename'], train_asm_df.columns)

    X_train_df = pd.DataFrame(X_train, columns=cols)
    X_test_df = pd.DataFrame(X_test, columns=cols)
    y_train_df = pd.DataFrame(y_train, columns=train_labels_df.columns)
    y_test_df = pd.DataFrame(y_test, columns=train_labels_df.columns)

    train_df = pd.concat([X_train_df, y_train_df], axis=1)
    test_df = pd.concat([X_test_df, y_test_df], axis=1)

    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)


def load_data(train_file_path, test_file_path, dataframes_required=False):
    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)

    X = train_df[train_df.columns[2:971]]
    Y = train_df[train_df.columns[-1]]
    Xts = test_df[test_df.columns[2:971]]
    Yts = test_df[test_df.columns[-1]]

    if dataframes_required is True:
        return X, Xts, Y, Yts, train_df, test_df
    elif dataframes_required is False:
        return X, Xts, Y, Yts


def standardize_data(X, Xts, Y, Yts, train_df, test_df):
    std_scale = StandardScaler().fit(X)
    X_train_norm = std_scale.transform(X)

    training_norm_col = pd.DataFrame(X_train_norm, index=X.index, columns=X.columns)
    train_df.update(training_norm_col)

    X_test_norm = std_scale.transform(Xts)
    testing_norm_col = pd.DataFrame(X_test_norm, index=Xts.index, columns=Xts.columns)
    test_df.update(testing_norm_col)

    train_df.to_csv(proc_train_csv_path, index=False)
    test_df.to_csv(proc_test_csv_path, index=False)


if __name__ == "__main__":
    #    generate_train_test_files()
    X, Xts, Y, Yts, train_df, test_df = load_data(train_csv_path, test_csv_path, dataframes_required=True)
    standardize_data(X, Xts, Y, Yts, train_df, test_df)
