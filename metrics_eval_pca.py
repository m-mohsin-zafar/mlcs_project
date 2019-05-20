import pickle
import os
from pycm import ConfusionMatrix
import pandas as pd
import numpy as np
from paths import *
from asm_csv_manipulation import load_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

knn_result_object_names = ['pycm_all_cm_results_list_1_obj.pkl', 'pycm_all_cm_results_list_2_obj.pkl']
n_classes = 9


def load_object(fname):
    with open(fname, 'rb') as inp:
        obj = pickle.load(inp)

    return obj


def load_data_pca():
    # Load Data
    train_df = pd.read_csv(pca_train_csv_path)
    test_df = pd.read_csv(pca_test_csv_path)

    X_train = train_df.drop(['0', '971'], 1)
    X_test = test_df.drop(['0', '971'], 1)
    y_train = train_df['971'].astype(int)
    y_test = test_df['971'].astype(int)

    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values

    return X_train, y_train, X_test, y_test


# KNN Results Extraction from Saved Objects
ALREADY_HAVE_KNN_CV_RESULTS = True

if not ALREADY_HAVE_KNN_CV_RESULTS:
    results_list = []
    results_objs = [load_object(os.path.join(knn_pycm_dumps_path, i)) for i in knn_result_object_names]
    for o in results_objs:
        results_list = results_list + o

    kfcv_results = []
    for particular_configuration in results_list:
        print(particular_configuration)
        folds = range(2, 12)
        scores = []
        for fold in folds:
            cm = particular_configuration[fold]
            f1_scores = [v for v in cm.F1.values()]
            scores.append(f1_scores)

        dump_filename = str('knn_' + str(particular_configuration[0]) + '_' + str(particular_configuration[1]))
        scores_df = pd.DataFrame(scores)
        scores_df.to_csv(os.path.join(knn_pycm_dumps_path, (dump_filename + ".csv")), index=False)
        kfcv_results.append(
            ([particular_configuration[0], particular_configuration[1]] + scores_df.mean(axis=0).tolist()))

    knn_scores_df = pd.DataFrame(kfcv_results, columns=['p', 'k', 'C1_mean_f1',
                                                        'C2_mean_f1', 'C3_mean_f1', 'C4_mean_f1',
                                                        'C5_mean_f1', 'C6_mean_f1', 'C7_mean_f1',
                                                        'C8_mean_f1', 'C9_mean_f1'])
    knn_scores_df = knn_scores_df.assign(overall_mean=knn_scores_df.iloc[:, 2:].mean(axis=1).tolist())
    knn_scores_df.to_csv(os.path.join(knn_pycm_dumps_path, 'knn_results.csv'), index=False)

# SVM Polynomial Kernel Results Computations
# ALREADY_HAVE_POLY_SVM_RESULTS = False
#
# if not ALREADY_HAVE_POLY_SVM_RESULTS:
#     for file in os.listdir(dumps_path):
#         filename = os.fsdecode(file)
#         if filename.startswith('poly'):
#             print(filename)

# Determining Best Configurations
knn_results_df = pd.read_csv(os.path.join(knn_pycm_dumps_path, 'knn_results.csv'))
best_knn = knn_results_df.loc[knn_results_df['overall_mean'].idxmax()]

# Based on AUC ROC Scores
linear_svm_auc_roc_results_df = pd.read_csv(os.path.join(dumps_path, "linear_svm_metrics.csv"))
rbf_svm_auc_roc_results_df = pd.read_csv(os.path.join(dumps_path, "rbf_svm_metrics.csv"))

best_linear_svm_roc = linear_svm_auc_roc_results_df.loc[linear_svm_auc_roc_results_df['overall_mean'].idxmax()]
best_rbf_svm_roc = rbf_svm_auc_roc_results_df.loc[rbf_svm_auc_roc_results_df['overall_mean'].idxmax()]

# Based on F1 scores
linear_svm_f1_results_df = pd.read_csv(os.path.join(svm_pycm_dumps_path, "linear_svm_metrics.csv"))
rbf_svm_f1_results_df = pd.read_csv(os.path.join(svm_pycm_dumps_path, "rbf_svm_metrics.csv"))

best_linear_svm_f1 = linear_svm_f1_results_df.loc[linear_svm_f1_results_df['overall_mean'].idxmax()]
best_rbf_svm_f1 = rbf_svm_f1_results_df.loc[rbf_svm_f1_results_df['overall_mean'].idxmax()]

# Loading the Data For Final Evaluations
X_train, Y_train, X_test, Y_test = load_data_pca()

# Determining Results for KNN
# ----------------------------------------------------------------
# Based on Best Configurations found
knn_clf = KNeighborsClassifier(
    n_neighbors=best_knn.k.astype(int),
    p=best_knn.p.astype(int),
    n_jobs=-1
)
knn_clf.fit(X_train, Y_train)
y_pred = knn_clf.predict(X_test)
cm = ConfusionMatrix(actual_vector=Y_test, predict_vector=y_pred)
f1_scores = [v for v in cm.F1.values()]
print(f1_scores)
print("(KNN) Mean F1 Score = ", np.array(f1_scores).mean())

# # Based on Random Guess
# knn_clf_guess = KNeighborsClassifier(
#     n_neighbors=12,
#     p=2,
#     n_jobs=-1
# )
# knn_clf_guess.fit(X_train, Y_train)
# y_pred_guess = knn_clf_guess.predict(X_test)
# cm_guess = ConfusionMatrix(actual_vector=Y_test, predict_vector=y_pred_guess)
# f1_scores_guess = [v for v in cm_guess.F1.values()]
# print(f1_scores_guess)
# print("(Guess KNN) Mean F1 Score = ", np.array(f1_scores_guess).mean())

# --------------------------------------------------------------
# Determining Results for SVM Based on F1 Scores
# Linear SVM
f_linear_svm_clf = SVC(
    C=best_linear_svm_f1.C,
    kernel=best_linear_svm_f1.Kernel
)
f_linear_svm_clf.fit(X_train, Y_train)
y_pred_f_linear = f_linear_svm_clf.predict(X_test)
cm_f_linear = ConfusionMatrix(actual_vector=Y_test, predict_vector=y_pred_f_linear)
f1_scores_linear_svm = [v for v in cm_f_linear.F1.values()]
print(f1_scores_linear_svm)
print("(Linear SVM) Mean F1 Score = ", np.array(f1_scores_linear_svm).mean())

# --------------------------------------------------------------
# RBF Kernel
f_rbf_svm_clf = SVC(
    C=best_rbf_svm_f1.C,
    kernel=best_rbf_svm_f1.Kernel,
    gamma=best_rbf_svm_f1.Gamma
)
f_rbf_svm_clf.fit(X_train, Y_train)
y_pred_f_rbf = f_rbf_svm_clf.predict(X_test)
cm_f_rbf = ConfusionMatrix(actual_vector=Y_test, predict_vector=y_pred_f_rbf)
f1_scores_rbf_svm = [v for v in cm_f_rbf.F1.values()]
print(f1_scores_rbf_svm)
print("(RBF SVM) Mean F1 Score = ", np.array(f1_scores_rbf_svm).mean())

# --------------------------------------------------------------
# Determining Results for SVM Based on AUC ROC Scores
# Linear SVM

Y_train_binarized = label_binarize(Y_train, classes=[1, 2, 3, 4, 5, 6, 7, 8, 9])
Y_test_binarized = label_binarize(Y_test, classes=[1, 2, 3, 4, 5, 6, 7, 8, 9])

linear_svm_clf = OneVsRestClassifier(SVC(
    C=best_linear_svm_roc.C,
    kernel=best_linear_svm_roc.Kernel
), n_jobs=-1)
linear_svm_clf.fit(X_train, Y_train_binarized)
y_scores_linear = linear_svm_clf.decision_function(X_test)
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_test_binarized[:, i], y_scores_linear[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

print(roc_auc)
print("(Linear SVM) Mean ROC AUC Score = ", np.fromiter(roc_auc.values(), dtype=float).mean())

# --------------------------------------------------------------
# RBF Kernel

rbf_svm_clf = OneVsRestClassifier(SVC(
    C=best_rbf_svm_roc.C,
    kernel=best_rbf_svm_roc.Kernel,
    gamma=best_rbf_svm_roc.Gamma
), n_jobs=-1)
rbf_svm_clf.fit(X_train, Y_train_binarized)
y_scores_rbf = rbf_svm_clf.decision_function(X_test)
# Compute ROC curve and ROC area for each class
rbf_fpr = dict()
rbf_tpr = dict()
rbf_roc_auc = dict()
for i in range(n_classes):
    rbf_fpr[i], rbf_tpr[i], _ = roc_curve(Y_test_binarized[:, i], y_scores_rbf[:, i])
    rbf_roc_auc[i] = auc(rbf_fpr[i], rbf_tpr[i])

print(rbf_roc_auc)
print("(RBF SVM) Mean ROC AUC Score = ", np.fromiter(rbf_roc_auc.values(), dtype=float).mean())


# Comparisons Based on F1 Scores
if best_knn.overall_mean > best_linear_svm_f1.overall_mean > best_rbf_svm_f1.overall_mean:
    print(best_knn)
elif best_linear_svm_f1.overall_mean > best_knn.overall_mean > best_rbf_svm_f1.overall_mean:
    print(best_linear_svm_f1)
else:
    print(best_rbf_svm_f1)

# Comparison Based on AUC ROC
if best_linear_svm_roc.overall_mean > best_rbf_svm_roc.overall_mean:
    print(best_linear_svm_roc)
else:
    print(best_rbf_svm_roc)
