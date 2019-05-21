import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold
from asm_csv_manipulation import load_data

root_path = os.getcwd()
asm_csv_path = os.path.join(root_path, 'train_asm.csv')
labels_csv_path = os.path.join(root_path, 'trainLabels.csv')
train_csv_path = os.path.join(root_path, 'original', 'train.csv')
test_csv_path = os.path.join(root_path, 'original', 'test.csv')
proc_train_csv_path = os.path.join(root_path, 'processed_data', 'train_proc.csv')
proc_test_csv_path = os.path.join(root_path, 'processed_data', 'test_proc.csv')
dumps_path = os.path.join(root_path, 'content', 'auc_roc_svm_dumps')

n_classes = 9


# A General Function to Evaluate Metrics
def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    y_score = model.decision_function(X_test)
    # print(model)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return fpr, tpr, roc_auc


def perform_kfcv(model, X, Y, folds, dump_filename):
    skf = StratifiedKFold(n_splits=folds)
    scores = []
    # Applying K-Cross Validation
    for train_index, test_index in skf.split(X, Y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        y_train = label_binarize(y_train, classes=[1, 2, 3, 4, 5, 6, 7, 8, 9])
        y_test = label_binarize(y_test, classes=[1, 2, 3, 4, 5, 6, 7, 8, 9])
        fpr, tpr, roc_auc = get_score(model, X_train, X_test, y_train, y_test)
        scores.append(roc_auc)

    scores_df = pd.DataFrame(scores)
    scores_df.to_csv(os.path.join(dumps_path, (dump_filename+".csv")), index=False)
    return scores_df.mean(axis=0).tolist()


def save_object(obj, fname):
    with open(fname, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def main():
    # Loading the Data
    X, Xts, Y, Yts = load_data(proc_train_csv_path, proc_test_csv_path)

    X = X.values
    Xts = Xts.values
    Y = Y.values
    Yts = Yts.values

    # Setting Folds for Cross Validation
    folds = 10

    # Setting up Values to apply Grid Search on SVR
    kernel_values = ['poly']
    c_values = [0.01, 0.1, 1, 100, 500, 1000, 1500]
    gamma_values = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    degree_values = range(2, 32)

    # Setting up Score Lists for Evaluation later on
    poly_svm_scores = []
    scores_list = [poly_svm_scores]

    for kernel, scr_list in zip(kernel_values, scores_list):
        for c in c_values:
            if kernel != 'linear':
                for gamma in gamma_values:
                    if kernel == 'poly':
                        for degree in degree_values:
                            svm_clf = OneVsRestClassifier(SVC(kernel=kernel, degree=degree, gamma=gamma, C=c), n_jobs=2)
                            dump_filename = str(kernel+"_"+str(c)+"_"+str(gamma)+"_"+str(degree))
                            dump_filename = dump_filename.replace('.', '_')
                            roc_auc_list = perform_kfcv(svm_clf, X, Y, folds, dump_filename)
                            scr_list.append(
                                ([kernel, c, gamma, degree] + roc_auc_list))
                    else:
                        svm_clf = OneVsRestClassifier(SVC(kernel=kernel, gamma=gamma, C=c), n_jobs=2)
                        dump_filename = str(kernel + "_" + str(c) + "_" + str(gamma))
                        dump_filename = dump_filename.replace('.', '_')
                        roc_auc_list = perform_kfcv(svm_clf, X, Y, folds, dump_filename)
                        scr_list.append(([kernel, c, gamma] + roc_auc_list))
            else:
                svm_clf = OneVsRestClassifier(SVC(kernel=kernel, C=c), n_jobs=2)
                dump_filename = str(kernel + "_" + str(c))
                dump_filename = dump_filename.replace('.', '_')
                roc_auc_list = perform_kfcv(svm_clf, X, Y, folds, dump_filename)
                scr_list.append(([kernel, c] + roc_auc_list))

    poly_svm_scores_df = pd.DataFrame(poly_svm_scores,
                                      columns=['Kernel', 'C', 'Gamma', 'Degree', 'C1_mean_auc_roc',
                                               'C2_mean_auc_roc', 'C3_mean_auc_roc', 'C4_mean_auc_roc',
                                               'C5_mean_auc_roc', 'C6_mean_auc_roc', 'C7_mean_auc_roc',
                                               'C8_mean_auc_roc', 'C9_mean_auc_roc'])
    poly_svm_scores_df = poly_svm_scores_df.assign(overall_mean=poly_svm_scores_df.iloc[:, 4:].mean(axis=1).tolist())

    # Saving DataFrames for future Reference
    poly_svm_scores_df.to_csv(os.path.join(dumps_path, "poly_svm_metrics.csv"), index=False)

    # Finding Best Scores and Configurations for each Model Based on Overall Mean
    best_poly = poly_svm_scores_df.loc[poly_svm_scores_df['overall_mean'].idxmax()]
    print(best_poly)

    # Saving resultant list object
    # save_object(clf_conf_list, 'pycm_all_cm_results_list_1_obj.pkl')


if __name__ == "__main__":
    main()
