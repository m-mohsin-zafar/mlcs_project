import os


root_path = os.getcwd()
# asm_csv_path = os.path.join(root_path, 'train_asm.csv')
asm_csv_path = os.path.join(root_path, 'feature_ext', 'io_content', 'train_asm.csv')
# labels_csv_path = os.path.join(root_path, 'trainLabels.csv')
labels_csv_path = os.path.join(root_path, 'feature_ext', 'io_content', 'trainLabels.csv')

train_csv_path = os.path.join(root_path, 'data', 'original', 'train.csv')
test_csv_path = os.path.join(root_path, 'data', 'original', 'test.csv')

proc_train_csv_path = os.path.join(root_path, 'data', 'processed', 'train_proc.csv')
proc_test_csv_path = os.path.join(root_path, 'data', 'processed', 'test_proc.csv')

pca_train_csv_path = os.path.join(root_path, 'data', 'processed', 'pca_train.csv')
pca_test_csv_path = os.path.join(root_path, 'data', 'processed', 'pca_test.csv')

dumps_path = os.path.join(root_path, 'data', 'dumps')
svm_auc_roc_dumps_path = os.path.join(dumps_path, 'auc_roc_svm_dumps')
svm_pycm_dumps_path = os.path.join(dumps_path, 'pycm_svm_dumps')
knn_pycm_dumps_path = os.path.join(dumps_path, 'pycm_knn_dumps')

LINEAR_SVM_ONE_VS_REST_PATH = os.path.join(root_path, 'data', 'saved_models', 'linear_svm_ovr_trained_model.joblib')
RBF_SVM_ONE_VS_REST_PATH = os.path.join(root_path, 'data', 'saved_models', 'rbf_svm_ovr_trained_model.joblib')
RBF_SVM_PATH = os.path.join(root_path, 'data', 'saved_models', 'rbf_svm_trained_model.joblib')
LINEAR_SVM_PATH = os.path.join(root_path, 'data', 'saved_models', 'linear_svm_trained_model.joblib')
KNN_MODEL_PATH = os.path.join(root_path, 'data', 'saved_models', 'knn_trained_model.joblib')
