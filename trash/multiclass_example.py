from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from pycm import ConfusionMatrix
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X, y = iris.data, iris.target

y_binarized = label_binarize(y, classes=[0, 1, 2])
n_classes = 3

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = \
    train_test_split(X, y_binarized, test_size=0.33, random_state=0)
X_train2, X_test2, y_train2, y_test2 = \
    train_test_split(X, y, test_size=0.33, random_state=0)

# classifier
clf = OneVsRestClassifier(SVC())
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_score = clf.decision_function(X_test)

clf2 = SVC()
clf2.fit(X_train2, y_train2)
y_pred2 = clf2.predict(X_test2)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

cm = ConfusionMatrix(actual_vector=y_test2, predict_vector=y_pred2)
print(roc_auc)
print(cm)

# Plot of a ROC curve for a specific class
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
