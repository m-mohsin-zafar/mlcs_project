from sklearn.decomposition import PCA
from paths import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Data
train_df = pd.read_csv(proc_train_csv_path, index_col=0)
test_df = pd.read_csv(proc_test_csv_path, index_col=0)

X_train = train_df.drop(['0', '971'], 1)
X_test = test_df.drop(['0', '971'], 1)
y_train = train_df['971'].astype(int)
y_test = test_df['971'].astype(int)

# Applying PCA
pca = PCA()
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_

# print(explained_variance)

plt.plot(np.arange(len(pca.explained_variance_ratio_)) + 1, np.cumsum(pca.explained_variance_ratio_),
         'o-')  # plot the scree graph
plt.axis([1, len(pca.explained_variance_ratio_) - 500, 0, 1])
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.title('Scree Graph')
plt.grid()
plt.show()

# From the Scree Plot we see that we can retain > 95% Variance for 400 features out of 970
pca_400 = PCA(n_components=400)
X_train = pca_400.fit_transform(X_train)
X_test = pca_400.transform(X_test)

cols = [('PC_'+str(i)) for i in range(1, 401)]

principal_X_train_df = pd.DataFrame(data=X_train, columns=cols)
principal_X_test_df = pd.DataFrame(data=X_test, columns=cols)

principal_train_df = pd.concat([train_df['0'], principal_X_train_df, y_train], axis=1)
principal_test_df = pd.concat([test_df['0'], principal_X_test_df, y_test], axis=1)

ALREADY_SAVED_CSVS = True

if not ALREADY_SAVED_CSVS:
    principal_train_df.to_csv(pca_train_csv_path, index=False)
    principal_test_df.to_csv(pca_test_csv_path, index=False)
