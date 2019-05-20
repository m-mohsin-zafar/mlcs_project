# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:58:49 2019

@author: Nasir_Ali
"""

from sklearn.decomposition import PCA #import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

data = pd.read_csv("train_proc.csv",header=0)   
data1=np.array(data)
data=data1[:,2:]
target=data1[:,-1]
data=data[:,0:-1]
pca64 = PCA(n_components=970)
pca64.fit(data) #training PCA
projected = pca64.transform(data) #projecting the data onto Principal components
print(data.shape)
print(projected.shape)
plt.plot(np.arange(len(pca64.explained_variance_ratio_))+1,np.cumsum(pca64.explained_variance_ratio_),'o-') #plot the scree graph
plt.axis([1,len(pca64.explained_variance_ratio_)-500,0,1])
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.title('Scree Graph')
plt.grid()
plt.show()
components=pca64.components_[0:420,:]
transform=data.dot(components.T)
generated=transform.dot(components)
i1 = 0 #first principal component
i2 = 1 #second principal component
plt.scatter(projected[:, i1], projected[:, i2],
            c=target, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('nipy_spectral', 10));
plt.grid()
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar();
plt.show()
with open('transform_data.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(transform)

csvFile.close()
with open('generated_data.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(generated)

csvFile.close()