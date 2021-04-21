# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 11:37:08 2021

@author: rahul
"""

import pandas as pd
import matplotlib.pylab as plt 
iris = pd.read_csv("E:\\Data Science\\Data Sheet\\iris.csv")
#normalization(EDA)
def norm_func(i):
    x = (i-i.min())/(i.max()  -  i.min())
    return (x)

df_norm = norm_func(iris.iloc[:,1:4])
df_norm.describe()

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
#linkage method (complete)
z = linkage(df_norm, method="complete",metric="euclidean")
#dendrogram
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  
    leaf_font_size=10.,  
)
plt.show()

from	sklearn.cluster	import	AgglomerativeClustering 
h_complete	=	AgglomerativeClustering(n_clusters=3,	linkage='complete',affinity = "euclidean").fit(df_norm) 
h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)

iris['clust']=cluster_labels  
iris = iris.iloc[:,1:]
iris.head()

iris.groupby(iris.clust).mean()
