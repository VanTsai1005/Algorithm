#coding:UTF-8

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import os
import unicodedata
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from sklearn.metrics import silhouette_samples
from matplotlib import cm

s1 = datetime.now()
path = "E:/project/wordcount5/"
files = [unicodedata.normalize('NFC', f) for f in os.listdir(path.decode("utf-8"))]
dict = {}
document_list=[]
for file in files:
    s = ""
    num = ""
    with open(path+file,"r") as f:
        words = f.readlines()
        countnumber = 0
        for word in words:
            s += word.split(" ")[0].strip() + " "
            num += word.split(" ")[1].strip() + " "
            countnumber = countnumber+1
            if countnumber > 40:
                break
        f.close()
        # print countnumber
        if countnumber > 40:
            document_list.append(s)
        # dict[file] = s

# print document_list
vectorizer = CountVectorizer(binary=True)
transformer = TfidfTransformer()
count = (vectorizer.fit_transform(document_list).todense())
# print count
###SVD1
# from sklearn.utils.extmath import randomized_svd
# U, Sigma, VT = randomized_svd( count, n_components=200, n_iter=5, random_state=0)

###SVD2
# from sklearn.decomposition import TruncatedSVD
# from sklearn.random_projection import sparse_random_matrix
# print type(count[0])
# svd = TruncatedSVD(n_components=700, n_iter=2, random_state=0)
# svd.fit(count)
# count = svd.fit_transform(count)
# #91%
# print count
# print len(count[0])
# # print X
# explained_variance = svd.explained_variance_ratio_.sum()
# print("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))
# print count

###Kmean++(graph)
km = KMeans(n_clusters=20, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=0)
y_km = km.fit_predict(count)

###Kmean++(graph)
distortions = []
for i in range(20,21):
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=0)
    km.fit(count)
    distortions.append(km.inertia_)
print len(distortions)
plt.plot(range(1,2),distortions,color="r",marker="o")
plt.xlabel("ClustersNumber")
plt.ylabel("Distirtion")
plt.show()

# cluster_labels = np.unique(y_km)
# n_clusters = cluster_labels.shape[0]
# silhouette_vals = silhouette_samples(count, y_km, metric='euclidean')
# y_ax_lower, y_ax_upper = 0, 0
# yticks = []
# for i, c in enumerate(cluster_labels):
#     c_silhouette_vals = silhouette_vals[y_km == c]
#     c_silhouette_vals.sort()
#     y_ax_upper += len(c_silhouette_vals)
#     color = cm.jet(i / n_clusters)
#     plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0,edgecolor='none', color=color)
#     yticks.append((y_ax_lower + y_ax_upper) / 2)
#     y_ax_lower += len(c_silhouette_vals)
#
# silhouette_avg = np.mean(silhouette_vals)
# plt.axvline(silhouette_avg, color="red", linestyle="--")
#
# plt.yticks(yticks, cluster_labels + 1)
# plt.ylabel('Cluster')
# plt.xlabel('Silhouette coefficient')
#
# plt.tight_layout()
# # plt.savefig('E:/silhouette.png', dpi=300)
# plt.show()





s2 = datetime.now()
print "All  Finish "+str(s2-s1)+"!!"




