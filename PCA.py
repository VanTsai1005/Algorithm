from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import unicodedata

path = "E:/project/testKmeans2/"
files = [unicodedata.normalize('NFC', f) for f in os.listdir(path.decode("utf-8"))]

dict = {}
document_list = []
name_list = []
for file in files:
    ss = ""
    name_list.append(file)
    with open(path + file, "r") as f:
        words = f.readlines()
        for word in words:
            ss += word.split(" ")[0].strip() + " "
        f.close()
        document_list.append(ss)
        dict[file] = ss

vectorizer = CountVectorizer(binary=True)
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(document_list))
words = vectorizer.get_feature_names()
weight = tfidf.toarray()

clf = KMeans(n_clusters=2)
s = clf.fit(weight)
label = []
print(clf.labels_)
i = 1
for i in range(len(clf.labels_)):
    print name_list[i], clf.labels_[i-1]
    label.append(clf.labels_[i-1])
    i = i + 1
print(clf.inertia_)

from sklearn.decomposition import PCA

pca = PCA(n_components=3)
newData = pca.fit_transform(weight)
print newData


x1 = []
y1 = []
z1 = []
x2 = []
y2 = []
z2 = []
for i in range(0,len(newData)):
    if label[i]==1:
        x1.append(newData[i][0])
        y1.append(newData[i][1])
        z1.append(newData[i][2])
    else:
        x2.append(newData[i][0])
        y2.append(newData[i][1])
        z2.append(newData[i][2])
# i = 0
# while i < len(newData)/4:
#     x1.append(newData[i][0])
#     y1.append(newData[i][1])
#     i += 1
#
# x2 = []
# y2 = []
# i = len(newData)/4
# while i < len(newData)/4*2:
#     x2.append(newData[i][0])
#     y2.append(newData[i][1])
#     i += 1
#
# x3 = []
# y3 = []
# i = len(newData)/4*2
# while i < len(newData)/4*3:
#     x3.append(newData[i][0])
#     y3.append(newData[i][1])
#     i += 1
#
# x4 = []
# y4 = []
# i = len(newData)/4*3
# while i < len(newData)/4*4:
#     x4.append(newData[i][0])
#     y4.append(newData[i][1])
#     i += 1

plt.plot(x1, y1, z1, 'or')
plt.plot(x2, y2, z2, 'og')
# plt.plot(x3, y3, 'ob')
# plt.plot(x4, y4, 'ok')
plt.show()
