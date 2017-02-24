from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import unicodedata

for ii in range(4,5):
    path = "E:/project/wordcount{}/".format(ii)

    files = [unicodedata.normalize('NFC', f) for f in os.listdir(path.decode("utf-8"))]
    dict = {}
    document_list=[]
    name_list = []
    for file in files:
        s = ""
        name_list.append(file)
        with open(path+file,"r") as f:
            words = f.readlines()
            for word in words:
                s += word.split(" ")[0].strip() + " "
            f.close()
            document_list.append(s)
            dict[file] = s

    vectorizer = CountVectorizer(binary=True)
    transformer = TfidfTransformer()
    count = (vectorizer.fit_transform(document_list))
    # count = tmp.toarray()

    # wordlist = vectorizer.get_feature_names()
    # intList = count[0]
    # for i in range(1,len(count)):
    #     print i
    #     for j in range(0,len(count[i])):
    #         intList[j] += count[i][j]
    #
    # with open("E:/delWords.txt","w") as f:
    #     for idx, item in enumerate(wordlist):
    #         s = item.encode("utf-8") + " "+ str(intList[idx])
    #         f.write(s+"\n")
    #     f.close()



    # delWords = []
    # for idx,item in enumerate(intList):
    #     if item<=1:
    #         delWords.append(wordlist[idx])
    #
    # with open("E:/delWords.txt","w") as f:
    #     for word in delWords:
    #         f.write(word.encode("utf-8")+"\n")
    #     f.close()

    #
    #
    # print len(wordlist)
    # print len(intList)
    # for i in range(len(intList)):
    #     if intList[i] <=1:
    #         intList
    # # for i in range(1, len(count)):
    #
    #
    # kmeans = KMeans(n_clusters=15, random_state=0).fit(count)
    # kmeans = KMeans(n_clusters=20, n_init=10, max_iter=300, tol = 0.0001, random_state=0).fit(count)
    # labels = kmeans.labels_
    #
    # # with open("E:/kmeans.txt","w")as f:
    # #     dict = {}
    # #     for idx, name in enumerate(name_list):
    # #         s = name + " , " + unicode(labels[idx])
    # #         print s
    # #         f.write(s.encode("utf-8") + "\n")
    # #     f.close()
    #
    # with open("E:/kmeans{}.txt".format(ii),"w")as f:
    #     dict = {}
    #     for idx, name in enumerate(name_list):
    #         dict[name] = int(labels[idx])
    #
    #     sortDict = sorted(dict.items(), lambda x, y: cmp(x[1], y[1]), reverse=False)
    #     for item in sortDict:
    #         f.write(str(item[1])+" "+item[0].encode("utf-8")+"\n")
    #         s = str(item[1])+" "+item[0].encode("utf-8")
    #     f.close()

    ####
    distortions = []
    allList = []
    for i in range(1,21):
        # dict = {}
        # for i in range(1, 21):
        #     dict[str(i)] = 0
        kmeans = KMeans(n_clusters=i*10 , n_init=10, max_iter=300, tol = 0.0001, random_state=0).fit(count)
        # labels = kmeans.labels_
        # for item in labels:
        #     if dict.has_key(str(item)):
        #         dict[str(item)] += 1
        # allList.append(dict)
        # dict.items()
        distortions.append(kmeans.inertia_)

    # with open("E:/kmeans{}.txt".format(ii),"w")as f:
    #     for idx,aDict in enumerate(allList):
    #         f.write("-------------------  "+str(idx)+"  ---------------------"+"\n")
    #         for item in aDict:
    #             s = item[0] + " " + str(item[1])
    #             f.write(s + "\n")
    #     f.close()

    if ii ==1:
        color1 = "r-"
    elif ii == 2:
        color1 = "b-"
    elif ii == 3:
        color1 = "g-"
    elif ii == 4:
        color1 = "k-"
    else:
        color1 = "c-"
    aList = []
    # for item in distortions:
    #     aList.append(item/max(distortions))
    plt.plot(range(10, 210, 10), distortions, color1)
plt.xlabel("ClustersNumber")
plt.ylabel("Distirtion")
plt.show()


