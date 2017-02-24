# -*- coding: utf-8 -*-

# 參考網址：http://hxjc.lingw.net/article-6661993-1.html

import sys
import os
import jieba
import pymongo
import json
import codecs
from pymongo import MongoClient
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np


#導入訓練集
for i in range(0,7):
    article_list = []
    # start = 1 + 1000*i
    # end = 1001 + 1000*i
    start = 1 + 1000*i
    end = 1001 + 1000*i
    for j in range(start,end):
        try:
            with open("E:/project/segmentation1/article_"+str(j)+".txt","r") as f:
                article_list.append(f.read())
        except:
            break
        finally:
            f.close()
    # s = "一不小心 一位 一個不小心 一個人 一個月 一共 一分 一列 一到 一包"
    # s1 = "一個不小心 一個人 一個月 一到 一位 "
    # article_list.append(s)
    # article_list.append(s1)
    # s = "親子家庭 "
    # article_list.append(s)

    # client=MongoClient('mongodb://10.120.37.128:27017')
    # db = client.__getitem__("family")
    # travel = db.__getitem__("travel")
    # find_contain_keyword_list=list(db.travel.find())
    # jieba.load_userdict("E:/userWord.txt")
    # jieba.load_userdict("E:/ab104g3_userdict.txt")
    #
    # article_list=[]
    #
    # for dicts in find_contain_keyword_list :
    #     content=dicts['content']
    #     seglist=jieba.cut(content,cut_all=False)
    #     s = " ".join((seglist))
    #     article_list.append(" ".join((seglist)))


    #从文件导入停用词表
    # stpwrdpath = "extra_dict/hlt_stop_words.txt"
    # stpwrd_dic = open(stpwrdpath, 'rb')
    # stpwrd_content = stpwrd_dic.read()
    #将停用词表转换为list
    # stpwrdlst = stpwrd_content.splitlines()
    # stpwrd_dic.close()

    #将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer=CountVectorizer(binary=True) #创建词袋数据结构

    #创建hash向量词袋
    # vectorizer = HashingVectorizer(stop_words =stpwrdlst,n_features = 10000) #设置停用词词表,设置最大维度10000


    # #统计每个词语的tf-idf权值
    transformer = TfidfTransformer()

    # #第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    # tfidf = transformer.fit_transform(vectorizer.fit_transform(article_list))

    # # step1:得出結果(0, 3174)	1 (x,y)z x是應該是指在第幾個文本,y應該是指該字詞排列後的索引值,z應該代表該詞在所有文本中出現過幾次
    vectorizer.fit_transform(article_list)
    count = (vectorizer.fit_transform(article_list).todense())
    #
    for j in range(0,len(vectorizer.get_feature_names())):
        s = vectorizer.get_feature_names().__getitem__(j).encode("utf-8")+" , "+str(count[10,j])+" , "+str(count[17,j])
        print s
    # print (vectorizer.get_feature_names())
    # for word in vectorizer.get_feature_names():
    #     print word
    # count= (vectorizer.fit_transform(article_list).todense())
    # print count[0:2,:]
    # print pairwise_distances(count, metric="cosine")
    # print count[0, :]
    # print count[1, :]
    # print 'here...',cosine_distances([[1,0]], [[0,1]])
    #
    # print count.shape
    # count = (vectorizer.fit_transform(article_list).todense())
    aList = []
    #
    for j in range(0,22):
        for k in range(j+1,22):
            for x,y in [[j,k]]:
                dict = {
                    "start": "",
                    "end": "",
                    "value": 0.0
                }
                # x = j
                # y = 20
                dist = euclidean_distances(count[x,:],count[y,:])
                dist1 = 1 - cosine_distances(count[x,:], count[y,:])
                # print ('文檔{}與文檔{}的歐式距離{}'.format(x,y,dist))
                # print ('文檔{}與文檔{}的cos距離{}'.format(x, y, dist1))
                # print "--------------------------------------------------------------------------"
                # dict["start"] = str(j)
                # dict["end"] = str(k)
                # dict["value"] = dist1
                # aList.append(dict)
    #
    # bList = []
    # while len(aList)>0:
    #     # print len(aList)
    #     minV = -1
    #     j = -1
    #     for  j in range(0,len(aList)):
    #         aItem = aList.__getitem__(j)
    #         aValue = aItem["value"]
    #         if aValue > minV:
    #             minV = aValue
    #             idx = j
    #
    #     if idx>=0:
    #         bList.append(aList.pop(idx))
    #
    # for j in bList:
    #     print ('文檔{}與文檔{}的cos距離{}'.format(j["start"], j["end"], j["value"]))


    # print(vectorizer.vocabulary_)
    # step2:得出結果(0, 3174)	1 (x,y)z x是應該是指在第幾個文本,y應該是指該字詞排列後的索引值,z應該代表該詞的tf-idf值
    tfidf = transformer.fit_transform(vectorizer.fit_transform(article_list))
    # print tfidf

    #vectorizer.get_feature_names()可查看總共有哪些詞
    # 問題：和vectorizer.vocabulary_有何不同？？？？
    wordlist = vectorizer.get_feature_names()
    # print wordlist

    # 利用toaarray()這個方法，將step2所產生的向量轉換為tf-idf矩陣,元素a[i][j]表示j词在i类文本中的tf-idf权重
    weightlist = tfidf.toarray()
    # print weightlist
    # print type(weightlist)   #<type 'numpy.ndarray'>
    # print len(weightlist)    #14

    # #打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
    # for i in range(len(weightlist)):
    #     print "-------這裏輸出的是第",i,"類文本中詞語的if-idf權重------"
    #     for j in range(len(wordlist)):
    #         print wordlist[j],weightlist[i][j]


    # for i, j in sorted(weightlist, key=lambda x: x[1], reverse=True):
    #     print "%-15s %.10f" % (i, j)


    # for i in range(len(weightlist)):
    #     result_dist={}
    #     print "-------這裏輸出的是第",i,"類文本中詞語的if-idf權重------"
    #     for j in range(len(wordlist)):
    #         result_dist[wordlist[j]]=weightlist[i][j]
    #     for k,v in sorted(result_dist.iteritems(), key=lambda x: x[1],reverse=True):
    #         print (k.encode('utf-8')+' '+str(v))
    #


        # tf_idf_file.write(str(i)+'\n')
    #     for j in range(len(wordlist)):
    #         result_dist[wordlist[j]]=weightlist[ii][j]
    #     for k,v in sorted(result_dist.iteritems(), key=lambda x: x[1],reverse=True):
    #         if v>0:
    #             s = k + ' ' + unicode(v)
    #             sList.append(s)
    #             # tf_idf_file.write(k+' '+str(v)+'\n')
    #
    # with open("E:/project/tfidf/Result"+str(start)+"_"+str(end)+".txt","w") as f:
    #     for item in sList:
    #         f.write(item.encode("utf-8")+"\n")
    #     f.close()

# def cosine_distiance(arr1,arr2):
#     numerator=np.dot(arr1,arr2.T)
#     denominator=np.outer(
#         np.sqrt(np.square(arr1).sum(1)),
#         np.sqrt(np.square(arr2).sum(1))
#     )
#     return np.nan_to_num(np.divide(numerator,denominator))

# print cosine_distiance(count[0],count[20])