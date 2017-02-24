# -*- coding: utf-8 -*-
import sys
import pymongo
import codecs
import json
from pymongo import MongoClient
from collections import OrderedDict
from datetime import datetime
from threading import Thread
from Queue import Queue
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np




s1 = datetime.now()

document_list=[]
for document_no in range(70248):
    # print document_no
    file_name="/Users/mac/PycharmProjects/untitled1/project/document_file/{}".format(document_no)+".txt"
    myfile=codecs.open(file_name,"rb",'utf-8')
    readfiles=myfile.read()
    document_list.append(readfiles)
    document_no = document_no + 1
    myfile.close()

vectorizer=CountVectorizer(binary=True)
transformer = TfidfTransformer()

tfidf = transformer.fit_transform(vectorizer.fit_transform(document_list))
wordlist = vectorizer.get_feature_names()
weightlist = tfidf.toarray()



print len(weightlist) #70248

# for i in range():
for i in range(len(weightlist)):
    file_name2='/Users/mac/PycharmProjects/untitled1/project/tfidf_document2/{}'.format(i)+'.txt'
    with codecs.open(file_name2, "w", 'utf-8') as tf_idf_file:
        result_dist={}
        # print "-------這裏輸出的是第",i,"類文本中詞語的if-idf權重------"
        for j in range(len(wordlist)):
            result_dist[wordlist[j]]=weightlist[i][j]
        for k,v in sorted(result_dist.iteritems(), key=lambda x: x[1],reverse=True):
            if v>0:
                # print (k.encode('utf-8') + ' ' + str(v))
                tf_idf_file.write(k+' '+str(v)+'\n')
        tf_idf_file.close()

s2 = datetime.now()
print "All  Finish - " + str(s2 - s1) + "!!"
