# -*- coding: utf-8 -*-
import os
import codecs
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

document_list=[]
uri = "mongodb://10.120.37.128:27017"
client = MongoClient(uri)
db = client.__getitem__("family")
collection = db.__getitem__("travel2")
for item in collection.find():
    document_list.append(item["content2"])

vectorizer = CountVectorizer(binary=True)
transformer = TfidfTransformer()

tfidf = transformer.fit_transform(vectorizer.fit_transform(document_list))
print type(tfidf[0][0])
print str(tfidf[0][0])

wordlist = vectorizer.get_feature_names()
# weightlist = tfidf.toarray()

for i in range(70248):
    file_name2='E:/project/tfidf1/{}'.format(i)+'.txt'
    with codecs.open(file_name2, "w", 'utf-8') as tf_idf_file:
        result_dist={}
        # print "-------這裏輸出的是第",i,"類文本中詞語的if-idf權重------"
        for j in range(len(wordlist)):
            print len(tfidf[i][j])
            # result_dist[wordlist[j]]=float(s.split("\t")[1].split("(")[0])
        # for k,v in sorted(result_dist.iteritems(), key=lambda x: x[1],reverse=True):
        #     if v>0:
        #         tf_idf_file.write(k+' '+str(v)+'\n')
        # tf_idf_file.close()
print "All  Finish !!"