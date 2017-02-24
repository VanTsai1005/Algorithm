# coding:utf-8
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import os
import unicodedata
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
from pymongo import MongoClient

def cos_method(sList):
    dict = {}
    dict["rec8"] = [u"溜滑梯", u"小溜滑梯", u"大型溜滑梯", u"溜滑梯球池", u"小型溜滑梯", u"溜滑梯民宿", u"氣墊溜滑梯", u"旋轉溜滑梯"]
    dict["rec9"] = [u"玩具", u"小玩具", u"益智玩具", u"玩具車", u"玩具區", u"玩具組", u"木製玩具", u"桌上玩具"]
    dict["rec20"] = [u"盪鞦韆", u"鞦韆", u"盪秋千", u"搖搖馬", u"搖搖木馬", u"搖馬", u"健身器材", u"健身"]
    dict["rec5"] = [u"玩水", u"水池", u"戲水", u"戲水池", u"戲水區", u"水區", u"噴水池", u"玩水區"]
    dict["rec2"] = [u"動物", u"小動物", u"動物園", u"野生動物", u"小動物區", u"動物餵食", u"野生動物園", u"動植物"]
    dict["rec15"] = [u"草地", u"草皮", u"草地上", u"滑草", u"草原", u"大片草地", u"小草原", u"大草原"]
    dict["rec3"] = [u"博物館", u"美術館", u"探索館", u"科學教育", u"教育館", u"藝術館", u"歷史文物", u"科教館"]
    dict["rec10"] = [u"玩沙區", u"公園", u"戲沙池", u"沙池", u"決明子沙坑", u"決明子沙池", u"沙地", u"大沙坑"]
    dict["rec11"] = [u"球池", u"氣球", u"玩球", u"汽球", u"大球池", u"球池區", u"溜滑梯球池", u"小球池"]
    dict["rec13"] = [u"3D彩繪", u"塗鴉", u"塗鴉區", u"畫畫", u"塗鴉牆", u"繪畫", u"畫牆", u"畫紙"]
    dict["rec19"] = [u"下午茶", u"紅茶", u"蛋糕", u"沙拉", u"餅乾", u"冰淇淋", u"生日", u"生日趴"]
    dict["rec1"] = [u"停車", u"免費停車場", u"地下停車場", u"收費停車場", u"停車場", u"公有停車場", u"付費停車場", u"停車免費"]
    dict["rec14"] = [u"腳踏車", u"騎單車", u"綠色隧道", u"單車", u"自行車", u"腳踏車", u"綠色隧道", u"單車道"]
    dict["rec18"] = [u"遊樂設施", u"遊樂園", u"遊樂場", u"主題樂園", u"主題園", u"娛樂設施", u"湯姆熊", u"遊樂設備"]
    dict["rec6"] = [u"手作教室", u"文創園區", u"手作活動", u"捏陶", u"捏黏土", u"鬆餅diy", u"體驗館", u"手作活動"]
    dict["rec16"] = [u"工廠", u"觀光工廠", u"文創", u"文創園區", u"文創體驗館", u"工房", u"襪仔王觀光工廠", u"蜡藝觀光工廠"]
    dict["rec4"] = [u"哺乳室", u"哺乳", u"尿布", u"尿布台", u"換尿布", u"尿布檯", u"有尿布台", u"換尿布區"]
    dict["rec12"] = [u"登山步道", u"森林浴", u"山林", u"原始森林", u"森林生態區", u"自然風情景觀餐廳", u"森林步道", u"散步道"]
    dict["rec7"] = [u"海邊", u"海洋館", u"海灘", u"海洋主題", u"海水浴場", u"海洋", u"大海", u"海濱"]
    dict["rec17"] = [u"農場", u"休閒農場", u"大農場", u"休閒農莊", u"有機農場", u"有機菜園", u"烤肉", u"菜園"]
    # sList = inputString.split(",")
    usrData = []
    usrAry = np.zeros((1, 6231))
    for i in sList:
        filename="rec{}".format(i)
        usrData.append(dict[filename])
        if len(usrData)==5:
            break

    path = "E:/project/wordcount_final/"
    files = [unicodedata.normalize('NFC', f) for f in os.listdir(path.decode("utf-8"))]
    document_list = []
    name_list = []
    for file in files:
        ss = ""
        name_list.append(file.split(".")[0])
        with open(path + file, "r") as f:
            words = f.readlines()
            for word in words:
                ss += word.split(" ")[0].strip() + " "
            f.close()
            document_list.append(ss)

    vectorizer = CountVectorizer()
    # transformer = TfidfTransformer()
    count = (vectorizer.fit_transform(document_list).todense())
    words = vectorizer.get_feature_names()

    for alist in usrData:
        for s in alist:
            for idx, word in enumerate(words):
                if word == s:
                    usrAry[0,idx]=1

    dict={}
    for i in range(len(count)):
        # dist = euclidean_distances(count[i,:], usrAry)
        dist1 = 1 - cosine_distances(count[i,:], usrAry)
        if dist1 > 0:
            dict[i] = dist1

    sortDict = sorted(dict.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)

    dict = {}
    with open("E:/project/namegid.txt" ,"r") as f:
        for line in f.readlines():
            name = line.strip().split(" &")[0].split(":")[1].decode("utf-8").strip()
            gid = line.split("gid:")[1].strip()
            dict[name] = int(gid)
        f.close()

    n = 0
    recommendList = []
    for item in sortDict:
        if n < 10:
            name = name_list[item[0]]
            recommendList.append(dict[name])
            print name_list[item[0]]
            n += 1



    # uri = "mongodb://13.112.64.212:27017"
    # client = MongoClient(uri)
    # db = client.__getitem__("g3site")
    # collection = db.__getitem__("g3site_travel")
    # recList = []
    # for name in recommendList:
    #     for item in collection.find():
    #         if name == item["name"]:
    #             recList.append(item["gid"])
    #             break
    return recommendList

sList = [13,16,18,19,20]
print cos_method(sList)