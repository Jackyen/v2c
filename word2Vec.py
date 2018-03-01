# This Python file uses the following encoding: utf-8

import sys
sys.path.append(r'/usr/local/lib/python2.7/site-packages')

import jieba
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from hanziconv import HanziConv
import word2vec

# 讀檔
fileTrainRead = []
with open('corpus.txt') as fileTrainRaw:
  for line in fileTrainRaw:
      fileTrainRead.append(HanziConv.toTraditional(line)) # 簡轉繁

# 斷詞
fileTrainSeg=[]
for i in range(len(fileTrainRead)):
    fileTrainSeg.append([' '.join(list(jieba.cut(fileTrainRead[i][9:-11],cut_all=False)))])
# 因為會跑很久，檢核是否資料有持續在跑
    if i % 50000 == 0 :
        print i

# 精確模式、同時也是預設模式
seg_list1 = jieba.cut("一是嬰兒哭啼二是學遊戲三是青春物語四是碰巧遇見你", cut_all=False)
print "Default Mode: " + "/ ".join(seg_list1) + " No."  
# Result
# 一是/ 嬰兒/ 哭啼/ 二是/ 學遊戲/ 三/ 是/ 青春/ 物語/ 四/ 是/ 碰/ 巧遇/ 見/ 你

# 將jieba的斷詞產出存檔
fileSegWordDonePath ='corpusSegDone.txt'
with open(fileSegWordDonePath,'wb') as fW:
    for i in range(len(fileTrainSeg)):
        fW.write(fileTrainSeg[i][0].encode('utf-8'))
        fW.write('\n')

# 檢視斷詞jieba的結果
def PrintListChinese(list):
    for i in range(len(list)):
        print list[i],

PrintListChinese(fileTrainSeg[10])
#將字集轉成向量檔
word2vec.word2vec('corpusSegDone.txt', 'corpusWord2Vec.bin', size=300,verbose=True)

