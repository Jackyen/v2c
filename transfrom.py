# encoding=utf-8


import word2vec

import numpy as np
# 視覺化套件
import matplotlib
import matplotlib.pyplot as plt
# 主成分因子
from sklearn.decomposition import PCA

model = word2vec.load('corpusWord2Vec.bin')


# 選擇你想丟進去的字詞
# 顯示空間距離相近的詞
# 放入字詞: '寶寶'


indexes = model.cosine( u'經濟' )
for index in indexes[0]:
    print model.vocab[index]


# 在高維向量空間(k=300)找出與所選字詞距離最接近的前10名
index1,metrics1 = model.cosine(u'經濟')
index2,metrics2 = model.cosine(u'成長')
index3,metrics3 = model.cosine(u'食物')
index4,metrics4 = model.cosine(u'村民')
index5,metrics5 = model.cosine(u'幼兒園')
 
# 所選字詞
index01 = np.where(model.vocab == u'經濟')
index02 = np.where(model.vocab == u'成長')
index03 = np.where(model.vocab == u'食物')
index04 = np.where(model.vocab == u'村民')
index05 = np.where(model.vocab == u'幼兒園')
# 將所選字詞與其最接近之前10名合併 
index1 = np.append(index1,index01)
index2 = np.append(index2,index02)
index3 = np.append(index3,index03)
index4 = np.append(index4,index04)
index5 = np.append(index5,index05)

# 引入上述將文章斷詞後轉為300維向量的資料
rawWordVec = model.vectors

# 將原本300維向量空間降為2維
X_reduced = PCA(n_components=2).fit_transform(rawWordVec)


# 須先下載wqy-microhei.ttc，因中文顯示需做特殊處理
zhfont = matplotlib.font_manager.FontProperties(fname='./wqy-microhei.ttc')
# 畫圖
fig = plt.figure()
ax = fig.add_subplot(111)
 
for i in index1:
      ax.text(X_reduced[i][0],X_reduced[i][1],model.vocab[i], fontproperties=zhfont,color='C3')
for i in index2:
      ax.text(X_reduced[i][0],X_reduced[i][1],model.vocab[i], fontproperties = zhfont,color= 'C1')
for i in index3:
      ax.text(X_reduced[i][0],X_reduced[i][1],model.vocab[i], fontproperties=zhfont,color='C7')
for i in index4:
      ax.text(X_reduced[i][0],X_reduced[i][1],model.vocab[i], fontproperties=zhfont,color='C0')
for i in index5:
      ax.text(X_reduced[i][0],X_reduced[i][1],model.vocab[i], fontproperties=zhfont,color='C4')
ax.axis([0,0.5,-0.2,0.6])
plt.figure(figsize=(60,60))
plt.show()


