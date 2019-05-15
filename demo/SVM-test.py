from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

#准备训练样本
x=[[1,8],[3,20],[1,15],[3,35],[5,35],[4,40],[7,80],[6,49]]
y=[1,1,0,0,1,0,0,1]

##开始训练
clf=svm.SVC()  ##默认参数：kernel='rbf'
clf.fit(x,y)

#print("预测...")
#res=clf.predict([[2,2]])  ##两个方括号表面传入的参数是矩阵而不是list

##根据训练出的模型绘制样本点
for i in x:
    res=clf.predict(np.array(i).reshape(1, -1))
    if res > 0:
        plt.scatter(i[0],i[1],c='r',marker='*')
    else :
        plt.scatter(i[0],i[1],c='g',marker='*')

##生成随机实验数据(15行2列)
rdm_arr=np.random.randint(1, 15, size=(15,2))
##回执实验数据点
for i in rdm_arr:
    res=clf.predict(np.array(i).reshape(1, -1))
    if res > 0:
        plt.scatter(i[0],i[1],c='r',marker='.')
    else :
        plt.scatter(i[0],i[1],c='g',marker='.')
##显示绘图结果
plt.show()