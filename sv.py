# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from matplotlib import pyplot as plt

#获取数据(不进行标准化处理)
def dataloading1():
    #读取表格文件的第1列到第57列（从第0列第0行开始）第0行到第3999行
    data = pd.read_csv("D:/AI/income.csv",usecols=range(1,58),header=None)#文件“income.csv”保存在D盘的AI文件下。
    print(data.head(5)) #查看读取数据的前五行
    data =  np.array(data)
    #读取表格文件的第58列，也就是数据的标签：0/1
    label = np.array(pd.read_csv("D:/AI/income.csv",usecols=[58],header=None))
    traindata = data[:3000]#取前3000行（从第0行到第2999行作为训练数据）
    trainlabel = label[:3000]#取前3000行的标签
    testdata = data[3000:]#取后1000行作为测试数据
    testlabel = label[3000:]#取后1000行的标签
    return traindata,testdata,trainlabel,testlabel

#获取数据（对特征数据的最后三列进行标准化处理，其他特征数据直接读取）
def dataloading2():
    label = np.array(pd.read_csv("D:/AI/income.csv",usecols=[58],header=None))
    trainlabel = label[:3000]#取前3000行的标签
    testlabel = label[3000:]#取后1000行的标签
    #读取表格文件的第1列到第55列（从第0列第0行开始）,也就是数值本身就在0-1之间的
    data = np.array(pd.read_csv("D:/AI/income.csv",usecols=range(1,55),header=None))
    #读取特征，只读取值比1大的，本就处于0-1之间的数据不进行标准化操作
    data2 = np.array(pd.read_csv("D:/AI/income.csv",usecols=range(55,58),header=None))

    #均值和方差都只针对训练数据。这里写的均值与方差仅适用于这次的模型
    average = np.sum(data2[:3000],axis=0)/3000 #求均值
    averages = np.tile(average,(3000,1))
    variance = ( np.sum ( ( averages-data2[:3000] )**2,axis=0 ) /3000 )**0.5 #求方差（开方后）
    averages = np.tile(average,(4000,1))

    #无论测试数据还是训练数据都需要把数值标准化
    data2 -= averages
    data3 = []
    for i in range(len(variance)):
        if variance[i] == 0:#判断方差是否为0，如果是，则将所有数据置1，否则按标准化公式x=(x-average)/variance
            data3.append(np.ones(4000))
        else:
            data3.append(data2[:,i]/variance[i])
        data = np.column_stack((data,data3[i])) #增加一列特征
    
    traindata = data[:3000]#取前3000行（从第0行到第2999行作为训练数据）
    testdata = data[3000:]#取后1000行作为测试数据
    return traindata,testdata,trainlabel,testlabel

"""
函数参数：
    predicterror:通过sdg的predict_proba函数获得。
                 是m×2的矩阵。每行第一个数据是数据的标签为0的概率，第二个数据是标签为1的概率，总行数m是数据个数。
    testlabel:m×1的矩阵。每行代表一个数据的实际标签。
函数返回值：
    loss:损失函数的值
"""
def loss(predicterror,testlabel):
    m,n = np.shape(testlabel) #m表示数据个数
    loss = 0.0
    # 如果概率比0.00000001(这个值只是随便设了一个概率)小，这时候取对数的结果为-inf,负无穷大，后面的计算无法进行
    # 对这种概率，不妨将这个概率赋值为0.00000001，如果算法分类正确，这个值赋值为多少对损失函数影响不大。
    # 如果算法分类错误，0.00000001取对数结果是-18左右，也比较大，可以用于损失函数计算。
    for i in range(m):
        if predicterror[i][0] < 1e-8:
            log0 = np.log(1e-8)
        else: 
            log0 = np.log(predicterror[i][0])
        if predicterror[i][1] < 1e-8:
            log1 = np.log(1e-8)
        else:
            log1 = np.log(predicterror[i][1])
        loss += (1-testlabel[i])*log0 + testlabel[i]*log1
    loss = (-1)*loss/m
    return loss

"""#求不同学习率时正确率和损失函数随迭代次数的变化情况
函数参数：
    traindata：训练数据的特征数据集。3000×57的矩阵，每行代表一个特征数据向量
    trainlabel：训练数据的实际标签。3000×1的矩阵，每行代表一个数据的标签
    testdata：测试数据的特征数据集。1000×57的矩阵，每行代表一个特征数据向量
    testlabel：测试数据的实际标签。1000×1的矩阵，每行代表一个数据的标签
    eta：学习率
函数返回值：
    iter1：[300,600,1000,1400,1700,2000,2500,3000,3400,3800,4200,4600,5000,5500,6000]
    testloss：15次不同迭代次数下的损失函数值
    rightrate：15次不同迭代次数下的测试数据的正确率
"""
def rightrate_loss(traindata,trainlabel,testdata,testlabel,eta=1e-1):
    rightrate=[]
    testloss=[]
    m,n = np.shape(testdata)
    iter1 = [300,600,1000,1400,1700,2000,2500,3000,3400,3800,4200,4600,5000,5500,6000]
#eta=[1e-1,1e-2,1e-3,1e-4,1e-5]
    for j in range(15):
        # SGDC是随机梯度下降法，loss选择‘log’表示逻辑回归模型，
	    # max_iter是最大迭代次数，eta0是学习率，也就是每次沿梯度更新的程度
        sgdc=SGDClassifier(loss='log',max_iter=iter1[j],eta0=eta)
        sgdc.fit(traindata, np.ravel(trainlabel))#求参数
        rightrate.append(1-sum((sgdc.predict(testdata)-np.ravel(testlabel))**2)/m)#求正确率
        predicterror = sgdc.predict_proba(testdata)#求概率
        loss1 = np.array(loss(predicterror,testlabel))
        testloss.append(loss1)#求损失函数
    return iter1,testloss,rightrate

#图像显示
def display(_ter,rightrate1,rightrate2,rightrate3,rightrate4,rightrate5,testloss1,testloss2,testloss3,testloss4,testloss5):
    plt.figure()
    ax1 = plt.subplot(2,1,1) #第一行第一列,显示不同学习率情况下正确率随迭代次数变化的情况
    ax2 = plt.subplot(2,1,2) #第二行第一列，显示不同学习率情况下测试数据损失函数随迭代次数的变化情况
    plt.sca(ax1)
    plt.plot(_iter,rightrate1,color='skyblue',label='eta0=0.1')
    plt.plot(_iter,rightrate2,color='red',label='eta0=0.01')
    plt.plot(_iter,rightrate3,color='blue',label='eta0=0.001')
    plt.plot(_iter,rightrate4,color='green',label='eta0=0.0001')
    plt.plot(_iter,rightrate5,color='yellow',label='eta0=0.00001')
    plt.ylabel('rightrate')
    plt.legend()
    plt.sca(ax2)
    plt.plot(_iter,testloss1, color='skyblue', label='eta0=0.1')
    plt.plot(_iter,testloss2, color='red', label='eta0=0.01')
    plt.plot(_iter,testloss3, color='blue', label='eta0=0.001')
    plt.plot(_iter,testloss4, color='green', label='eta0=0.0001')
    plt.plot(_iter,testloss5, color='yellow', label='eta0=0.00001')
    plt.xlabel('max_iter')
    plt.ylabel('testdata_loss')
    plt.legend()
    plt.show()

#主函数
#1.数据读取(不进行标准化处理)
traindata,testdata,trainlabel,testlabel = dataloading1()

#2.研究学习率和迭代次数对随机梯度下降算法正确率和测试数据
_iter,testloss1,rightrate1 = rightrate_loss( traindata, trainlabel, testdata, testlabel,eta=1e-1)
_iter,testloss2,rightrate2 = rightrate_loss( traindata, trainlabel, testdata, testlabel,eta=1e-2)
_iter,testloss3,rightrate3 = rightrate_loss( traindata, trainlabel, testdata, testlabel,eta=1e-3)
_iter,testloss4,rightrate4 = rightrate_loss( traindata, trainlabel, testdata, testlabel,eta=1e-4)
_iter,testloss5,rightrate5 = rightrate_loss( traindata, trainlabel, testdata, testlabel,eta=1e-5)
display(_iter,rightrate1,rightrate2,rightrate3,rightrate4,rightrate5,testloss1,testloss2,testloss3,testloss4,testloss5)

#3.尝试提高正确率？数据标准化
# 其实也不知道该怎么提高，在网上找的资料也是说逻辑回归本身正确率就不算太高，
# 然后想到在knn算法中有一个像素点二值化，观察这次的收入表数据，后三列数据范围不是0-1，可以考虑将他们标准化到0-1（正负）
# 采用的方法是z-score，也就是求均值和方差，然后把数据线性变换到0-1
traindata,testdata,trainlabel,testlabel = dataloading2()

# 4.测试标准化的数据的正确率
_iter,testloss1,rightrate1 = rightrate_loss( traindata, trainlabel, testdata, testlabel,eta=1e-1)
_iter,testloss2,rightrate2 = rightrate_loss( traindata, trainlabel, testdata, testlabel,eta=1e-2)
_iter,testloss3,rightrate3 = rightrate_loss( traindata, trainlabel, testdata, testlabel,eta=1e-3)
_iter,testloss4,rightrate4 = rightrate_loss( traindata, trainlabel, testdata, testlabel,eta=1e-4)
_iter,testloss5,rightrate5 = rightrate_loss( traindata, trainlabel, testdata, testlabel,eta=1e-5)
display(_iter,rightrate1,rightrate2,rightrate3,rightrate4,rightrate5,testloss1,testloss2,testloss3,testloss4,testloss5)

#5.求不同正则化参数时正确率和测试数据/训练数据(标准化的数据)的损失函数在较为理想的情况（学习率为0.001，最大迭代次数为1700）的变化情况
rightrate = []
testloss = []
trainloss = []
alpha1 = [0.5,0.1,0.09,0.06,0.01,0.0009,0.0006,0.0003,0.0001,0.00009,0.00005,0.00001]
massage = []
m,n = np.shape(testdata)
print("alpha rightrate testloss trainloss")
for j in range(12):
    # SGDC是随机梯度下降法，loss选择‘log’表示逻辑回归模型，
	# max_iter是最大迭代次数，eta0是学习率，也就是每次沿梯度更新的程度
    sgdc=SGDClassifier(loss='log',alpha=alpha1[j],max_iter=1500,eta0=0.001)
    sgdc.fit(traindata, np.ravel(trainlabel))#求参数
    right = 1-sum((sgdc.predict(testdata)-np.ravel(testlabel))**2)/m #求正确率
    rightrate.append(right)
    predicterror = sgdc.predict_proba(testdata)#求标签的概率
    loss1 = loss(predicterror,testlabel)#测试数据的损失函数
    loss2 = loss(sgdc.predict_proba(traindata),trainlabel)#训练数据的损失函数
    testloss.append(loss1)
    trainloss.append(loss2)
    print([alpha1[j],right,loss1,loss2])
plt.figure()
ax1 = plt.subplot(2,1,1) #第一行第一列,显示正确率随正则化参数变化变化的变化情况
ax2 = plt.subplot(2,1,2) #第二行第一列，显示测试数据/训练数据损失函数随正则化参数的变化情况
plt.sca(ax1)
plt.plot(alpha1,rightrate,color='skyblue',label="test_rightrate")
plt.ylabel('rightrate')
plt.xlim(0, 0.5)
plt.legend()
plt.sca(ax2)
plt.plot(alpha1,testloss, color='skyblue', label='testloss')
plt.plot(alpha1,trainloss, color='red', label='trainloss')
plt.xlabel('alpha')
plt.ylabel('loss')
plt.xlim(0, 0.5)
plt.legend()
plt.show()



















    
