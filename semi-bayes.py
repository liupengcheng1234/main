import numpy as np
import cv2
import matplotlib.pyplot as plt

#"半朴素贝叶斯分类器的数据集设计"
#正态分布数据集

data1_1 = 2.5*(np.random.randn(100,1)+1)
#"利用data1_1产生的数据产生数据data1_2，因此属性之间存在依赖关系"
data1_2 = 15 - data1_1  - (np.random.randn(100,1)+1)
data1 = np.hstack((data1_1,data1_2))
data2_1 = 5*(np.random.randn(100,1)+3)
data2_2 = data2_1 + (np.random.randn(100,1)+1)
data2 = np.hstack((data2_1,data2_2))
plt.plot(data1[:,0],data1[:,1],'ro')
plt.plot(data2[:,0],data2[:,1],'bx')
plt.show()
#为两类数据加入标签1,2
data1 = np.hstack((data1,np.zeros([100,1])+1))
data2 = np.hstack((data2,np.zeros([100,1])+2))

#设计半朴素贝叶斯分类器
#对以上数据离散化
data1_min_1,data1_max_1 = np.min(data1[:,0]),np.max(data1[:,0])
data1_min_2,data1_max_2 = np.min(data1[:,1]),np.max(data1[:,1])
data2_min_1,data2_max_1 = np.min(data2[:,0]),np.max(data2[:,0])
data2_min_2,data2_max_2 = np.min(data2[:,1]),np.max(data2[:,1])
data_min = np.min([data1_min_1,data1_min_2,data2_min_1,data2_min_2])
data_max = np.max([data1_max_1,data1_max_2,data2_max_1,data2_max_2])
#离散值设置
dispersed = 16
t1=np.linspace(data_min,data_max,dispersed+1)

data1_cop = np.zeros(data1.shape)
data2_cop = np.zeros(data2.shape)
data1_cop[:,2]=data1[:,2]
data2_cop[:,2]=data2[:,2]
for i in range(dispersed):
    data1_cop[data1[:,0]>=t1[i],0] = i
    data1_cop[data1[:,1]>=t1[i],1] = i
    data2_cop[data2[:,0]>=t1[i],0] = i
    data2_cop[data2[:,1]>=t1[i],1] = i
#再次绘图，输出离散后的数据值
plt.plot(data1_cop[:,0],data1_cop[:,1],'ro')
plt.plot(data2_cop[:,0],data2_cop[:,1],'bx')
plt.show()

#数据划分，划分为测试集和训练集
train_data =  np.vstack((data1_cop[20:,:],data2_cop[20:,:]))
test_data = np.vstack((data1_cop[:20,:],data2_cop[:20,:]))

num =np.zeros([dispersed,4])
num_rely =np.zeros([dispersed,dispersed,2])
 #半朴素贝叶斯分类器算法设计,假设已经知道其依赖关系
def beyes_function(train_data):
    data1_cop = train_data[0:80,:]
    data2_cop = train_data[80:160,:]
    for i in range(dispersed):
        num[i,0]=np.sum(data1_cop[:,0] == i)
        num[i,1]=np.sum(data1_cop[:,1] == i)
        num[i,2]=np.sum(data2_cop[:,0] == i)
        num[i,3]=np.sum(data2_cop[:,1] == i)
         #计入依赖关系
        for j in range(dispersed):
            num_rely[i,j,0]=np.sum(data1_cop[data1_cop[:,0]==i,1]==j)
            num_rely[i,j,1]=np.sum(data2_cop[data2_cop[:,0]==i,1]==j)
    return num,num_rely

beyes_function(train_data)
a=1
#以上过程训练完，开始测试数据
def test_data_fun(data,flag):
    #已知其属性，计算属于某一类的概率值
    data=[int(data[0]),int(data[1]),int(data[2])]
    L=np.sum(num[:,0])+np.sum(num[:,2])
    class_num =2
    class_first = (num[data[0],0]+1)/(L+class_num *dispersed)
    #计入依赖
    
    if flag == 0:
        L=num[data[0],0]
        class_first = class_first * ((num_rely[data[0],data[1],0])+1)/(L+dispersed)
    else:
        class_first = class_first * (num[data[1],1]+1)/(L+class_num *dispersed)
    #已知其属性，计算属于某类的概率值
    L=np.sum(num[:,0])+np.sum(num[:,2])
    class_second = (num[data[0],2]+1)/(L+class_num *dispersed)
    #计入依赖
    
    if flag==0:
        L=num[data[0],2]
        class_second = class_second * ((num_rely[data[0],data[1],1])+1)/(L+dispersed)
    else:
        class_second = class_second * (num[data[1],3]+1)/(L+class_num *dispersed)
    if class_second >= class_first:
        return 2
    else:
        return 1
result=[]
result_true = test_data[:,2]
for i in range(40):
    data = test_data[i,:]
    result.append(test_data_fun(data,0))
print(result_true)
print(result)

#输出准确率
right_rate = sum(result==result_true)/len(result_true)
print('准确率%f'%right_rate)

result=[]
#采用贝叶斯算法对上述分类
for i in range(40):
    data = test_data[i,:]
    result.append(test_data_fun(data,1))
print(result_true)
print(result)

#输出准确率
right_rate = sum(result==result_true)/len(result_true)
print('准确率%f'%right_rate)





















