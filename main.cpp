# 蚁群算法
import os
import numpy as np  
import matplotlib.pyplot as plt  
os.getcwd()
# 52个城市的坐标
coordinates = np.array([[565.0,575.0],[25.0,185.0],[345.0,750.0],[945.0,685.0],[845.0,655.0],  
[880.0,660.0],[25.0,230.0],[525.0,1000.0],[580.0,1175.0],[650.0,1130.0],  
[1605.0,620.0],[1220.0,580.0],[1465.0,200.0],[1530.0,5.0],[845.0,680.0],  
[725.0,370.0],[145.0,665.0],[415.0,635.0],[510.0,875.0],[560.0,365.0],  
[300.0,465.0],[520.0,585.0],[480.0,415.0],[835.0,625.0],[975.0,580.0],  
[1215.0,245.0],[1320.0,315.0],[1250.0,400.0],[660.0,180.0],[410.0,250.0],  
[420.0,555.0],[575.0,665.0],[1150.0,1160.0],[700.0,580.0],[685.0,595.0],  
[685.0,610.0],[770.0,610.0],[795.0,645.0],[720.0,635.0],[760.0,650.0],  
[475.0,960.0],[95.0,260.0],[875.0,920.0],[700.0,500.0],[555.0,815.0],  
[830.0,485.0],[1170.0, 65.0],[830.0,610.0],[605.0,625.0],[595.0,360.0],  
[1340.0,725.0],[1740.0,245.0]]) 
# 邻接矩阵
def getdistmat(coordinates):  
    num = coordinates.shape[0]  
    distmat = np.zeros((52,52))  
    for i in range(num):  
        for j in range(i,num):  
            distmat[i][j] = distmat[j][i]=np.linalg.norm(coordinates[i]-coordinates[j])  
    return distmat  
distmat = getdistmat(coordinates)  # 邻接矩阵
numant = 40 #蚂蚁个数  
numcity = coordinates.shape[0] #城市个数  
alpha = 1   #信息素重要程度因子  
beta = 5    #启发函数重要程度因子  
rho = 0.1   #信息素的挥发速度  
Q = 1  
iter = 0  
itermax = 20  
# 启发函数矩阵，表示蚂蚁从城市i转移到矩阵j的期望程度
etatable = 1.0/(distmat+np.diag([1e10]*numcity)) 
pheromonetable  = np.ones((numcity,numcity)) # 信息素矩阵  
pathtable = np.zeros((numant,numcity)).astype(int) #路径记录表  
# distmat = getdistmat(coordinates) #城市的距离矩阵
lengthaver = np.zeros(itermax) #各代路径的平均长度  
lengthbest = np.zeros(itermax) #各代及其之前遇到的最佳路径长度  
pathbest = np.zeros((itermax,numcity)) # 各代及其之前遇到的最佳路径长度  
while iter < itermax:  
    # 随机产生各个蚂蚁的起点城市  
    if numant <= numcity:#城市数比蚂蚁数多  
        pathtable[:,0] = np.random.permutation(range(0,numcity))[:numant]  
    else: #蚂蚁数比城市数多，需要补足  
        pathtable[:numcity,0] = np.random.permutation(range(0,numcity))[:]  
        pathtable[numcity:,0] = np.random.permutation(range(0,numcity))[:numant-numcity]  
    length = np.zeros(numant) #计算各个蚂蚁的路径距离  
    for i in range(numant):    
        #i=0  
        visiting = pathtable[i,0] # 当前所在的城市  
        #visited = set() #已访问过的城市，防止重复  
        #visited.add(visiting) #增加元素  
        unvisited = set(range(numcity))#未访问的城市  
        unvisited.remove(visiting) #删除元素  
        for j in range(1,numcity):#循环numcity-1次，访问剩余的numcity-1个城市  
            #j=1  
            #每次用轮盘法选择下一个要访问的城市  
            listunvisited = list(unvisited)    
            probtrans = np.zeros(len(listunvisited))    
            for k in range(len(listunvisited)):  
                probtrans[k] = np.power(pheromonetable[visiting][listunvisited[k]],alpha)\
                               *np.power(etatable[visiting][listunvisited[k]],alpha)  
            cumsumprobtrans = (probtrans/sum(probtrans)).cumsum()  
            cumsumprobtrans -= np.random.rand()  
            # k = listunvisited[find(cumsumprobtrans>0)[0]] #下一个要访问的城市
            k = listunvisited[list(cumsumprobtrans>0).index(True)] #下一个要访问的城市  
            pathtable[i,j] = k  
            unvisited.remove(k)  
            #visited.add(k)  
            length[i] += distmat[visiting][k]  
            visiting = k  
    #蚂蚁的路径距离包括最后一个城市和第一个城市的距离
        length[i] += distmat[visiting][pathtable[i,0]]    
    #print length  
    # 包含所有蚂蚁的一个迭代结束后，统计本次迭代的若干统计参数    
    lengthaver[iter] = length.mean()  
    if iter == 0:  
        lengthbest[iter] = length.min()  
        pathbest[iter] = pathtable[length.argmin()].copy()        
    else:  
        if length.min() > lengthbest[iter-1]:  
            lengthbest[iter] = lengthbest[iter-1]  
            pathbest[iter] = pathbest[iter-1].copy()  
        else:  
            lengthbest[iter] = length.min()  
            pathbest[iter] = pathtable[length.argmin()].copy()      
    # 更新信息素  
    changepheromonetable = np.zeros((numcity,numcity))  
    for i in range(numant):  
        for j in range(numcity-1):  
            changepheromonetable[pathtable[i,j]][pathtable[i,j+1]] += Q/distmat[pathtable[i,j]][pathtable[i,j+1]]  
        changepheromonetable[pathtable[i,j+1]][pathtable[i,0]] += Q/distmat[pathtable[i,j+1]][pathtable[i,0]]  
    pheromonetable = (1-rho)*pheromonetable + changepheromonetable 
      iter += 1 #迭代次数指示器+1  
# 做出平均路径长度和最优路径长度          
fig,axes = plt.subplots(nrows=2,ncols=2,figsize=(12,10))  
axes[0,0].plot(lengthaver,'k',marker = u'')  
axes[0,0].set_title('Average Length')  
axes[0,0].set_xlabel(u'iteration')  
axes[1,0].plot(lengthbest,'k',marker = u'')  
axes[1,0].set_title('Best Length')  
axes[1,0].set_xlabel(u'iteration')  
#fig.savefig('Average_Best.png',dpi=500,bbox_inches='tight')
#plt.show()
#plt.close()  
#作出找到的最优路径图  
bestpath = pathbest[-1]  
axes[0,1].plot(coordinates[:,0],coordinates[:,1],'r.',marker=u'$\cdot$')  
axes[0,1].xaxis.limit_range_for_scale(-100,2000)  
axes[0,1].yaxis.limit_range_for_scale(-100,1500)  
for i in range(numcity-1):#  
    m,n = int(bestpath[i]),int(bestpath[i+1])
    print(m,n)
    axes[0,1].plot([coordinates[m][0],coordinates[n][0]],[coordinates[m][1],coordinates[n][1]],'k')
axes[0,1].plot([coordinates[ int(bestpath[0]) ][0],coordinates[n][0]],[coordinates[ int(bestpath[0]) ][1],coordinates[n][1]],'b')  
#ax=plt.gca()  
axes[0,1].set_title("Best Path")  
axes[0,1].set_xlabel('X axis')  
axes[0,1].set_ylabel('Y_axis')  
#plt.savefig('Best Path.png',dpi=500,bbox_inches='tight')
plt.show()
#plt.close()
