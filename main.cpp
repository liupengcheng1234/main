import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import pydotplus
from sklearn.datasets import load_iris
# 花萼长度、花萼宽度，花瓣长度，花瓣宽度
iris_feature_E = ['sepal length','sepal width','petal length', 'petal width']
iris_feature = ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']
iris_class = 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'
if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    data = load_iris()
    x = data.data
    print(x.shape)
    y = data.target
    print(y.shape)
    #仅使用前两列特征
    x = x[:, :2]
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=1)
    # 决策树参数估计
    # min_samples_split = 10：如果该结点包含的样本数目大于10，则(有可能)对其分支
    # min_samples_leaf = 10：若将某结点分支后，得到的每个子结点样本数目都大于10，则完成分支；否则，不进行分支
    model = DecisionTreeClassifier(criterion='entropy')
    model.fit(x_train, y_train)            #训练数据
    y_test_hat = model.predict(x_test)      # 测试数据
    print('accuracy_score:', accuracy_score(y_test, y_test_hat))
        
      # 画图
    N, M = 50, 50  # 横纵各采样多少个值
    x1_min, x2_min = [min(x[:,0]),min(x[:,1])]
    x1_max, x2_max = [max(x[:,0]),max(x[:,1])]
    
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
    x_show = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
    print( x_show.shape)
    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    y_show_hat = model.predict(x_show)  # 预测值
    print( y_show_hat.shape)
    print(y_show_hat)
    y_show_hat = y_show_hat.reshape(x1.shape)  # 使之与输入的形状相同
    print( y_show_hat)
    plt.figure(1, figsize=(10,4), facecolor='w')
    plt.subplot(1, 2, 1)                    # 1行2列的第一张子图
    plt.pcolormesh(x1, x2, y_show_hat, cmap=cm_light)  # 预测值的显示
    plt.scatter(x_test[:,0], x_test[:,1], c=y_test.ravel(), edgecolors='k', s=150, zorder=10,   cmap=cm_dark, marker='*')  # 测试数据
    plt.scatter(x[:,0], x[:,1], c=y.ravel(), edgecolors='k', s=40, cmap=cm_dark)  # 全部数据
    plt.xlabel(iris_feature[0], fontsize=15)
    plt.ylabel(iris_feature[1], fontsize=15)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid(True)
    plt.title('class', fontsize=17)
    plt.show()
     # 训练集上的预测结果
    y_test = y_test.reshape(-1)
    print( y_test_hat)
    print( y_test)
    result = (y_test_hat == y_test) # True则预测正确，False则预测错误
    acc = np.mean(result)
    print( u'准确度: %.2f%%' % (100 * acc))
    # 过拟合：错误率
    depth = np.arange(1, 15)
    err_list = []
    for d in depth:
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=d)
        clf.fit(x_train, y_train)
        y_test_hat = clf.predict(x_test)  # 测试数据
        result = (y_test_hat == y_test)  # True则预测正确，False则预测错误
        err = 1 - np.mean(result)
        err_list.append(err)
        # print d, ' 准确度: %.2f%%' % (100 * err)
        print( d, u' 错误率: %.2f%%' % (100 * err))
    plt.subplot(1, 2, 2)         # 1行2列的第2张子图
    plt.plot(depth, err_list, 'ro-', lw=2)
    plt.xlabel('Depth', fontsize=15)
    plt.ylabel('Error ratio', fontsize=15)
    plt.title('decision tree', fontsize=17)
    plt.grid(True)
    plt.show()
