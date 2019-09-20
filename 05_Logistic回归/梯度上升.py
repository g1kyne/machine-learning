#!/usr/bin/env python
# -*- coding:utf-8 -*-
# https://blog.csdn.net/c406495762/article/details/77723333
import numpy as np
import matplotlib.pyplot as plt
import random

from matplotlib.font_manager import FontProperties

"""梯度上升算法测试函数
    Xi+1 = Xi + alpha * f(x)导数"""
def Gradient_Ascent_Test():
    def f_prime(x_old):  # f(x)的导数
        return -2 * x_old + 4
    x_old = -1  # 给一个计算导数的初始值x，比new小
    x_new = 0  # 从0开始迭代
    alpha = 0.01
    precision = 0.00000001
    while abs(x_new - x_old) > precision:
        x_old = x_new
        x_new = x_old + alpha * f_prime(x_old)
    print(x_new)


"""加载数据  数据第一二列为第一二个特征 第三列为分类标签
    -0.017612	14.053064	0
    -1.395634	4.662541	1"""
def loadDataSet():
    dataList = []
    labelList = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        arrList = line.strip().split()  # 去回车且按空格分割
        dataList.append([1.0, float(arrList[0]), float(arrList[1])])
        labelList.append(int(arrList[2]))
    fr.close()
    return dataList, labelList


"""Sigmoid函数"""
def sigmoid(intX):
    return 1.0 / (1+np.exp(-intX))


"""梯度上升算法找到最佳回归系数  P78页"""
def gradAscent(dataList, labelList):
    dataMat = np.mat(dataList)
    labelMat = np.mat(labelList).transpose()
    m, n = dataMat.shape
    weights = np.ones((n, 1))  # 回归系数矩阵初始化为1
    weights_arr = np.array([])  # 用于画图
    alpha = 0.001
    R = 500  # 重复次数
    for i in range(R):
        h = sigmoid(dataMat * weights)
        error = labelMat - h
        weights = weights + alpha * dataMat.transpose() * error
        weights_arr = np.append(weights_arr, weights)
    weights_arr = weights_arr.reshape(R, n)
    return weights.getA(), weights_arr


"""随机梯度上升算法：一次仅用一个样本点来更新回归系数"""
def stocGradAscent0(dataMat, labelList):
    m, n = dataMat.shape
    weights = np.ones(n)
    alpha = 0.01
    for i in range(m):
        h = sigmoid(sum(dataMat[i] * weights))
        error = labelList[i] - h
        weights = weights + alpha * error * dataMat[i]
    return weights


"""改进的随机梯度上升算法：新增迭代次数，随机选择一个样本"""
def stocGradAscent1(dataMat, labelList, numIter=150):  # 新增迭代次数
    m, n = dataMat.shape
    weights = np.ones(n)  # [1,1,1]
    weights_arr = np.array([])
    for j in range(numIter):  # 迭代numIter次
        dataIndex = list(range(m))  # 生成0到99的列表
        for i in range(m):  # 遍历每个样本
            alpha = 4/(1.0 + j + i) + 0.01 # 降低alpha大小
            randIndex = int(random.uniform(0, len(dataIndex)))  # 随机生成一个0到未被选择列表长度范围的数作为索引
            h = sigmoid(sum(dataMat[randIndex] * weights))  # 计算该样本的h
            error = labelList[randIndex] - h
            weights = weights + alpha * error * dataMat[randIndex]  # 迭代出weights
            weights_arr = np.append(weights_arr, weights, axis=0)  # 在列方向上添加
            del(dataIndex[randIndex])  # 删除已经被选到的值
    weights_arr = weights_arr.reshape(numIter*m, n)
    return weights, weights_arr


"""绘制数据集"""
def plotDataSet():
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)  # 转化为numpy数组
    n = np.shape(dataMat)[0]  # 样本个数
    xcord1 = []; ycord1 = []  # 正样本
    xcord2 = []; ycord2 = []  # 负样本
    for i in range(n):
        if int(labelMat[i]) == 1:  # 1为正样本
            xcord1.append(dataArr[i, 1])  # 横坐标，即第一个特征值 ：数组的第i行第0列
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)  # 添加子图
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='s', alpha=.5)  # 绘制正样本的散点图
    ax.scatter(xcord2, ycord2, s=20, c='green', alpha=.5)  # 绘制负样本
    plt.title('DataSet')  # 绘制title
    plt.xlabel('x1')
    plt.ylabel('x2')  # 绘制label
    plt.show()  # 显示


"""绘制数据集并拟合最佳回归分割线"""
def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)  # 转化为numpy数组
    n = np.shape(dataMat)[0]  # 样本个数
    xcord1 = []; ycord1 = []  # 正样本
    xcord2 = []; ycord2 = []  # 负样本
    for i in range(n):
        if int(labelMat[i]) == 1:  # 1为正样本
            xcord1.append(dataArr[i, 1])  # 横坐标，即第一个特征值 ：数组的第i行第0列
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)  # 添加子图
    # 画点
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='s', alpha=.5)  # 绘制正样本的散点图
    ax.scatter(xcord2, ycord2, s=20, c='green', alpha=.5)  # 绘制负样本
    # 画线
    x1 = np.arange(-3.0, 3.0, 0.1)
    x2 = (-weights[0] - weights[1] * x1) / weights[2]  # 0 = w0x0 + w1x1 + w2x2
    ax.plot(x1, x2)
    plt.title('BestFit')  # 绘制title
    plt.xlabel('x1')
    plt.ylabel('x2')  # 绘制label
    plt.show()  # 显示


"""绘制迭代次数与回归系数关系"""
def plotWeights(weights_array1,weights_array2):
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    # 将画布分为3行2列的子图，axs[0][0]表示第一行第一列的子图，不共享xy轴，每个图大小为20*10
    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False, figsize=(20, 10))
    x1 = np.arange(0, len(weights_arr1), 1)
    # 绘制w0与迭代次数关系
    axs[0][0].plot(x1, weights_arr1[:, 0])
    axs[0][0].set_title(u'改进的梯度上升算法：回归系数与迭代次数关系', FontProperties=font)
    axs[0][0].set_ylabel(u'W0', FontProperties=font)
    # 绘制W1与迭代次数关系
    axs[1][0].plot(x1, weights_arr1[:, 1])
    axs[1][0].set_ylabel(u'W1', FontProperties=font)
    # 绘制W2与迭代次数关系
    axs[2][0].plot(x1, weights_arr1[:, 2])
    axs[2][0].set_ylabel(u'W2', FontProperties=font)
    axs[2][0].set_xlabel(u'迭代次数', FontProperties=font)

    x2 = np.arange(0, len(weights_arr2), 1)
    # 绘制w0与迭代次数关系
    axs[0][1].plot(x2, weights_arr2[:, 0])
    axs[0][1].set_title(u'梯度上升算法：回归系数与迭代次数关系', FontProperties=font)
    axs[0][1].set_ylabel(u'W0', FontProperties=font)
    # 绘制W1与迭代次数关系
    axs[1][1].plot(x2, weights_arr2[:, 1])
    axs[1][1].set_ylabel(u'W1', FontProperties=font)
    # 绘制W2与迭代次数关系
    axs[2][1].plot(x2, weights_arr2[:, 2])
    axs[2][1].set_ylabel(u'W2', FontProperties=font)
    axs[2][1].set_xlabel(u'迭代次数', FontProperties=font)

    plt.show()

if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    # print(dataMat)
    # [[1.0, -0.017612, 14.053064], [1.0, -1.395634, 4.662541],...]
    # plotDataSet()
    weights2, weights_arr2 = gradAscent(np.array(dataMat), labelMat)
    weights1, weights_arr1 = stocGradAscent1(np.array(dataMat), labelMat)
    plotWeights(weights_arr1, weights_arr2)














