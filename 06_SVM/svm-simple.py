#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# https://blog.csdn.net/c406495762/article/details/78072313
import numpy as np
import matplotlib.pyplot as plt
import random

"""读取数据"""
def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        dataArr = line.strip().split('\t')
        dataMat.append([float(dataArr[0]), float(dataArr[1])])
        labelMat.append(float(dataArr[-1]))
    return dataMat, labelMat


"""数据可视化"""
def showDataSet(dataMat, labelMat):
    data_plus = []  # 正样本
    data_minus = []  # 负样本
    # 遍历所以数据集，若标签大于0，加入到正样本，反之负样本
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    # 转化为numpy矩阵
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    # 画散点图  转置 行变为列后，第0行为第一个特征值，第1行为第二个特征值
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1])

    plt.show()


"""随机选择alpha j"""
def selectJrand(i, m):  # i为第一个alpha下标，m为alpha个数
    # 另j = i，判断此条件相等时在0到m中随机选择一个数
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


"""修剪alpha"""
def clipAlpha(aj, H, L):
    # 将aj范围控制在L-H中
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj


"""简化版SMO算法"""
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):  # C松弛函数，toler容错率，maxIter最大迭代次数
    # 将x,y转换为numpy的mat存储
    dataMatrix = np.mat(dataMatIn); labelMat = np.mat(classLabels).transpose()
    # 初始化b参数，统计dataMatrix维度
    b = 0
    m, n = np.shape(dataMatrix)
    # alpha参数初始化为(m,1)的mat
    alphas = np.mat(np.zeros((m, 1)))
    # 初始化迭代次数,记录没有alpha改变时的迭代次数
    iter_num = 0
    # 迭代matIter次
    while iter_num < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            # 步骤1：计算误差Ei
            # np.multiply(alphas * labelMat) 100*1 X 100*1对应元素相乘得到和相乘矩阵大小一致的矩阵，即得到100*1
            # dataMatrix * dataMatrix[i, :].T  dataMatrix为100*2，dataMatrix[i, :]为1*2，所以把后者转置为2*1，相乘得到100*1的矩阵
            # 所以再把np.multiply(alphas * labelMat)整体转置为1*100  最终为1*1的数
            fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMat[i])
            # 优化alpha,更设定一定的容错率
            if((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and(alphas[i] > 0)):
                # 随机选择另一个与alpha_i成对优化的alpha_j
                j = selectJrand(i, m)
                # 步骤1：计算误差Ej
                fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                # 保存更新前的alpha值，使用深拷贝
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy()
                # 步骤2：计算上下界L和H
                if labelMat[i] != labelMat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H: print('L==H'); continue
                # 步骤3：计算eta
                eta = dataMatrix[i, :]*dataMatrix[i, :].T + dataMatrix[j, :]*dataMatrix[j, :].T - 2.0*dataMatrix[i, :]*dataMatrix[j, :].T
                # eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - dataMatrix[j,:] * dataMatrix[j, :].T
                if eta <= 0: print('eta<=0'); continue
                # 步骤4：更新alpha_j
                alphas[j] += labelMat[j]*(Ei - Ej)/eta
                # 步骤5：修剪alpha_j
                alphas[j] = clipAlpha(alphas[j], H, L)
                if abs(alphas[j]-alphaJold) < 0.00001: print('alpha_j变化太小'); continue
                # 步骤6：更新alpha_i
                alphas[i] += labelMat[i]*labelMat[j]*(alphaJold-alphas[j])
                # 步骤7：更新b1 b2
                b1 = b - Ei - labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                # 步骤8：根据b1 b2更新b
                if (alphas[i] > 0) and (alphas[i] < C): b = b1
                elif (alphas[j] > 0) and (alphas[j] < C): b = b2
                else: b = (b1+b2)/2.0
                # 统计优化次数
                alphaPairsChanged += 1
                # 打印统计信息
                print('第%d次迭代 样本：%d,alpha优化次数:%d'%(iter_num,i,alphaPairsChanged))
        # 更新迭代次数
        if(alphaPairsChanged == 0):iter_num += 1
        else:iter_num = 0
        print('迭代次数：%d'%iter_num)
    return b, alphas


"""计算w"""
def get_w(dataMat, labelMat, alphas):
    alphas,dataMat,labelMat = np.array(alphas), np.array(dataMat), np.array(labelMat)
    # labelMat.reshape(1, -1).T  将labelMat转为1行不管多少列的数组（这里为100列），再转置变为 100*1
    # np.tile(labelMat.reshape(1, -1).T, (1, 2))  将上面100*1的数组扩为1行，2列  即100*2
    # 与dataMat对应位置相乘，得到的np.tile(labelMat.reshape(1, -1).T, (1, 2)) * dataMat 还是100*2
    # 又alpha为100*1  所以将上式转置为2*100  使用.dot执行矩阵乘法 2*100  X  100*1  得到一个w为 2*1 的数组
    w = np.dot((np.tile(labelMat.reshape(1, -1).T, (1, 2)) * dataMat).T, alphas)
    return w.tolist()


"""分类结果可视化"""
def showClassifer(dataMat, w, b):
    data_plus = []  # 正样本
    data_minus = []  # 负样本
    # 遍历所以数据集，若标签大于0，加入到正样本，反之负样本
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    # 转化为numpy矩阵
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    # 画散点图  转置 行变为列后，第0行为第一个特征值，第1行为第二个特征值
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7)
    # 绘制直线
    x1 = max(dataMat)[0]
    x2 = min(dataMat)[0]
    a1 = w[0][0]
    a2 = w[1][0]
    b = float(b)
    # 直线的一般形式ax+by+c=0
    y1 = (-b - a1*x1)/a2
    y2 = (-b - a1*x2)/a2
    plt.plot([x1, x2], [y1, y2])
    # 找出支持向量点 enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 0:  # 只有支持向量的点才有alpha值
            x1, x2 = dataMat[i]
            plt.scatter([x1], [x2], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolors='red')
    plt.show()











if __name__ == '__main__':
    dataMat, labelMat = loadDataSet('testSet.txt')
    b, alphas = smoSimple(dataMat, labelMat, 0.6, 0.001, 40)
    w = get_w(dataMat, labelMat, alphas)
    showClassifer(dataMat, w, b)











