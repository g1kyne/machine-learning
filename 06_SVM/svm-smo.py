#!/usr/bin/env python
# -*- coding:utf-8 -*-
# https://blog.csdn.net/c406495762/article/details/78158354
import numpy as np
import matplotlib.pyplot as plt
import random


"""数据结构，维护所有需要操作的值"""
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn  # 数据矩阵
        self.labelMat = classLabels  # 数据标签
        self.C = C  # 松弛变量
        self.tol = toler  # 容错率
        self.m = np.shape(dataMatIn)[0]  # 数据矩阵行数
        self.alphas = np.mat(np.zeros((self.m, 1)))  # 根据矩阵行数初始化alpha参数为0
        self.b = 0  # 初始化b参数为0
        self.eCache = np.mat(np.zeros((self.m, 2)))  # 根据矩阵行数初始化虎误差缓存，第一列为是否有效的标志位，第二列为实际的误差E的值。


"""读取数据"""
def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        dataArr = line.strip().split('\t')
        dataMat.append([float(dataArr[0]), float(dataArr[1])])
        labelMat.append(float(dataArr[-1]))
    return dataMat, labelMat


"""计算误差"""
def calcEk(oS, k):  # oS：数据结构   k:标号为k的数据
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T) + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek


"""随机选择alpha j"""
def selectJrand(i, m):  # i为第一个alpha下标，m为alpha个数
    # 另j = i，判断此条件相等时在0到m中随机选择一个数
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


"""内循环启发方式选择j"""
def selectJ(i, oS, Ei):  # i:标号为i的数据的索引值  Ei:标号为i的数据误差
    maxK = -1; maxDeltaE = 0; Ej = 0  # 找到使|Ei-Ej|最大的样本
    oS.eCache[i] = [1, Ei]  # 1表示有效,将Ei存入缓存误差
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]  # 返回误差不为0的数据的索引值   .A 为把矩阵转为数组
    if len(validEcacheList) > 1:  # 若存在不为0的误差
        for k in validEcacheList:  # 遍历，找到最大的误差Ek
            if k == i: continue  # 不计算i,节省时间
            Ek = calcEk(oS, k)  # 计算Ek
            deltaE = abs(Ei - Ek)  # 计算|Ei-Ek|
            if deltaE > maxDeltaE:  # 找到最大的deltaE
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:  # 若没有不为0的误差
        j = selectJrand(i,oS.m)  # 随机选择一个样本
        Ej = calcEk(oS, j)
    return j, Ej


"""计算Ek,并更新误差缓存"""
def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]  # 更新误差缓存


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


"""修剪alpha"""
def clipAlpha(aj, H, L):
    # 将aj范围控制在L-H中
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj


"""优化版SMO算法"""
def innerL(i, oS):
    """
    :param i: 标号为i的数据的索引值
    :param oS:数据结构
    :return:1：有任意一堆alpha值发生变化  0：没有任意一对alpha值发生变化或变化太小
    """
    # 步骤1：计算误差Ei
    Ei = calcEk(oS, i)
    # 优化alpha,设定一定的容错率
    if (oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C) or (oS.labelMat[i] * Ei > oS.tol):
        # 使用内循环启发方式2选择alpha_j,并计算Ej
        j, Ej = selectJ(i, oS, Ei)
        # 保存更新前的alpha值，使用深拷贝
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy()
        # 步骤2：计算上下界L和H
        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0, oS.alphas[j]-oS.alphas[i])
            H = min(oS.C, oS.C+oS.alphas[j]-oS.alphas[i])
        else:
            L = max(0, oS.alphas[j]+oS.alphas[i]-oS.C)
            H = min(oS.C, oS.alphas[j]+oS.alphas[i])
        if L == H:
            print('L==H')
            return 0
        # 步骤3：计算eta
        eta = oS.X[i, :]*oS.X[i, :].T + oS.X[j, :]*oS.X[j, :].T - 2.0*oS.X[i, :]*oS.X[j, :].T
        if eta <= 0:
            print('eta<=0')
            return 0
        # 步骤4：更新alpha_j
        oS.alphas[j] += oS.labelMat[j] * (Ei - Ej)/eta
        # 步骤5：修剪alpha_j
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        # 更新Ej至误差缓存
        updateEk(oS, j)
        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            print('alpha_j变化太小')
            return 0
        # 步骤6：更新alpha_i
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold-oS.alphas[j])
        # 更新Ei至误差缓存
        updateEk(oS, i)
        # 步骤7：更新b_1和b_2
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        # 步骤8：根据b_1和b_2更新b
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


"""完整的线性SMO算法"""
def smoP(dataMatIn, classLabels, C, toler, maxIter):
    """
    :param dataMatIn: 数据矩阵
    :param classLabels: 数据标签
    :param C: 松弛变量
    :param toler: 容错率
    :param maxIter: 最大迭代次数
    :return:oS.b  SMO算法计算的b
            oS.alphas   SMO算法计算的alphas
    """
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler)  # 初始化数据结构
    iter = 0  # 初始化当前迭代次数
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):  # 遍历整个数据集alpha都没有更新或者超过最大迭代次数，则退出循环
        alphaPairsChanged = 0
        if entireSet:  # 遍历整个数据集
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)  # 使用优化算法 alpha是否发生变化
                print('全样本遍历：第%d次迭代 样本:%d,alpha优化次数:%d' % (iter, i, alphaPairsChanged))
            iter += 1
        else:  # 遍历非边界值
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]  # 遍历不在边界0和C的alpha
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print('非边界遍历：第%d次迭代 样本%d,alpha优化次数：%d'% (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:  # 遍历一次后改为非边界遍历
            entireSet = False
        elif alphaPairsChanged == 0:  # 如果alpha没有更新,计算全样本遍历
            entireSet = True
        print('迭代次数:%d'% iter)
    return oS.b, oS.alphas


"""计算w"""
def get_w(dataMat, labelMat, alphas):
    alphas,dataMat,labelMat = np.array(alphas), np.array(dataMat), np.array(labelMat)
    # labelMat.reshape(1, -1).T  将labelMat转为1行不管多少列的数组（这里为100列），再转置变为 100*1
    # np.tile(labelMat.reshape(1, -1).T, (1, 2))  将上面100*1的数组扩为1行，2列  即100*2
    # 与dataMat对应位置相乘，得到的np.tile(labelMat.reshape(1, -1).T, (1, 2)) * dataMat 还是100*2
    # 又alpha为100*1  所以将上式转置为2*100  使用.dot执行矩阵乘法 2*100  X  100*1  得到一个w为 2*1 的数组
    w = np.dot((np.tile(labelMat.reshape(1, -1).T, (1, 2)) * dataMat).T, alphas)
    return w.tolist()


"""计算w"""
def calcWs(alphas, dataArr, classLabels):
    X = np.mat(dataArr); labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * labelMat[i], X[i,:].T)
    return w


"""分类结果可视化"""
def showClassifer(dataMat, labelMat, w, b):
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
    b, alphas = smoP(dataMat, labelMat, 0.6, 0.001, 40)
    w = calcWs(alphas, dataMat, labelMat)
    showClassifer(dataMat, labelMat, w, b)





















