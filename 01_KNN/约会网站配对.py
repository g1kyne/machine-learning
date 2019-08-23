import numpy as np
import operator  # 运算符模块
import matplotlib
import matplotlib.pyplot as plt

"""
部分txt数据格式为：
| 每年获得的飞行常客里程数 | 玩视频游戏所耗时间百分比 | 每周消费的冰淇淋公升数| 标签|
|		40920			 |8.326976				  |0.953952			    |  3 |
|		14488			 |7.153469				  |1.673904				|  2 |
|		26052			 |1.441871				  |0.805124				|  1 |
"""


def knn(inX, dataSet, labels, k):
    """用于分类的输入向量，训练集，标签类别向量，选择最近邻居的数目"""
    dataSetSize = dataSet.shape[0]  # 训练集行数，即有多少个数据
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # 计算对应元素差值
    sqDiffMat = diffMat**2  # 平方
    sqDistances = sqDiffMat.sum(axis=1)  # 按行累加
    distances = sqDistances**0.5  # 开放，求欧式距离
    sortedDistIndicies = distances.argsort()  # 按距离递增排序.返回值为索引
    classCount = {}  # 字典，存储某类别的出现频率
    for i in range(k):  # 计算距离最小的k个点所在类别的出现频率
        voteIlabel = labels[sortedDistIndicies[i]]  # 第i个对应的标签
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 统计出现次数
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]  # 返回出现次数最高的类别


# 将文本记录转换为numpy的解析程序
def file2matrix(filename):  # 传入文件名
    fr = open(filename)  # 打开文件
    arrayOlines = fr.readlines()  # 按行读取文件
    #print(arrayOlines)
    #arrayOlines的内容为：['40920\t8.326976\t0.953952\t3\n', '14488\t7.153469\t1.673904\t2\n',...]
    numberOfLines = len(arrayOlines)  #多少行，即多少个数据
    returnMat = np.zeros((numberOfLines, 3))  # 创建以0填充的矩阵
    classLabelVector = []  #存储标签向量
    index = 0
    # lalels = {'didntLike': 1, 'smallDoses': 2, 'largeDoses': 3}
    for line in arrayOlines:  # 处理每行数据 line的内容为每行数据：'40920\t8.326976\t0.953952\t3\n'
        line = line.strip()  # 截取掉回车
        listFromLine = line.split('\t')  # 按tab分割数据
        returnMat[index, :] = listFromLine[0:3]  # 将前3个特征元素存储到特征矩阵中
        classLabelVector.append(int(listFromLine[-1]))  # 将最后一列（标签）存储到标签向量中
        # 标签向量只有一列，直接用append，而特征矩阵有3列
        index += 1  # 行数+1
    return returnMat, classLabelVector


# 测试
# datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
# print(datingDataMat)
# print(datingLabels[0:20])


# 画图
def plot():
    fig = plt.figure()  # 创建图像
    ax = fig.add_subplot(111)  # 1*1网格，第1子图
    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0*np.array(datingLabels), 15.0*np.array(datingLabels))
    plt.show()


# plot()


# 归一化特征值 mewValue = (oldValue-min)/(max-min)
def autoNorm(dataSet):
    minVals = dataSet.min(0)  # 0表示在列中选最小值，此时的minVals为1行3列的向量
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    m = dataSet.shape[0]  # 行数，即数据个数
    normDataSet = dataSet - np.tile(minVals, (m, 1))  # 因为minVals为1*3的矩阵，所以需以行重复m次，列重复1次扩充
    normDataSet = normDataSet/np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


# 测试
# normDataSet, ranges, minVals = autoNorm(datingDataMat)
# print(normDataSet)
# print(ranges)
# print(minVals)


# 测试约会网站预测效果
def datingClassTest():
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normDataSet, ranges, minVals = autoNorm(datingDataMat)
    m = normDataSet.shape[0]  # 样本数量
    numTestVecs = int(m*0.10)  # 用于作为测试样本的数目（也是选取训练集的上标起始点）
    errorCount = 0  # 错误分类的计算器
    for i in range(numTestVecs):
        classifierResult = knn(normDataSet[i, :], normDataSet[numTestVecs:m, :], datingLabels[numTestVecs:m], 4)
        print('真实值为：' + str(datingLabels[i]), '  预测值为：' + str(classifierResult))
        if classifierResult != datingLabels[i]:  # 预测值与真实值不等式，计数器+1
            errorCount += 1
    print('错误率为：' + str((errorCount / numTestVecs)))


# 测试
# datingClassTest()


# 新数据的预测
def classifyPerson():
    resultList = ['不喜欢', '一点点喜欢', '很喜欢']
    percentTats = float(input("玩视频游戏所耗时间百分比："))
    ffMiles = float(input("每年获得的飞行常客里程数："))
    iceCream = float(input("每年消费冰淇淋公升数："))
    datingDateMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDateMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = knn((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print('你喜欢这个人的程度：' + resultList[classifierResult-1])


classifyPerson()






