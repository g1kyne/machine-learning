from numpy import *
import operator  # 运算符模块

def createDataSet():
    """创建数据集和标签"""
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

group,labels = createDataSet()
print(group)
print(labels)

def knn(inX, dataSet, labels, k):
    """用于分类的输入向量，训练集，标签类别向量，选择最近邻居的数目"""
    dataSetSize = dataSet.shape[0]  # 训练集行数，即有多少个数据
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # 计算对应元素差值
    """
    tile函数：将输入向量变成和训练数据一样的形状
    例如：输入向量为[1.2, 1]，
    变换后为：行上重复dataSetSize次，列上重复1次
    [[1.2,   1],
    [1.2,   1],
    [1.2,   1],
    [1.2,   1]]
    """
    sqDiffMat = diffMat**2  # 平方
    sqDistances = sqDiffMat.sum(axis=1)  # 按行累加
    distances = sqDistances**0.5  # 开放，求欧式距离
    sortedDistIndicies = distances.argsort()  # 按距离递增排序.返回值为索引
    classCount = {}  # 字典，存储某类别的出现频率
    for i in range(k):  # 计算距离最小的k个点所在类别的出现频率
        voteIlabel = labels[sortedDistIndicies[i]]  # 第i个对应的标签
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 统计出现次数
        """
        当votaIlabel的值存在时，返回classCount[voteIlabel](即出现次数)；
        不存在时，返回0
        """
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    """
    operator.itemgetter(1):返回前面classCount迭代器中的第1列
    sorted中key指定按value值（key:value）排序：  按operator.itemgetter(1)的值（即出现次数）排列
    reverse=True降序
    """
    return sortedClassCount[0][0]  # 返回出现次数最高的类别


print(knn([0, 0], group, labels, 3))  # 输出为B  sortedClassCount[0]值为：('B', 2)
