from math import log
import pickle


# 创建数据集
def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],         # 数据集
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']		# 分类属性
    return dataSet, labels  # 返回数据集和分类属性


# 输出预先存储的树信息，避免每次测试代码都要从数据中创建树
def retrieveTree(i):
    listOfTrees = [
        {'没有浮出水面': {0: 'no', 1: {'有脚蹼': {0: 'no', 1: 'yes'}}}},
        {'没有浮出水面': {0: 'no', 1: {'有脚蹼': {0: {'头':{0: 'no', 1: 'yes'}}, 1: 'no'}}}}
    ]
    return listOfTrees[i]


# 计算给定数据集的香农熵
# H = -sum{p(i)log2[p[i]]} i: 1-n
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)  # 数据集数据个数
    labelCounts = {}  # 创建字典，{类别标签：数据集中为该标签的数据总数}
    # 创建的字典的作用： 后面计算选择某分类的概率
    for featVec in dataSet:  # 遍历每条数据
        currentLabel = featVec[-1]  # 当前数据的类别标签存放在最后一列
        if currentLabel not in labelCounts.keys():  # 若该标签不在字典内，将该标签对应的数据总数设为0
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1  # 在字典里时，该类别标签数量+1
    shannonEnt = 0.0  # 香农熵
    for key in labelCounts:  # 遍历字典中所有key值，即类别标签
        prob = float(labelCounts[key] / numEntries)  # 选择该标签的概率   选择该标签的数目/数据总数
        shannonEnt -= prob * log(prob, 2)  # 计算香农熵
    return shannonEnt


# 测试 熵越高，混合的数据越多
# myDat, labels = createDataSet()
# print(myDat)
# print(calcShannonEnt(myDat))


"""
得到熵后，按照最大信息增益的方法划分数据集，当然也可以按照信息增益率或基尼指数来划分
"""


# 按照给定特征划分数据集 分类某一个特征向量，然后返回去掉该特征后的数据集（因为该特征已被处理）
def splitDataSet(dataSet, axis, value):  # 待划分的数据集,待划分的特征下标，将该属性值与value比较
    retDataSet = []  # 创建新的列表对象
    # python语言不考虑内存分配问题。  在函数内部对列表对象的修改，将会影响该列表对象的整个生存周期。
    # 因为该段代码将会在同一数据集上调用多次，为了不修改原始数据集，需要创建一个新的列表对象
    for featVec in dataSet:  # 处理每组数据
        if featVec[axis] == value:  # 如果待划分的特征与value的值相等，就去掉该特征列，得到去掉该特征后的数据集
            reducedFeatVec = featVec[:axis]  # 取该特征前面的特征列
            reducedFeatVec.extend(featVec[axis+1:])  # 取到该特征后的特征列，与前面的拼接在一起
            retDataSet.append(reducedFeatVec)  # 得到的每条数据加到retDataSet中
    return retDataSet  # 最后的数据集比原数据集少了一列值为value的特征
"""
执行过程：
[[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
splitDataSet(myDat, 0, 1)
[[1, 'yes'], [1, 'yes'], [0, 'no']]
先判断下标为0的特征，如果特征值为1就留下该特征，并且返回将该特征去掉后的数据集
"""

'''extend 和 append 区别
append:  a = [1, 2, 3]  b = [4, 5, 6]
         a.append(b)
         a = [1, 2, 3, [4,5,6]]
append是直接把b作为整体加到a中

extend:  a = [1, 2, 3, 4, 5, 6]
extend 把b的元素分开后加到a中
'''


# 选择最好的数据集划分方式 Gain = H(D) - H(D|A) D为训练集，A为特征
# 在划分数据集之前之后信息发生的变化称为信息增益,信息增益是熵的减少
# 计算每个特征值划分数据集获得的信息增益，获得信息增益最高的特征就是最好的选择
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 特征属性个数，最后一列是当前实例的类别标签
    baseEntropy = calcShannonEnt(dataSet)  # 原始熵
    bestInfoGain = 0.0  # 最高的信息增益
    bestFeature = -1  # 划分数据集的最优特征
    for i in range(numFeatures):  # 遍历每个特征
        featList = [example[i] for example in dataSet]  # 先按行遍历数据集，取出第i个特征的属性值值放入列表中
        uniqueVals = set(featList)  # 集合，去重，去除特征i重复的属性值
        newEntropy = 0.0  # 新香农熵
        # 计算按不同属性值划分的香农熵
        for value in uniqueVals:  # 遍历特征i可能取到的所有属性值
            subDataSet = splitDataSet(dataSet, i, value)  # 按每个属性值划分数据集
            prob = len(subDataSet) / len(dataSet)  # 按属性值=value划分的数据集占整个数据集的比例
            # 计算第i个特征的一个可能属性值的新熵，遍历所有可能出现的属性值，将每个属性值的新熵求和，得到第i个特征的新熵
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy  # 第i个特征值的信息增益
        # 比较得到信息增益最高的特征i
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature  # 返回信息增益最高，即用于划分数据集的最优特征的下标


# 测试
# myDat, labels = createDataSet()
# print(chooseBestFeatureToSplit(myDat))  # 输出为0，第0个特征是最好的用于划分数据集的特征


# 如果数据集已经处理了所有属性，但是类标签依然不是唯一的，此时需要
# 采用 多数表决 的方法定义该叶子节点
def majorityCnt(classList):  # 分类列表
    classCount = {}  # 创建字典，存储每个类标签出现的频率
    for vote in classList:  # 遍历列表
        if vote not in classCount.keys():  # 如果标签 不在字典中，就添加该标签，且出现次数设为0
            classCount[vote] = 0
        classCount[vote] += 1  # 在字典中，出现次数+1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 对出现次数进行排序
    return sortedClassCount[0][0]  # 返回出现次数最多的分类名称


# 创建树
def createTree(dataSet, labels):  # 数据集  特征的标签值：‘没有浮出水面’，‘有脚蹼
    classList = [example[-1] for example in dataSet]  # 类别列表
    # 递归停止的第一个条件,类别完全相同则停止
    if classList.count(classList[0]) == len(classList):  # count()计算classList[0]在列表中出现的次数，若与总数相等，表示类别完全相同
        return classList[0]
    # 递归停止的第二个条件，用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组
    if len(dataSet[0]) == 1:  # 第1条数据长度为1，即只有最后一列的类别，没有特征了
        return majorityCnt(classList)  # 返回出现次数最多的分类名称
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 最优特征的下标
    bestFeatLabel = labels[bestFeat]  # 最优特征索引对应的标签值  labels[0]='没有浮出水面'
    myTree = {bestFeatLabel: {}}  # 创建树
    del(labels[bestFeat])  # 删除最优特征对应的标签值，因为此时的最优特征已分完
    # 若用del(bestFeatLabel)只是删除了引用，不会删除原数据
    featValues = [example[bestFeat] for example in dataSet]  # 取出example[bestFeat]的值，即数据集中最优特征可能出现的属性值放入列表中
    uniqueVals = set(featValues)  # 去重
    for value in uniqueVals:
        # 为了保证每次调用函数createTree()时不改变原始列表的内容，使用新列表subLabels代替原始列表
        subLabels = labels[:]  # 去除最优特征后的特征标签
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


# 测试
# myDat, labels = createDataSet()
# print(createTree(myDat, labels))
# {'没有浮出水面': {0: 'no', 1: {'有脚蹼': {0: 'no', 1: 'yes'}}}}
# 包含了很多代表树结构信息的嵌套字典，
# 第一个关键字是 '没有浮出水面' 是第一个划分数据集的特征名字，该关键字的值也是另一个数据字典
# 第二个关键字是' 没有浮出水面' 特征划分的数据集，关键字的值是'没有浮出水面'节点的子节点。这些值可能是类标签，页可能是另一个数据字典
# 如果是类标签，则该子节点就是叶子节点； 如果是另一个数据节点，则子节点是一个判断节点，这种格式不断重复构成树


# 使用决策树分类
def classify(inputTree, featLabels, testVec):  # 构建好的树，特征标签（即特征的名称），要测试的特征向量
    firstStr = list(inputTree.keys())[0]  # 树的第一个特征标签
    secondDict = inputTree[firstStr]  # 第一个特征标签对应的value，也是一棵树（即一个复合字典）
    featIndex = featLabels.index(firstStr)  # 对featLabels使用index()方法，查找featLabels中第一个出现firstStr的下标（即第一个分类特征的下标）
    for key in secondDict.keys():  # 遍历第二颗树的所有key值（此时的key值是第一个特征可能出现的属性值）
        if testVec[featIndex] == key:  # 如果待测特征向量的第一个特征对应的value值 == 决策树上的数值。
            if type(secondDict[key]).__name__ == 'dict':  # 如果该值后面还是一棵树，就继续递归直至不是树（即到达了叶子节点，找到了分类类别）
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]  # 如果第二颗树的key值（即第一个特征的取值）对应的value值不是字典，即到达了叶节点，分类完成
    return classLabel  # 返回最后的分类


# 测试
# myDat, labels = createDataSet()
# print(labels)  # ['没有浮出水面', '有脚蹼']
# myTree = retrieveTree(0)
# print(myTree)  # {'没有浮出水面': {0: 'no', 1: {'有脚蹼': {0: 'no', 1: 'yes'}}}}
# print(classify(myTree, labels, [1, 0]))  # no
# print(classify(myTree, labels, [1, 1]))  # yes


# 在每次执行分类前调用已经构造好的决策树
# 使用pickle序列化对象，序列化对象可以在磁盘上保存对象，并在需要的时候读取出来
def storeTree(inputTree, filename):  # 将决策树存到文件中
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grapTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)









