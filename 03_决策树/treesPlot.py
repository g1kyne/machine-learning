import matplotlib.pyplot as plt

# 定义文本框和箭头样式
decisionNode = dict(boxstyle='sawtooth', fc='0.8')  # 决策节点的文本框类型为 锯齿形 ，边框粗细为0.8
leafNode = dict(boxstyle='round4', fc='0.8')  # 叶子节点文本框类型为 圆一点的四边形，边框粗细为0.8
arrow_args = dict(arrowstyle='<-')  # 箭头类型


def plotNode(nodeTxt, centerPt, parentPt, nodeType):  # 节点文本信息，节点框位置，箭头起始位置，节点类型
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt,
                            textcoords='axes fraction', va='center', ha='center',
                            bbox=nodeType, arrowprops=arrow_args)
    # 该函数的作用是为绘制的图上指定的数据点xy（传入）添加一个注释nodeTxt（传入）
    # xycoords:指定xy坐标类型  axes fraction：取值是小数，范围是([0, 1], [0, 1])
    # xytext：注释位置，在这里是指方框位置，由函数传入
    # textcoords：指定xytext坐标类型


def createPlot():
    fig = plt.figure(1, facecolor='white')  # 创建新的画布1，背景为白色
    fig.clf()  # 清空figure1的内容
    '''
   在新建的figure 1里面创建一个1行1列的子figure的网格,并把网格里面第1个子figure的Axes实例axes返回给ax1作为函数createPlot()的属性
    ，这个属性ax1相当于一个全局变量,可以给plotNode函数使用
    '''
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode('决策节点', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('叶节点', (0.8, 0.1), (0.3, 0.8),leafNode)
    plt.show()


# 测试
# createPlot()


# 获取叶节点数目-------》x轴长度
def getNumLeafs(myTree):
    numLeafs = 0  # 叶节点个数
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]  # 树中第一个key值
    secondDict = myTree[firstStr]  # 第一个key值对应的value，此时的value是第二个个字典
    for key in secondDict.keys():  # 遍历第二个字典的所有key值
        if type(secondDict[key]).__name__ == 'dict':  # 如果该key值对应的value仍然是字典类型，说明它不是叶子节点
            numLeafs += getNumLeafs(secondDict[key])  # 对该key值对应的value再用该函数，
        else:   # 递归直到不是字典类型，即到达了叶子节点，将节点数+1
            numLeafs += 1
    return numLeafs


# 获取树的层数(即判断节点个数)---------》确定y轴高度
def getTreeDepth(myTree):
    maxDepth = 0
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]  # 树中第一个key值
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 如果是字典类型，即是判断节点
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1  # 一旦到达叶子节点，从递归调用中返回
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


# 输出预先存储的树信息，避免每次测试代码都要从数据中创建树
def retrieveTree(i):
    listOfTrees = [
        {'没有浮出水面': {0: 'no', 1: {'有脚蹼': {0: 'no', 1: 'yes'}}}},
        {'没有浮出水面': {0: 'no', 1: {'有脚蹼': {0: {'头':{0: 'no', 1: 'yes'}}, 1: 'no'}}}}
    ]
    return listOfTrees[i]


# 测试
print(retrieveTree(0))   # {'没有浮出水面': {0: 'no', 1: {'有脚蹼': {0: 'no', 1: 'yes'}}}}
myTree = retrieveTree(0)
print(getNumLeafs(myTree))  # 树0有   3 个叶子节点
print(getTreeDepth(myTree))  # 2 个判断节点


# 在父子节点之间填充文本信息
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)









