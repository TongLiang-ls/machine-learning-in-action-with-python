# 决策树(DecisionTree)

[TOC]

优点：

* 计算复杂度不高
* 对中间缺失值不敏感
* 可以处理不相关数据

缺点：

* 可能产生过度匹配度问题


## 构造决策树

>构造过程需要解决以下几个问题：
* 1.哪个特征在分类时起决定作用
* 2.某个分支下数据属于同一类型，则无需划分，否则继续重复划分这一步骤
* 3.而划分子集的算法是相同的，只要一直重复，直到同一数据类型的数据在一个子集内即可

可以发现，决策树过程是一个递归步骤，基线条件是：子集内所有数据都是同一类型；
关于递归，可以复习之前的笔记[递归笔记](https://www.jianshu.com/p/e9cbac709181)

___

__一般步骤__
* 1.收集数据
* 2.准备数据-只适合标签数据，需要离散化
* 3.分析数据
* 4.训练算法
* 5.测试算法
* 6.使用算法


## ID3算法

>对于每次划分数据集，第一次划分的特征选择如何来选呢？需要引入以下几个概念来量化特征选择的过程：1.信息增益、2.香侬熵。

### 信息增益(information gain)

__对于划分数据集的大原则就是：将无序变有序__

* 信息增益:数据集划分前后，信息的变化

所以，信息增益越大，说明用这个特征来划分前后数据集信息变化越大，即这个特征的区分度越高(_这个特征带来的信息越多，越重要_)，适合用来作为划分数据集的特征。

* 信息熵(entropy):离散随机事件出现的概率，系统越有序，熵越低，反之越混乱，熵越高。_所以可以作为一个系统有序化程度的度量。_

(公式图片)

### 计算信息熵

```
dataSet = [[1, 1, 'yes'],
           [1, 1, 'yes'],
           [1, 0, 'no'],
           [0, 1, 'no'],
           [0, 1, 'no']]
labels = ['不浮出水面是否生存','是否有蹼']

from math import log

def Entropy(dataSet):
    num = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: 
        currentLabel = featVec[-1]
        labelCounts[currentLabel] = labelCounts.get(currentLabel,0) +1
        #统计类别出现的次数，使用get方法，出现加1，没有添加新类别+1
    entropy = 0.0
    for key in labelCounts:  
        prob = float(labelCounts[key])/num
        entropy -= prob * log(prob,2)
        #计算类别概率以及香侬熵
    return entropy

```

`基尼不纯度`：

### 划分数据集

```
def SplitDataSet(dataSet,axis,value): 
    '''
    dataSet:待划分的数据集
    axis:用来划分数据集的特征(最佳划分特征)-特征在列表的位置，第一个第二个...分别对应0,1,2....
    value:特征返回值-特征的结果只能是0，1(是，否)-->(1,0)
    '''
    split_set = []#设置一个空list，用于存放划分结果
    for vector in dataSet:
        result_axis = vector[axis]
        if result_axis == value:
            tempV = vector[:]#copy，复制向量为了下一步pop方法时，数据集内容保持不变
            tempV.pop(axis)
            split_set.append(tempV)
    return split_set

```





### 选择最佳划分特征

* 最佳划分特征其实就是用该特征划分数据集后，`前后信息熵改变的大小(信息增益)`，差值越大，信息增益越大，说明该特征包含的信息量越大，适合用于作为分类特征。
* 核心就是计算数据集划分前后的信息熵。
* 划分前的信息熵是信息熵公式和函数计算，划分后的信息熵是计算划分后数据集的信息熵。

idea:
- 1.dataSet--找出特征有哪些
- 2.对每个特征，进行数据集划分--SplitDataSet()
- 3.计算信息增益:信息熵-条件熵(p*entropy)--Entrpy()
- 4.选择一个最大的信息增益
- 5.确定该特征为最佳特征


```
def chooseBestFeatureToSplit(dataSet):
    '''
    根据信息增益来选择最佳特征的划分数据集
    '''
    featureNum = len(dataSet[0])-1#特征个数，list长度-1，最后一个是分类标签
    initEntropy = Entropy(dataSet)#初始信息熵
    infoGain = []
    conditionEntropy = 0
    for axis in range(featureNum):
        conditionEntropy = 0
        class_value = set([x[axis] for x in dataSet])#当前特征的类别集合
        for value in class_value:
            splitSet = SplitDataSet(dataSet,axis,value)
            prob = len(splitSet)/float(len(dataSet))
            conditionEntropy += prob*Entropy(splitSet)
            #计算条件熵
        splitInfoGain = initEntropy-conditionEntropy
        #一次划分后的信息增益
        infoGain.extend([splitInfoGain])
        gianMax = max(infoGain)
        infoGain.index(gianMax)
        print('the index of best feature to split is : %d'%infoGain.index(gianMax))
        return infoGain.index(gianMax)



```

### 递归构建决策树

递归结束的基线条件：
* 遍历完所有属性
* 每个分支下所有实例都具有相同分类

叶子结点/终止块：实例具有相同分类
___
决策树的步骤:1.从数据集中选择一个最佳特征，按其进行数据划分；2.如果划分的数据集类别相同，或者用来划分的特征已经用完，则无需再划分，否则重复1，对剩下的数据集进行划分。划分数据集方法是相同的，等于采用了递归方法。

__idea:__
*  判断特征是否用完，或者数据集内类别是否相同
* else，开始划分数据集，得到子集
* 重新判断子集

```
def majorityCnt(classList):
    '''
    用来统计到达叶子结点的类别
    '''
    classCount={}
    for vote in classList:
        classCount[vote] = classCount.get(vote,0) +1 
    sortedClassCount = sorted(classCount.items(), key=lambda x:x[1], reverse=True)
    return sortedClassCount[0][0] #返回多数的那个类别


def creatTree(dataSet,labels):
    '''
    创建决策树
    dataSet:数据集
    labels:标签类别集
    '''
    classList = [x[-1] for x in dataSet]#叶子节点类别(yes/no)
    classListType = set(classList)
    if len(classListType) == 1:
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    #递归基线条件：特征用光/划分子集内类别都相同
    else:
        
        #划分数据集的过程
        bestFeature_index = chooseBestFeatureToSplit(dataSet)#最佳特征索引
        bestlabels = labels[bestFeature_index]#最好特征对应的名称
        
        #创建树结构
        myTree = {bestlabels:{}}
        
        #每使用一次最佳特征，特征数量-1
        subLabels=labels[:]
        del subLabels[bestFeature_index]
        
        #划分数据集
        featureType = set([x[bestFeature_index] for x in dataSet])#最好特征对应的列的类别
        for value in featureType:
            myTree[bestlabels][value] = creatTree(SplitDataSet(dataSet,bestFeature_index,value),subLabels)#最佳特征划分后对应的子集
    return myTree

```

>__书上代码跑完一次会清空labels，每次都需要重新读取数据集，不方便，原因是对labels数据集直接进行del操作__

## 小结

* 《机器学习实战》一书这一章节，写的比较模糊，首先他没有介绍信息增益如何计算，只提及是数据集前后信息熵的差值，许多书上直接介绍是信息熵-条件熵，计算方式一目了然
* 其次，决策树算法的主要步骤为：`特征选择、决策树生成、决策树剪枝`。本书该章节只是涉及来id3算法来选择特征和生成决策树，并未介绍其他两种算法和剪枝的过程(可能这部分内容比较复杂，容易打击初学者的信心)。
* 本章关于特征选择和数据集划分花了我很多时间，通过阅读其他书籍，才有所理解。
    -`特征选择`：是根据最佳特征来确定的，选择依据是比较用各个特征对数据集进行划分后的信息增益大小，选择信息增益最大的特征作为划分特征。
    -`数据集划分`：划分过程其实是一个递归过程，递归的基线条件是1.特征已经用完；2.划分后的数据集信息增益<一个固定的阈值；3.划分后的子集中类别都属于同一类型，而对于特征用完还不能划分纯净的数据集选择类别比例大的作为叶子结点。
* 决策树容易产生过拟合现象，因此需要进行`剪枝`。具体内容可以在接下来的章节或者《机器学习》、《统计学习方法》中查看。









# 推荐阅读
《统计学习方法》
[sklearn-DecisionTree](https://scikit-learn.org/stable/modules/tree.html)
