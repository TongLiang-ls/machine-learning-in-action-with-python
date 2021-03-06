# k-means聚类

应用：
* 市场分割->对不同类型的客户分别制定不同服务；
* 社交网络分析->用于发现社交网络中关系密切的朋友；
* 

簇识别(cluster identification):给出聚类结果的含义

优点：容易实现
缺点：可能收敛到局部最小值，大规模数据集收敛慢

k个簇，通过其质心(centroid)-**所有点点中心**来描述。

>流程：
* 从样本中选择k个实例，作为初始质心
* 计算各个样本点到k个质心到距离，并将其归类
* 更新质心，继续迭代

## 随机初始化

1. K<m，质点点数小于样本数
2. 随机选k个样本，k个质点与k个样本相等

>k-means会停留在局部最小值处，需要多次运行算法，比较结果后选择一个代价函数最小的结果。

## 聚类数选择

查看cost function和质点数点关系

>聚类的距离计算方法很多，可以根据目的有所选择

聚类的种类：
* 原型聚类
    - k-means
    - 学习向量量化(LVQ)
    - 高斯混合聚类
* 密度聚类->假设聚类结构通过样本分布的紧密程度确定
    - DBSCAN
* 层次聚类->不同层次对数据集划分，形成树形聚类结构，2种策略
    -  自底向上-AGNES
    -  自顶向下

## 代码
```


```

## 二分k-means(bisecting k-means)
用误差平方和(SSE)来作为划分依据。

* 将所有点作为一个簇，将其一分为二
* 选择一个继续划分，依据是这样划分，能最大程度降低SSE.


