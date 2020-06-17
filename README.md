# 
1.income.csv是实验的数据集。具体介绍：

（1）CSV文件，大小为4000行×59列;

（2）4000行数据对应着4000个人，ID编号从1到4000;

（3）59列数据中，第一列为ID，最后一列label(1或0)表示年收入是否大于50K，中间的57列为57种属性值；

（4）将数据中前3000项作为训练集，后1000项作为测试集，使用logistic回归进行二分类，实现语言为Python。

2.sv.py是使用逻辑回归实现个人收入预测（二分类问题）的代码实现。

3.参考资料：

（1）SGDClassifier部分参数解释：https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html

（2）解读随机梯度下降的含义：https://www.cnblogs.com/lliuye/p/9451903.html

（3）机器学习之逻辑回归（纯python实现）：https://www.jianshu.com/p/4cfb4f734358

（4）机器学习--Logistic回归计算过程的推导：https://blog.csdn.net/ligang_csdn/article/details/53838743

