# 一、项目介绍
这个项目是一个作业，将TsingZ0大佬的PFLlib项目fork成为我自己的项目，用于学习和研究。
# 二、项目思路
折线图是fedavg客户端与sgd之间随训练轮数的距离变化

柱状图是fedavg模型和sgd之间的层间距离关系
# 三、项目实现
一共分两步进行：

## 第一步：数据集的划分：以MINST数据集为例,其他数据集类似：

一、iid情况：

python dataset\generate_MNIST.py iid balance -

二、dir情况下：

python dataset\generate_MNIST.py noniid balance dir

注：dir情况下，需要指定dirichlet分布的参数alpha，把dataset_utils.py中的alpha参数修改为指定值即可，随着alpha的增大，数据集的分布会越来越非iid。

三、pat情况下：

python dataset\generate_MNIST.py noniid balance pat

## 第二步：运行main函数

python main.py - nc 10 

注：这里看你自己的需求，自己看一下main函数。
