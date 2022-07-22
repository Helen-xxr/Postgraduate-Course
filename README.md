# Postgraduate-Course

  构想：分三个层面：算法、平台、应用。算法是基础，平台是工具，应用结合实验室具体项目。基于这方面考虑，结合实验室的项目情况，暂且制定以下题目。
## 一、学习实验内容
### 1.算法：
  * 目的：使研究生们尽快熟悉人工智能、机器学习与大数据领域的基础算法。
  * 要求：
    * （1） 理解算法本身，用程序设计语言（不限语言，但建议选择Python、C、C++、或Java）实现算法。
    * （2） 除了数学运算（例如求函数值、矩阵求逆等）可以调用现有的库函数，其他一律不准调库。
  * 内容：实现以下算法
    * （1） Kmeans
    * （2） KNN
    * （3） Logistic Regression
    * （4） Softmax Regreesion
    * （5） BP神经网络
    * （6） CART
    * （7） 贝叶斯方法
    * （8） PCA
### 2.平台
  * 目的：掌握项目开发过程中必须使用的一些工具。
  * 要求：完成平台环境搭建，跑示例程序
  * 内容
  #### （1） Hadoop生态环境
    * a.基于Kafka的消息订阅系统
    * b.基于Sqoop和Flume的数据ETL
    * c.基于Oozie的流程管理平台
    * d.Spark环境下非结构化数据的读写
    * e.基于Spark Mllib机器学习应用实践——分类
    * f.基于Spark Mllib机器学习应用实践——回归
    * g.基于Spark Mllib机器学习应用实践——聚类
    * h.基于Spark Mllib机器学习应用实践——决策树
  #### （2） Tensorflow：以下题目请参考《Tensorflow实战》
    * a.简单CNN对MNIST手写体的数字识别
    * b.AlexNet
    * c.VGGNet
    * d.GoogLeNet
    * e.ResNet
    * f.LSTM
  #### （3） Docker和Kubernetes容器集群平台
    * a.在CentOS或者Ubuntu（虚拟机或者物理机上都可以）上面安装部署Docker和Kubernetes环境
    * b.下载或者自己手动通过Dockerfile生成ELK (ElasticSearch, Logstash, Kibana) 的Docker镜像，完成日志系统搭建并测试
    * c.在Kubernetes环境下搭建Tensorflow, 完成(2) 中一个任务
### 3.应用
  * 目的：训练学生利用平台和算法，完成具体事务。
  * 要求：完成演示Demo
  * 内容：
    * （1） 针对文本的分类
    * （2） 运动目标检测
    * （3） 运动目标识别
    * （4） 选择Docker平台的，在Docker平台中完成(1)-(3)中任意一个
### 4.《深度学习》公开课
  吴恩达《深度学习》公开课学习，到（网易或coursera）公开课，申请账号，通过相关课程的学习。

## 二、 提交内容
  完成10个学习报告，内容：
    * 1)8个算法
    * 2)选择完成1个平台（大学习报告）
    * 3)公开课学习总结

  学习报告应包含运行环境说明，数据准备，算法流程或者平台搭建说明，运行效果或结论。
