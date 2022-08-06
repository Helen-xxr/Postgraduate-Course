import numpy as np


class KNN:
    # 初始化相关参数
    def __init__(self, x_test, x_train, k):
        # 测试数据与已知样本的距离
        self.neighbor_distance = np.zeros((len(x_test), len(x_train)))
        # 样本预测值
        self.prediction = np.zeros((len(x_test,)))
        # 邻近样本点个数
        self.k = k

    # 计算两点间的欧式距离
    def distance(self, dot_1, dot_2):
        return np.sqrt(np.sum(np.power((dot_1 - dot_2), 2)))

    # 预测测试样本属于k类中的哪一类
    def KNNPrediction(self, distance, y_train):
        # 获取距离最小的前k个索引
        k_index = distance.argsort()[:self.k]
        # 获取距离最小的前k个物体的类别
        k_prediction = [y_train[index] for index in k_index]
        # 取出类别出现次数最多的类别作为预测结果
        prediction = np.argmax(np.bincount(k_prediction))
        return prediction

    def KNNTraining(self, x_test, x_train, y_train):
        for i in range(len(x_test)):
            for j in range(len(x_train)):
                self.neighbor_distance[i][j] = self.distance(x_test[i], x_train[j])
                # 获取所有样本数据的预测值
                self.prediction[i] = self.KNNPrediction(self.neighbor_distance[i], y_train)

        return self.prediction
