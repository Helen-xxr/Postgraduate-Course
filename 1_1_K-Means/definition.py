import numpy as np


class KMeans:

    # 定义两点间的距离——欧氏距离
    def distance(self, dot_1, dot_2):
        return np.sqrt(np.sum(np.power((dot_1 - dot_2), 2)))

    # 定义随机产生k个质心
    def generateCenter(self, data, k):
        # 获取数据集的特征数
        num_features = np.shape(data)[1]

        # 生成k个质心矩阵
        center_dot = np.mat(np.zeros((k, num_features)))
        for i in range(num_features):
            # 找出每个特征的最小值
            min_dot = np.min(data[:, i])
            # 找出每个特征的范围
            range_dot = float(np.max(data[:, i]) - min_dot)
            # 在该范围中随机产生质心
            center_dot[:, i] = min_dot + range_dot * np.random.rand(k, 1)

        return center_dot

    def KMeansTraining(self, data, k):
        # 获取数据集的数据数量
        num_training = np.shape(data)[0]
        # 生成数据的误差记录列表
        error_list = np.mat(np.zeros((num_training, 2)))
        center_dot = self.generateCenter(data, k)
        flag = True

        while flag:
            flag = False
            # 寻找每一个数据距离哪一个质心最近
            for i in range(num_training):
                min_distance = np.Inf
                min_index = -1
                for j in range(k):
                    # print("11111")
                    # print(data[i, :])
                    # print(center_dot[j, :])
                    distance_i_to_j = self.distance(center_dot[j, :], data[i, :])
                    # print("22222")
                    # print(distance_i_to_j)
                    if distance_i_to_j < min_distance:
                        min_distance = distance_i_to_j
                        min_index = j

                # 检测每个数据对应的质心下标是否被更新为当前最小距离对应的质心下标
                if error_list[i, 0] != min_index:
                    flag = True
                error_list[i, :] = min_index, min_distance**2
            # print("11111")
            # print(center_dot)

            # 更新质心的位置
            for m in range(k):
                # 找到以下标m为质心的所有点的集合
                dots_for_m = data[np.nonzero(error_list[:, 0].A == m)[0]]
                # print("22222")
                # print(dots_for_m)
                # 计算这些点的均值
                center_dot[m, :] = np.mean(dots_for_m, axis=0)

            # print("33333")
            # print(center_dot)

        # 返回新的质心列表以及误差列表
        return center_dot, error_list
