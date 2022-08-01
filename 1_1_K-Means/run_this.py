from definition import *
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

# test
# load data
iris = load_iris()
x = iris.data[:, 1:]
feature_names = iris.feature_names[1:]
# 将数据转换为矩阵的形式
data = np.mat(x.tolist())

# 输出数据集散点分布图
plt.scatter(x[:, 0], x[:, 1], c='r')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.show()

K_Means = KMeans()
center_dot, error_list = K_Means.KMeansTraining(data, 3)

print("=================== 聚类结果 ===================")
print()
print("质心列表: ")
print(center_dot)
print("误差列表: ")
print(error_list)
print()
print("===============================================")

# 画出聚类图
color = ['r', 'b', 'g']
x = np.array(data)
center_dot_arr = np.array(center_dot)
error_list_arr = np.array(error_list[:,0])

# 根据质心，画出对应聚类数据点
for i, n in enumerate(error_list_arr):
    plt.scatter(x[i][0], x[i][1], c=color[int(n[0])])
# 画出质心
plt.scatter(center_dot_arr[:, 0], center_dot_arr[:, 1], marker='*', s=160, c='m', label='Center of Mass')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.show()

