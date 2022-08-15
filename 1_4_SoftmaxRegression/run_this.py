from definition import *
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt


# test
# load data
iris = load_iris()
x = iris.data
y = iris.target  # y in {0, 1, 2}
feature_names = iris.feature_names

# 将数据集按一定比例分为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=2048)
y_label = y_train.reshape((len(y_train), 1))
# print(y_label)

# 输出数据集散点分布图
plt.scatter(x[y == 0, 1], x[y == 0, 2], c='r', label="Setosa")
plt.scatter(x[y == 1, 1], x[y == 1, 2], c='b', label="Versicolor")
plt.scatter(x[y == 2, 1], x[y == 2, 2], c='g', label="Virginica")
plt.xlabel(feature_names[1])
plt.ylabel(feature_names[2])
plt.legend()
plt.show()

# 使用s o f t m a x   r e g r e s s i o n算法进行训练
SoftRe = SoftmaxRegression(iteration=10000, learning_rate=0.1)
weights = SoftRe.gradientDescent(x_train, y_label, num_type=3)
y_prediction_train = SoftRe.softmaxTraining(x_train, weights)
y_prediction_test = SoftRe.softmaxPrediction(x_test, weights)
print("=================== 训练结果 ===================")
print()
print("Weights List: \n" + str(weights))
print("Train Label: \n" + str(y_train))
print("Train Prediction: \n" + str(y_prediction_train))

# 查看混淆矩阵，绘制热力图
confusion_matrix_test = metrics.confusion_matrix(y_prediction_train.tolist()[0], y_train)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_test, annot=True, cmap="Pastel2", annot_kws={'size':14})
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.xlabel("预测品种", fontsize=14)
plt.ylabel("实际品种", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("使用热力图查看训练结果的混淆矩阵", fontsize = 14)
plt.show()

print()
print()
print("=================== 测试结果 ===================")
print()
print("Test Label: \n" + str(y_test))
print("Test Prediction: \n" + str(y_prediction_test))

# 查看混淆矩阵，绘制热力图
confusion_matrix_test = metrics.confusion_matrix(y_prediction_test.tolist()[0], y_test)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_test, annot=True, cmap="Pastel2", annot_kws={'size':14})
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.xlabel("预测品种", fontsize=14)
plt.ylabel("实际品种", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("使用热力图查看测试结果的混淆矩阵", fontsize = 14)
plt.show()