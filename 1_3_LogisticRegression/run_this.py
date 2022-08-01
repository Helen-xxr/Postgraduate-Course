from definition import *
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns

# test
# load data
iris = load_iris()
x = iris.data[:100, :2]
y = iris.target[:100]  # y in {0, 1}
feature_names = iris.feature_names[:2]

# 将数据集按一定比例分为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=2048)


# 输出数据集散点分布图
plt.scatter(x[y == 0, 0], x[y == 0, 1], c='r', label="setosa")
plt.scatter(x[y == 1, 0], x[y == 1, 1], c='b', label="versicolor")
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.legend()
plt.show()

# 初始化theta和bias
num_features = x_train.shape[1]
theta = np.random.randn(num_features)
bias = np.random.randn(1)
np.random.seed(100)

# 使用逻辑回归算法进行训练
LogRe = LogisticRegression()
loss_list, params, grads = LogRe.logisticTraining(x_train, y_train, theta, bias, learning_rate=0.01, epochs=1000)
y_prediction_train = LogRe.logisticPrediciton(x_train, params)
print("=================== 训练结果 ===================")
print()
print("  theta: " + str(params['theta']))
print("  bias: " + str(params['bias']))
print("  accuracy on train set: " + str(metrics.accuracy_score(y_train, y_prediction_train)))
print()
print("===============================================")
LogRe.plotRes(x_train, y_train, params, feature_names)

# 进行预测
y_prediction_test = LogRe.logisticPrediciton(x_test, params)
print()
print()
print("=================== 测试结果 ===================")
print()
print("  accuracy on test set: " + str(metrics.accuracy_score(y_test, y_prediction_test)))
print("  precision on test set: " + str(metrics.precision_score(y_test, y_prediction_test)))
print("  recall on test set: " + str(metrics.recall_score(y_test, y_prediction_test)))
print()
print("===============================================")

# 查看混淆矩阵，绘制热力图
confusion_matrix_test = metrics.confusion_matrix(y_prediction_test, y_test)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_test, annot=True, cmap="Pastel2", annot_kws={'size':16})
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.xlabel("预测品种", fontsize=16)
plt.ylabel("实际品种", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()


