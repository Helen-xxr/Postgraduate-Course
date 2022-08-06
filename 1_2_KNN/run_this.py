import sklearn.datasets as datasets
import matplotlib as plt
from sklearn.model_selection import train_test_split
from definition import *

# test
# load data
x, y = datasets.load_digits(return_X_y=True)

# 可视化查看数据集前200的数据
# for i in range(200):
#     x_i = x[i]
#     x_i = np.array(x_i)
#     x_i = np.reshape(x_i, (8, 8))
#     plt.imshow(x_i)
#     plt.axis('off')
#     plt.savefig('dataset/x_' + str(i) + '.png')

# 将数据集按一定比例分为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2048)

# 使用KNN进行预测
knn = KNN(x_test, x_train, k=7)

# 输出预测结果
y_prediction = knn.KNNTraining(x_test, x_train, y_train)
print("=================== 测试结果 ===================")
print()
print("          INDEX         |         PREDICTION")
print("------------------------------------------------")
for index in range(20):
    print("           " + str(index) + "                      " + str(y_prediction[index]))
print("------------------------------------------------")
print("Accuracy on Test Set: " + str(np.mean(y_prediction == y_test)))
print()
print("===============================================")


