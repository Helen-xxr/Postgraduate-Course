import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:

    # define the hypothesis function
    def sigmoid(self, dot):
        return 1 / (1 + np.exp(-dot))

    # define the cost function
    def logisticLoss(self, x, y, theta, bias):
        num_train = x.shape[0]
        y_ = self.sigmoid(np.dot(x, theta) + bias)
        loss = -1 / num_train * np.sum(y * np.log(y_) + (1 - y) * np.log(1 - y_))
        d_theta = np.dot(x.T, (y_ - y)) / num_train
        d_bias = np.sum(y_ - y) / num_train
        loss = np.squeeze(loss)
        return y_, loss, d_theta, d_bias

    # 定义训练过程
    def logisticTraining(self, x, y, theta, bias, learning_rate, epochs):
        loss_list = []

        for i in range(epochs):
            y_, loss, d_theta, d_bias = self.logisticLoss(x, y, theta, bias)
            theta = theta - learning_rate * d_theta
            bias = bias - learning_rate * d_bias
            if i % 100 == 0:
                loss_list.append(loss)

        # 将训练结果存入参数
        params = {
            'theta': theta,
            'bias': bias,
        }
        grads = {
            'd_theta': d_theta,
            'd_bias': d_bias,
        }
        return loss_list, params, grads

    # 定义预测结果函数
    def logisticPrediciton(self, x, params):
        return self.sigmoid(np.dot(x, params['theta']) + params['bias']) >= 0.5

    # 输出分类散点图
    def plotRes(self, x, y, params, feature_names):
        x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        x_1 = np.linspace(x_min, x_max, 100)
        x_2 = (params['theta'][0] * x_1 + params['bias']) / -params['theta'][1]
        plt.plot(x_1, x_2, c='g', label = 'boundary')
        plt.scatter(x[y == 0, 0], x[y == 0, 1], c = 'r', label = 'setosa')
        plt.scatter(x[y == 1, 0], x[y == 1, 1], c = 'b', label = 'versicolor')
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.legend()
        plt.show()