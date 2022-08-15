import numpy as np


class SoftmaxRegression:

    def __init__(self, iteration, learning_rate, weights=None):
        self.iteration = iteration
        self.learning_rate = learning_rate
        self.weights = weights

    def softmaxLoss(self, error, y):
        num_train = np.shape(error)[0]
        cost = 0
        for i in range(num_train):
            if error[i, y[i, 0]] / np.sum(error[i, :]) > 0:
                cost += (-1) * np.log(error[i, y[i, 0]] / np.sum(error[i, :]))
        return cost / num_train

    # 使用梯度下降算法进行训练
    def gradientDescent(self, x, y, num_type):

        num_train, num_feature = np.shape(x)
        self.weights = np.mat(np.ones((num_feature, num_type)))

        for i in range(self.iteration):
            error = np.exp(x * self.weights)
            if i % 1000 == 0:
                print("Iteration: " + str(i) + "\tCost: " + str(self.softmaxLoss(error, y)))

            row_sum = -error.sum(axis=1)
            row_sum = row_sum.repeat(num_type, axis=1)
            error = error / row_sum
            for j in range(num_train):
                error[j, y[j, 0]] += 1

            self.weights = self.weights + self.learning_rate * x.T * error / num_train

        return self.weights

    def softmaxTraining(self, x, weights):
        train_result = x * weights
        train_result = train_result.argmax(axis=1)
        return train_result.flatten()

    def softmaxPrediction(self, x, weights):
        predict_result = x * weights
        predict_result = predict_result.argmax(axis=1)
        return predict_result.flatten()
