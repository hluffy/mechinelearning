import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class AdalineGD(object):
    """
    eta:float
    学习效率，处于0和1

    n_iter:int
    对训练数据惊醒学习改进次数

    w_:一维向量
    存储权重数值

    error_:
    存储每次迭代改进时，网络对数据进行错误判断的次数
    """

    def __init__(self, eta=0.01, n_iter = 50):
        self.eta = eta
        self.n_iter = n_iter
        pass

    def fit(self, X, y):
        """
        X:二维数组[n_sampls, n_features]
        n_samples 表示X中含有训练数据条目数
        n_features 含有4个数据的一维向量，用于表示一条训练条目

        y:一维向量
        用于存储每一训练条目对应的正确分类
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            print(output.shape)
            print(output)
            errors = (y - output)
            self.w_[1: ] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0, 1, -1)

from matplotlib.colors import ListedColormap

def plot_decision_regions(x, y, classifier, resolution = 0.02):
    marker = ('s', 'x', 'o', 'v')
    colors = ('red', 'blue', ' green', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = x[: , 0].min() - 1, x[: ,0].max()
    x2_min, x2_max = x[: , 1].min() - 1, x[: ,1].max()
    # print x1_min, x1_max, x2_min, x2_max
    # 185 从3.3- 6.98 每隔0.02
    # 255 从0  - 5.08 每隔0.02
    # xx1   从3.3 - 6.98 为一行  有185行相同的数据
    # xx2   从0   - 5.08 为一列  第一行全为0 第二行全1 (255, 185)

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                            np.arange(x2_min, x2_max, resolution))
    # 相当于 np.arange(x1_min, x1_max, resolution) np.arange(x2_min, x2_max, resolution)
    # 已经在分类了站如果是3.3 0 则为1 6.94 5.08 则-1

    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    print(xx1.ravel())
    print(xx2.ravel())
    print(z)

    z = z.reshape(xx1.shape)
    print(z)

    # 在两个分类之间画分界线
    plt.contourf(xx1, xx2, z, alpha = 0.4, cmap = cmap)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.xlabel('length of the huajing')
    plt.ylabel('length of the huaban')
    plt.legend(loc = 'upper left')
    plt.show()
    pass

def main():
    df = pd.read_csv('data.csv', header = None)
    y = df.loc[0:99, 4].values
    y = np.where(y == 'Iris-setosa', 1, - 1)
    x = df.iloc[0:100, [0, 2]].values
    plt.scatter(x[:50, 0], x[:50, 1], color = 'red', marker = 'o',label = 'setosa')
    plt.scatter(x[50:100, 0], x[50:100, 1],color = 'blue', marker = 'x', label = 'versicolor')
    plt.xlabel('length of the huajing')
    plt.ylabel('length of the huaban')
    plt.legend(loc = 'upper right')
    # plt.show()

    # p1 = Perceptron(eta = 0.1)
    # p1.fit(x, y)
    # plot_decision_regions(x, y, p1)

    ada = AdalineGD(eta = 0.0001, n_iter = 200)
    print(y.shape)
    print(y)
    ada.fit(x, y)
    plot_decision_regions(x, y, classifier=ada)

    # plt.title('Adaline-Gradient descent')
    # plt.xlabel('length of the huajing')
    # plt.ylabel('length of the huaban')
    # plt.legend(loc = 'upper left')
    # plt.show()

    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker = 'o')
    plt.xlabel('Epochs')
    plt.ylabel('sum-squard-error')
    plt.show()
    pass

if __name__ == '__main__':
    main()
