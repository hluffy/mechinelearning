import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Perceptron(object):
    """
    eta:学习率
    n_iter:权重向量的训练次数
    w_:神经分叉权重向量
    errors_:用于记录神经元判断出错次数
    """
    def __init__(self, eta = 0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        pass

    def fit(self, X, y):
        """
        输入训练数据，培训神经元，x输入样本向量，y对应样本分类
        X：shape[n_samples, n_features]
        X:[[1,2,3], [4,5,6]]
        n_samples: 2
        n_features: 3
        y: [1, -1]
        """

        """
        初始化权重向量为0
        加一是因为前面算法提到的w0，也就是步调函数阈值
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            """
            X:[[1,2,3], [4,5,6]]
            y:[1, -1]
            zip(X,y) = [[1,2,3, 1],[4,5,6, -1]]
            """
            for xi, target in zip(X, y):
                """
                update = n * (y-y')
                """
                update = self.eta * (target - self.predict(xi))

                """
                xi 是一个向量
                update * xi 等价：
                [△w(1) = X[1]*update, △w(2) = X[2]*update]
                """
                self.w_[1:] += update * xi
                self.w_[1:] += update * xi

                errors += int(update != 0.0)
                self.errors_.append(errors)
                pass


            pass
        pass

    def net_input(self, X):
        """
        z = W0*1 + W1*X1 + ... + Wn*Xn
        """
        return np.dot(X, self.w_[1:] + self.w_[0])
        pass

    def predict(self, X):
        return np.where(self.net_input(X)>= 0.0, 1, -1)
        pass

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
    plt.legend(loc = 'upper right')
    plt.show()
    pass

def main():
    df = pd.read_csv('data.csv', header = None)
    y = df.loc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', 1, - 1)
    x = df.iloc[0:100, [0, 2]].values
    plt.scatter(x[:50, 0], x[:50, 1], color = 'red', marker = 'o',label = 'setosa')
    plt.scatter(x[50:100, 0], x[50:100, 1],color = 'blue', marker = 'x', label = 'versicolor')
    plt.xlabel('length of the huajing')
    plt.ylabel('length of the huaban')
    plt.legend(loc = 'upper right')
    # plt.show()

    p1 = Perceptron(eta = 0.1)
    p1.fit(x, y)
    plot_decision_regions(x, y, p1)
    pass

if __name__ == '__main__':
    main()