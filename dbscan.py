# -*- coding=utf-8 -*- 
# time = '2020/4/12 16:34'

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


class DBScan(object):

    def __init__(self, minpts=5, eps=0.1):
        self.minpts = minpts
        self.eps = eps

    def distance(self, x, y):
        """计算距离"""
        return np.sqrt(np.sum((x - y) ** 2))

    def fit(self, dataset):
        """
        :param dataset: 数据集
        :param minpts: 最少点个数
        :param eps: 半径
        :return: 样本的簇集合
        """
        # 定义一个容器 用于存储样本的分类
        clusters = np.full((N, 1), -1)
        k = -1
        classifer = []
        for i in range(N):

            if clusters[i] != -1:
                continue
            else:
                # 取出样本点
                p = dataset[i]
                subdataset = [j for j in range(N) if self.distance(dataset[j], p) <= self.eps]

                if len(subdataset) < self.minpts:
                    continue
                else:
                    k += 1
                    clusters[i] = k
                    for j in subdataset:
                        clusters[j] = k
                        if j > i:
                            sub = [item for item in range(N) if self.distance(dataset[item], dataset[j]) <= self.eps]
                            if len(sub) >= self.minpts:
                                for t in sub:
                                    if t not in subdataset:
                                        subdataset.append(t)

            classifer.append(subdataset)
        print(len(classifer))

        # 可视化
        plt.scatter(dataset[classifer[0]][:, 0], dataset[classifer[0]][:, 1], color="red")
        plt.scatter(dataset[classifer[1]][:, 0], dataset[classifer[1]][:, 1], color="blue")
        plt.scatter(dataset[classifer[2]][:, 0], dataset[classifer[2]][:, 1], color="yellow")
        plt.scatter(dataset[np.argwhere(clusters==-1)[:,0]][:, 0],
                    dataset[np.argwhere(clusters==-1)[:,0]][:, 1], color="black")

        plt.show()

        return clusters


def load_dataset():
    """加载数据集"""
    x1, y1 = datasets.make_circles(n_samples=1000, factor=0.6, noise=0.06, random_state=3)
    x2, y2 = datasets.make_blobs(n_samples=200, n_features=2, centers=[[1.5, 1.5]],
                                 cluster_std=[[0.1]], random_state=2)

    dataset = np.vstack((x1,x2))
    # plt.scatter(dataset[:,0],dataset[: ,1],color="black")
    # plt.show()
    return dataset


dataset = load_dataset()
N, D_in = dataset.shape


if __name__ == '__main__':
    dataset = load_dataset()
    dbscan = DBScan(5, 0.1)
    dbscan.fit(dataset)
