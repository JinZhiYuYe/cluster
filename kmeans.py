# -*- coding=utf-8 -*- 
# time = '2020/4/11 16:28'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_dataset():
    """随机数据"""
    # np.random.seed(2)
    x1 = np.random.randint(1, 100, size=(20, 2))
    x2 = np.random.randint(10, 110, size=(30, 2))
    dataset = np.vstack((x1, x2))
    plt.scatter(dataset[:, 0], dataset[:, 1], color="black")
    plt.show()
    print(dataset.shape)
    return dataset


def distance(x, y):
    return np.square(x - y).sum()


def center(dataset):
    """初始簇中心"""
    df = pd.DataFrame(dataset)
    k = np.array(df.sample(2))
    return k


def kmeans(dataset, k):
    """
    :param dataset: 数据集
    :param k: 簇心点
    :return:
    """
    center_1 = k[0]
    center_2 = k[1]
    k1_list = []
    k2_list = []

    for i in range(dataset.shape[0]):
        if distance(center_1, dataset[i]) < distance(center_2, dataset[i]):
            k1_list.append(i)
        else:
            k2_list.append(i)

    # 计算平均值
    center_1_mean = np.true_divide(np.sum(dataset[k1_list], axis=0), len(k1_list))
    center_2_mean = np.true_divide(np.sum(dataset[k2_list], axis=0), len(k2_list))
    k = np.array([center_1_mean, center_2_mean])
    # print(k)
    if (center_1 == center_1_mean).all() and (center_2 == center_2_mean).all():
        print(center_1, center_2)
        plt.scatter(dataset[k1_list][1:, 0], dataset[k1_list][1:, 1], color="red")
        plt.scatter(dataset[k2_list][1:, 0], dataset[k2_list][1:, 1], color="blue")
        plt.scatter(center_1[0], center_1[1], marker="+", color="red", s=500)
        plt.scatter(center_2[0], center_2[1], marker="+", color="blue", s=500)
        plt.show()
    else:
        print(center_1, center_2)
        return kmeans(dataset, k)


if __name__ == '__main__':
    dataset = load_dataset()
    centerK = center(dataset)
    kmeans(dataset, centerK)
