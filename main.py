'''
程序调用入口

Author: Bruce Hou, Email: ecstayalive@163.com
'''

import torch
from CapsuleNet import *


class CreDataset(torch.utils.data.Dataset):
    '''
    创建数据集
    '''

    def __init__(self, data, label):
        import numpy as np
        self.data = np.float32(data)
        self.label = np.int64(label)

    def __len__(self):
        return int(self.data.shape[0])

    def __getitem__(self, index):
        return self.data[index], self.label[index]


def Load(train_dataset, test_dataset, batch_size=10):
    """
    加载数据
    :param train_dataset:
    :param test_dataset:
    :param batch_size: batch size
    :return: train_loader, test_loader
    """
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader


def fft(x):
    '''
    对数据进行离散傅里叶变换，在属于对数据集的预处理
    前后不改变数据的格式。这一步是可以选的，即输入数
    据可以是不是傅里叶变换后的
    :param x: 数据集
    :return:
    '''
    from scipy.fftpack import fft

    N = x.shape[1]  # 序列长度
    result = np.empty((x.shape[0], x.shape[1] // 2))  # 经过离散傅里叶变换后的结果
    for i in range(x.shape[0]):
        temp = np.abs(fft(x[i, :])) / N
        result[i] = temp[0:temp.shape[0] // 2]

    return result.reshape((result.shape[0], 32, 32))


if __name__ == '__main__':
    import numpy as np
    from sklearn.model_selection import KFold

    # 加载数据
    data, label = np.load("./dataset/train.npy"), np.load("./dataset/label.npy")
    # print("train.shape:", train.shape)
    # print("label.shape:", label.shape)

    # 原始数据经过离散傅里叶变换
    data, label = fft(data), label

    # 四折交叉验证
    kf = KFold(n_splits=4)
    count = 0
    for train_index, test_index in kf.split(data):
        # 创建验证集和数据集
        train_data = data[train_index, :, :]
        train_label = label[train_index]
        test_data = data[test_index, :, :]
        test_label = label[test_index]

        # 建立数据集合
        train_dataset = CreDataset(train_data, train_label)
        test_dataset = CreDataset(test_data, test_label)

        train_loader, test_loader = Load(train_dataset, test_dataset, batch_size=10)

        # 创建模型
        model = CapsuleNet(input_size=[1, 32, 32], classes=8, routings=3)
        print(model)
        model.cuda()
        # 训练
        train(model, train_loader, test_loader, count, 20)
        count += 1
