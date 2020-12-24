'''
This project is derived from the course work
and is an extension of the course work. Due
to the source of the dataset itself, the
dataset needs to be pre-processed before it
can be called.

Author: Bruce Hou, Email: ecstayalive@163.com
'''

import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
import os


class PreProcess:
    '''
    加载数据并保存成npy格式
    '''

    def __init__(self):
        pass

    def run(self):
        '''
        调用接口
        :return 采样数据，数据对应的标签
        '''
        data = self.load()
        # 判断文件是否存在
        files_exist = os.path.isfile('./dataset/train.npy')
        if not files_exist:
            print('files does not exist')

            # 数据集大小
            train_data = np.empty((1600, 2048))
            label = np.empty((1600,))
            # 转换数据格式并保存
            train_data, label = self.transform(data, train_data, label)
            return train_data, label

        else:
            print('files exist, now load')
            train_data = np.load('./dataset/train.npy')
            label = np.load('./dataset/label.npy')
            return train_data, label

    def load(self):
        '''
        加载数据
        :return 加载的数据
        '''
        dataset = scio.loadmat("./dataset/lecture_data.mat")
        return dataset

    def transform(self, data, train_data, label):
        '''
        改变格式，生成数据集并保存
        :param data 加载的mat数据
        :param train_data 需要的数据格式和形状
        :param label 数据对应的标签
        :return train_data, label
        '''
        temp1 = np.empty((8, 4096, 80))
        temp2 = np.empty((1, 4096, 160))

        temp = np.empty((320, 2048))

        temp1[0] = data['class0_train_normal']
        temp1[1] = data['class1_train_inner']
        temp1[2] = data['class2_train_outer']
        temp1[3] = data['class3_train_roller']
        temp1[4] = data['class4_train_crack']
        temp1[5] = data['class5_train_pitting']
        temp1[6] = data['class6_train_broken_tooth']
        temp1[7] = data['class7_train_missing_tooth']
        temp2[0] = data['test_data']
        temp3 = np.load('./dataset/result.npy')

        # 生成train_data和label数据集
        for i in range(8):
            for j in range(80):
                train_data[i * 160 + 2 * j, :] = temp1[i, 0:2048, j]
                train_data[i * 160 + 2 * j + 1, :] = temp1[i, 2048:4096, j]
                label[i * 160 + 2 * j:i * 160 + 2 * j + 2] = i

        #
        for i in range(160):
            temp[2 * i, :] = temp2[0, 0:2048, i]
            temp[2 * i + 1, :] = temp2[0, 2048:4096, i]

        for i in range(1280, 1600):
            train_data[i, :] = temp[i - 1280, :]
            label[i] = temp3[(i - 1280) // 2]

        # 打乱训练集和标签
        permutation = np.random.permutation(label.shape[0])
        print(permutation)
        train_data = train_data[permutation, :]
        label = label[permutation]

        np.save('./dataset/or_train.npy', train_data)
        np.save('./dataset/or_label.npy', label)
        # 对每一段序列添加噪声
        for i in range(train_data.shape[0]):
            train_noise = self.gen_gaussian_noise(train_data[i, :], 1)
            train_data[i, :] = train_data[i, :] + train_noise

        # 保存数据
        np.save('./dataset/train.npy', train_data)
        np.save('./dataset/label.npy', label)

        return train_data, label

    def gen_gaussian_noise(self, signal, SNR):
        """
        :param signal: 原始信号
        :param SNR: 添加噪声的信噪比
        :return: 生成的噪声
        """
        noise = np.random.randn(*signal.shape)  # *signal.shape 获取样本序列的尺寸
        # print(signal.shape)
        noise = noise - np.mean(noise)  # np.mean 求均值
        signal_power = (1 / signal.shape[0]) * np.sum(np.power(signal, 2))
        noise_variance = signal_power / np.power(10, (SNR / 10))
        noise = (np.sqrt(noise_variance) / np.std(noise)) * noise
        return noise

if __name__ == '__main__':

    # User's code here
    f = 125600
    load = PreProcess()
    train, label = load.run()
    or_train, or_label = np.load('./dataset/or_train.npy'), np.load('./dataset/or_label.npy')
    # 选取6个数据进行绘图
    # 第一幅图为加入噪声后的数据
    plt.figure(1)
    for i in range(0, 6):
        ax = plt.subplot(3, 2, i + 1)
        ax.set_title(str(label[i]))
        plt.plot(np.arange(2048), train[i, :])
    # 第二幅图为没有加入噪声的数据
    plt.figure(2)
    for i in range(0, 6):
        ax = plt.subplot(3, 2, i + 1)
        ax.set_title(str(or_label[i]))
        plt.plot(np.arange(2048), or_train[i, :])
    plt.show()
