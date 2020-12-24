# Degenerate-capsule-neural-network
## 退化胶囊神经网络
退化胶囊神经网络是通过改变极少的胶囊神经网络结构并将其应用与一些对位置要求不敏感的一些识别场合，但是保留了其快速泛化的特性

具体可以参考论文：https://arxiv.org/abs/1710.09829

程序参考代码： https://github.com/XifengGuo/CapsNet-Pytorch

## 使用说明
python main.py

## 代码结构
```
.
├── CapsuleLayer.py                 一些重要的胶囊层的定义
├── CapsuleNet.py                   退化胶囊网络的实现
├── dataset                         存放数据集
│   ├── label.npy
│   ├── lecture_data.mat
│   ├── train.npy
│   ├── with_noise.png
│   └── withnot_noise.png
├── LICENSE
├── main.py                         程序调用接口
├── model
├── Preprocess.py                   
├── README.md
└── Utils.py
```