'''
Degenerate capsule neural network.
Reference Papers: https://arxiv.org/abs/1710.09829
Reference Code： https://github.com/XifengGuo/CapsNet-Pytorch

Author: Bruce Hou, Email: ecstayalive@163.com
'''

import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from CapsuleLayer import DenseCapsule, PrimaryCapsule


class CapsuleNet(nn.Module):
    """
    A Capsule Network.
    :param input_size: data size = [channels, width, height]
    :param classes: number of classes
    :param routings: number of routing iterations
    Shape:
        - Input: (batch, channels, width, height), optional (batch, classes) .
        - Output:((batch, classes), (batch, channels, width, height))
    """

    def __init__(self, input_size, classes, routings):
        super(CapsuleNet, self).__init__()
        self.input_size = input_size
        self.classes = classes
        self.routings = routings

        # Layer 1: Just a conventional Conv2D layer
        self.conv1 = nn.Conv2d(input_size[0], 256, kernel_size=9, stride=1, padding=0)

        self.max_pool = nn.MaxPool2d(3)
        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_caps, dim_caps]
        self.primarycaps = PrimaryCapsule(256, 256, 8, kernel_size=9, stride=2, padding=0)

        # Layer 3: Capsule layer. Routing algorithm works here.
        # self.digitcaps = DenseCapsule(in_num_caps=32 * 6 * 6, in_dim_caps=8,
        #                               out_num_caps=classes, out_dim_caps=16, routings=routings)
        self.digitcaps = DenseCapsule(in_num_caps=2048, in_dim_caps=8,
                                      out_num_caps=classes, out_dim_caps=1, routings=routings)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        # print("经过巻积层1后的x形状", x.size())
        # x = self.max_pool(x)
        # print("经过池化层后的x形状", x.size())
        x = self.primarycaps(x)
        # print("经过初始胶囊层后的x形状", x.size())
        x = self.digitcaps(x)
        # print("经过胶囊层后的x形状", x.size())
        length = x.norm(dim=-1)
        return length


def caps_loss(y_true, y_pred):
    """
    Capsule loss = Margin loss + lam_recon * reconstruction loss.
    :param y_true: true labels, one-hot coding, size=[batch, classes]
    :param y_pred: predicted labels by CapsNet, size=[batch, classes]
    :param x: input data, size=[batch, channels, width, height]
    :return: Variable contains a scalar loss value.
    """
    #
    L = y_true * torch.clamp(0.9 - y_pred, min=0.) ** 2 + \
        0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.) ** 2
    L_margin = L.sum(dim=1).mean()
    # log_prob = torch.nn.functional.log_softmax(y_pred, dim=1)
    # L_margin = -torch.true_divide(torch.sum(log_prob * y_true), y_true.shape[0])

    return L_margin


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for x, y in test_loader:
        # 由于nn.conv2d输入为4-d向量(batch_size, channels, width, height)
        x = x.unsqueeze(1)
        # one-hot编码
        y = torch.zeros(y.size(0), 8).scatter_(1, y.view(-1, 1), 1.)

        with torch.no_grad():
            x, y = Variable(x.cuda()), Variable(y.cuda())

        y_pred = model(x)
        test_loss += caps_loss(y, y_pred).item() * x.size(0)  # sum up batch loss
        # print(y_pred.data)
        y_pred = y_pred.data.max(1)[1]
        y_true = y.data.max(1)[1]
        # print("y_pred:", y_pred.data)
        # print("y_true:", y_true)
        correct += y_pred.eq(y_true).cpu().sum()

    # print("correct:", correct)
    # print("length:", len(test_loader.dataset))

    test_loss = torch.true_divide(test_loss, len(test_loader.dataset))
    acc = torch.true_divide(correct, len(test_loader.dataset))

    print("acc:", acc)
    print("test loss:", test_loss)
    return test_loss.item(), acc.item()


def train(model, train_loader, test_loader, count, epoch):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param train_loader: torch.utils.data.DataLoader for training data
    :param test_loader: torch.utils.data.DataLoader for test data
    :param count
    :param epoch
    :return: The trained model
    """
    print('Begin Training' + '-' * 70)
    from time import time
    import csv
    logfile = open('./result/log' + str(count) + '.csv', 'w')
    logwriter = csv.DictWriter(logfile, fieldnames=['epoch', 'loss', 'val_loss', 'val_acc'])
    logwriter.writeheader()

    t0 = time()
    optimizer = Adam(model.parameters(), lr=0.001)
    lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    best_val_acc = 0.

    optimizer.step()  # 消除Warning

    # model.train()  # set to training mode
    for epoch in range(epoch):
        model.train()  # set to training mode
        lr_decay.step()  # decrease the learning rate by multiplying a factor `gamma`
        ti = time()
        training_loss = 0.0
        for i, (x, y) in enumerate(train_loader):  # batch training
            # 由于nn.conv2d输入为4-d向量(batch_size, channels, width, height)
            x = x.unsqueeze(1)
            # print("初始x输入形状", x.shape)

            # change to one-hot coding（改变标签使其变成one-hot编码）
            y = torch.zeros(y.size(0), 8).scatter_(1, y.view(-1, 1), 1.)

            x, y = Variable(x.cuda()), Variable(y.cuda())  # convert input data to GPU Variable

            optimizer.zero_grad()  # set gradients of optimizer to zero
            y_pred = model(x)  # forward
            # print("输出的y", y_pred)
            loss = caps_loss(y, y_pred)  # compute loss

            loss.backward()  # backward, compute all gradients of loss w.r.t all Variables
            training_loss += loss.item() * x.size(0)  # record the batch loss
            optimizer.step()  # update the trainable parameters with computed gradients
            # lr_decay.step()  # decrease the learning rate by multiplying a factor `gamma`
        # compute validation loss and acc
        val_loss, val_acc = test(model, test_loader)
        logwriter.writerow(dict(epoch=epoch, loss=training_loss / len(train_loader.dataset),
                                val_loss=val_loss, val_acc=val_acc))
        print("==> Epoch %02d: loss=%.5f, val_loss=%.5f, val_acc=%.4f, time=%ds"
              % (epoch, training_loss / len(train_loader.dataset),
                 val_loss, val_acc, time() - ti))
        if val_acc > best_val_acc:  # update best validation acc and save model
            best_val_acc = val_acc
            torch.save(model.state_dict(), './model/epoch%d.pkl' % epoch)
            print("best val_acc increased to %.4f" % best_val_acc)
    logfile.close()
    torch.save(model.state_dict(), './model/trained_model.pkl')
    print('Trained model saved to \'./model/trained_model.h5\'')
    print("Total time = %ds" % (time() - t0))
    return model
    print('End Training' + '-' * 70)


if __name__ == "__main__":
    pass
