import pickle as pkl
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from importlib import reload


def get_new_size(old_size, kernel_size=3, padding=0, dilation=1, stride=1):

    dim = len(old_size)

    if type(kernel_size) is not tuple:
        kernel_size = tuple([kernel_size] * dim)
    if type(stride) is not tuple:
        stride = tuple([stride] * dim)
    if type(stride) is not tuple:
        stride = tuple([stride] * dim)
    if type(padding) is not tuple:
        padding = tuple([padding] * dim)
    if type(dilation) is not tuple:
        dilation = tuple([dilation] * dim)

    # print(kernel_size, stride, padding, dilation)
    res = []
    for d in range(dim):
        cur_dim = (old_size[d] + (2 * padding[d]) - (dilation[d] * (kernel_size[d] - 1)) - 1) / (stride[d]) + 1
        res.append(int(np.floor(cur_dim)))

    return tuple(res)


class pianoRollConvNet(nn.Module):
    def __init__(self, img_size):
        super(pianoRollConvNet, self).__init__()
        layer1_chan = 50
        layer2_chan = 50
        layer3_chan = 50
        hidden_layer = 50

        conv1_kwargs = {'kernel_size': (3, 3, 2), 'padding': (1, 1, 0), 'dilation': (1, 1, 1)}
        conv2_kwargs = {'kernel_size': (3, 3), 'padding': (1, 1), 'dilation': (2, 1)}
        conv3_kwargs = {'kernel_size': (3, 3), 'padding': (1, 1), 'dilation': (4, 1)}

        self.layer1 = nn.Sequential(
            nn.Conv3d(1, layer1_chan, **conv1_kwargs),
            nn.ReLU(),
        )
        # SQUEEZE LAST DIMENSION HERE! 3D -> 2D

        self.layer2 = nn.Sequential(
            nn.BatchNorm2d(layer1_chan),
            nn.Conv2d(layer1_chan, layer2_chan, **conv2_kwargs),
            nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
            nn.BatchNorm2d(layer1_chan),
            nn.Conv2d(layer2_chan, layer3_chan, **conv3_kwargs),
            nn.ReLU(),
        )

        # self.drop_out = nn.Dropout()
        self.drop_out_2d = nn.Dropout2d()

        new_size = get_new_size(img_size, **conv1_kwargs)
        new_size = new_size[:-1]
        new_size = get_new_size(new_size, **conv2_kwargs)
        new_size = get_new_size(new_size, **conv3_kwargs)

        self.fc1 = nn.Linear(np.product(new_size) * layer3_chan, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = self.layer1(x)
        out = out.squeeze(-1)
        out = self.drop_out_2d(out)
        out = self.layer2(out)
        out = self.drop_out_2d(out)
        out = self.layer3(out)
        out = self.drop_out_2d(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.sig(out)
        return out


if __name__ == '__main__':
    sz = (80, 80, 2)
    asdf = torch.tensor(np.random.randint(2, size=sz), dtype=torch.float)
    asdf = asdf[None, None, :, :, :]
    #
    # conv1_kwargs = {'kernel_size': (3, 3, 2), 'padding': (1, 1, 0), 'dilation': (1, 1, 1)}
    # conv2_kwargs = {'kernel_size': (3, 3), 'padding': (1, 1), 'dilation': (2, 1)}
    # conv3_kwargs = {'kernel_size': (3, 3), 'padding': (1, 1), 'dilation': (4, 1)}
    # layertest = nn.Conv3d(1, 1, **conv1_kwargs)
    # layertest2 =  nn.Conv2d(1, 1, **conv2_kwargs)
    # layertest3 = nn.Conv2d(1, 1, **conv3_kwargs)
    #
    # res = layertest(asdf)
    # res = res.squeeze(-1)
    # res = layertest2(res)
    # res = layertest3(res)

    net = pianoRollConvNet(sz)

    res = net.forward(asdf)

