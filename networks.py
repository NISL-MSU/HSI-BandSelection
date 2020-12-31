from abc import ABC

import torch.nn as nn
from torch import reshape


class Hyper3DNetLite(nn.Module, ABC):
    def __init__(self, img_shape=(1, 50, 25, 25), classes=2, data='Kochia'):
        super(Hyper3DNetLite, self).__init__()
        if data == 'Kochia' or data == 'Avocado':
            stride = 2
            out = 7
        else:
            stride = 1
            out = 5
        self.classes = classes
        self.img_shape = img_shape

        self.conv_layer1 = nn.Sequential(nn.Conv3d(in_channels=img_shape[0], out_channels=16, kernel_size=3, padding=1),
                                         nn.ReLU(), nn.BatchNorm3d(16))
        self.conv_layer2 = nn.Sequential(nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
                                         nn.ReLU(), nn.BatchNorm3d(16))
        self.sepconv1 = nn.Sequential(nn.Conv2d(in_channels=16 * img_shape[1], out_channels=16 * img_shape[1],
                                                kernel_size=5, padding=2, groups=16 * img_shape[1]), nn.ReLU(),
                                      nn.Conv2d(in_channels=16 * img_shape[1], out_channels=320,
                                                kernel_size=1, padding=0), nn.ReLU(), nn.BatchNorm2d(320))
        self.sepconv2 = nn.Sequential(nn.Conv2d(in_channels=320, out_channels=320,
                                                kernel_size=3, padding=1, stride=stride, groups=320), nn.ReLU(),
                                      nn.Conv2d(in_channels=320, out_channels=256,
                                                kernel_size=1, padding=0), nn.ReLU(), nn.BatchNorm2d(256))
        self.sepconv3 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256,
                                                kernel_size=3, padding=1, stride=stride, groups=256), nn.ReLU(),
                                      nn.Conv2d(in_channels=256, out_channels=256,
                                                kernel_size=1, padding=0), nn.ReLU(), nn.BatchNorm2d(256))
        self.average = nn.AvgPool2d(kernel_size=out)

        if classes == 2:
            self.fc1 = nn.Linear(256, 1)
        else:
            self.fc1 = nn.Linear(256, self.classes)

    def forward(self, x):

        # 3D Feature extractor
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        # Reshape 3D-2D
        x = reshape(x, (x.shape[0], self.img_shape[1] * 16, self.img_shape[2], self.img_shape[3]))
        # 2D Spatial encoder
        x = self.sepconv1(x)
        x = self.sepconv2(x)
        x = self.sepconv3(x)
        # Global Average Pooling
        x = self.average(x)
        x = reshape(x, (x.shape[0], x.shape[1]))
        if self.classes == 2:
            x = self.fc1(x)
        else:
            x = self.fc1(x)
        return x


class WeedANN(nn.Module, ABC):
    def __init__(self, img_shape=(10,), classes=2):
        super(WeedANN, self).__init__()
        self.classes = classes
        self.img_shape = img_shape

        self.fc1 = nn.Sequential(nn.Linear(in_features=img_shape[0], out_features=500), nn.ReLU(), nn.BatchNorm1d(500))
        self.fc2 = nn.Sequential(nn.Linear(in_features=500, out_features=500), nn.ReLU(), nn.BatchNorm1d(500))

        if classes == 2:
            self.fc3 = nn.Linear(500, 1)
        else:
            self.fc3 = nn.Linear(500, self.classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class SimpleNet(nn.Module, ABC):
    def __init__(self, img_shape=(50, 25, 25), classes=2):
        super(SimpleNet, self).__init__()
        self.classes = classes
        self.img_shape = img_shape

        self.conv_layer1 = nn.Sequential(nn.Conv2d(in_channels=img_shape[0], out_channels=16, kernel_size=3, padding=1),
                                         nn.ReLU(), nn.BatchNorm2d(16))
        self.sepconv1 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=16,
                                                kernel_size=5, padding=2, groups=16), nn.ReLU(),
                                      nn.Conv2d(in_channels=16, out_channels=64,
                                                kernel_size=1, padding=0), nn.ReLU(), nn.BatchNorm2d(64))
        self.sepconv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64,
                                                kernel_size=3, padding=1, stride=2, groups=64), nn.ReLU(),
                                      nn.Conv2d(in_channels=64, out_channels=128,
                                                kernel_size=1, padding=0), nn.ReLU(), nn.BatchNorm2d(128))
        self.sepconv3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128,
                                                kernel_size=3, padding=1, stride=2, groups=128), nn.ReLU(),
                                      nn.Conv2d(in_channels=128, out_channels=256,
                                                kernel_size=1, padding=0), nn.ReLU(), nn.BatchNorm2d(256))
        self.average = nn.AvgPool2d(kernel_size=7)

        if classes == 2:
            self.fc1 = nn.Linear(256, 1)
        else:
            self.fc1 = nn.Linear(256, self.classes)

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.sepconv1(x)
        x = self.sepconv2(x)
        x = self.sepconv3(x)
        # Global Average Pooling
        x = self.average(x)
        x = reshape(x, (x.shape[0], x.shape[1]))
        if self.classes == 2:
            x = self.fc1(x)
        else:
            x = self.fc1(x)
        return x
