import torch.nn as nn
import torch.nn.functional as F
import torch

class CNNFullModel(nn.Module):
    def __init__(self, data_parallel=True):
        super(CNNFullModel, self).__init__()
        self.encoder = CNN(data_parallel=data_parallel)
        self.classifier = Classifier(data_parallel=data_parallel)

    def forward(self, x):
        features = self.encoder(x)
        output = self.classifier(features)
        return features, output
    
class CNNOurModel(nn.Module):
    def __init__(self, data_parallel=True):
        super(CNNOurModel, self).__init__()
        self.encoder = CNN(data_parallel=data_parallel)
        # 投影矩阵
        self.global_projector = nn.Linear(2048, 256)  # 共性特征
        self.local_projector = nn.Linear(2048, 128)   # 个性化特征
        self.classifier = Classifier(data_parallel=data_parallel)

    def forward(self, x):
        features = self.encoder(x) 
        # print('encoder', features)
        feature_g = self.global_projector(features)
        # print('feature_g', feature_g) 
        feature_l = self.local_projector(features)
        # print('feature_l', feature_l) 
        combined = torch.cat([feature_g, feature_l], dim=1) 
        output = self.classifier(combined)
        return [feature_g, feature_l], output

class CNN(nn.Module):
    def __init__(self, data_parallel=True):
        super(CNN, self).__init__()
        encoder = nn.Sequential()
        encoder.add_module("conv1", nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2))
        encoder.add_module("bn1", nn.BatchNorm2d(64))
        encoder.add_module("relu1", nn.ReLU())
        encoder.add_module("maxpool1", nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False))
        encoder.add_module("conv2", nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2))
        encoder.add_module("bn2", nn.BatchNorm2d(64))
        encoder.add_module("relu2", nn.ReLU())
        encoder.add_module("maxpool2", nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False))
        encoder.add_module("conv3", nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2))
        encoder.add_module("bn3", nn.BatchNorm2d(128))
        encoder.add_module("relu3", nn.ReLU())
        if data_parallel:
            self.encoder = nn.DataParallel(encoder)
        else:
            self.encoder = encoder
        linear = nn.Sequential()
        linear.add_module("fc1", nn.Linear(8192, 3072))
        linear.add_module("bn4", nn.BatchNorm1d(3072))
        linear.add_module("relu4", nn.ReLU())
        linear.add_module("dropout", nn.Dropout())
        linear.add_module("fc2", nn.Linear(3072, 2048))
        linear.add_module("bn5", nn.BatchNorm1d(2048))
        linear.add_module("relu5", nn.ReLU())
        if data_parallel:
            self.linear = nn.DataParallel(linear)
        else:
            self.linear = linear

    def forward(self, x):
        batch_size = x.size(0)
        feature = self.encoder(x)
        feature = feature.view(batch_size, -1)#8192
        feature = self.linear(feature)
        return feature

class Classifier(nn.Module):
    def __init__(self, data_parallel=True):
        super(Classifier, self).__init__()
        linear = nn.Sequential()
        linear.add_module("fc", nn.Linear(2048, 10))
        if data_parallel:
            self.linear = nn.DataParallel(linear)
        else:
            self.linear = linear

    def forward(self, x):
        x = self.linear(x)
        return x