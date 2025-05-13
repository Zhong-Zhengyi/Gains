import torch.nn as nn
import torch.nn.functional as F
import torch

class AmazonFullLSTM(nn.Module):
    def __init__(self, data_parallel=True):
        super(AmazonFullLSTM, self).__init__()
        self.encoder = AmazonLSTM(data_parallel)
        self.classifier = AmazonClassifier(data_parallel)
        
        # 正确应用数据并行
        if data_parallel:
            self.encoder = nn.DataParallel(self.encoder)
            self.classifier = nn.DataParallel(self.classifier)

    def forward(self, x):
        features = self.encoder(x)
        output = self.classifier(features)
        return features, output
    
class AmazonOurModel(nn.Module):
    def __init__(self, data_parallel=True):
        super(AmazonOurModel, self).__init__()
        self.encoder = AmazonLSTM(data_parallel)
        
        # 投影矩阵
        self.global_projector = nn.Linear(400, 256)  # 共性特征
        self.local_projector = nn.Linear(400, 128)   # 个性化特征

        self.classifier = AmazonClassifier(data_parallel)
        
        # 正确应用数据并行
        if data_parallel:
            self.encoder = nn.DataParallel(self.encoder)
            self.global_projector = nn.DataParallel(self.global_projector)
            self.local_projector = nn.DataParallel(self.local_projector)
            self.classifier = nn.DataParallel(self.classifier)
    
    def forward(self, x): 
        features = self.encoder(x) 
        feature_g = self.global_projector(features) 
        feature_l = self.local_projector(features) 
        combined = torch.cat([feature_g, feature_l], dim=1) 
        output = self.classifier(combined)
        return [feature_g, feature_l], output
    
class AmazonOurClassifier(nn.Module):
    def __init__(self, data_parallel=True):
        super(AmazonOurClassifier, self).__init__()
        self.fc = nn.Linear(256+128, 2)
        
        # 正确应用数据并行
        if data_parallel:
            self.fc = nn.DataParallel(self.fc)

    def forward(self, x):
        x = self.fc(x)
        return x
    

class AmazonLSTM(nn.Module):
    def __init__(self, data_parallel=True):
        super(AmazonLSTM, self).__init__()
        self.input_size = 5000
        self.hidden_size = 400
        self.num_layers = 2
        self.dropout = 0.5
        
        # 将输入数据重塑为序列形式
        self.reshape = nn.Linear(self.input_size, self.input_size)
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 全连接层，将LSTM输出映射到特征空间
        self.fc = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
        # 正确应用数据并行
        if data_parallel:
            self.reshape = nn.DataParallel(self.reshape)
            self.lstm = nn.DataParallel(self.lstm)
            self.fc = nn.DataParallel(self.fc)

    def forward(self, x):
        # 重塑输入以适应LSTM
        batch_size = x.size(0)
        x = self.reshape(x)
        x = x.unsqueeze(1)  # 添加序列长度维度
        
        # 通过LSTM
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]
        
        # 通过全连接层
        feature = self.fc(lstm_out)
        
        return feature


class AmazonClassifier(nn.Module):
    def __init__(self, data_parallel=True):
        super(AmazonClassifier, self).__init__()
        linear = nn.Sequential()
        linear.add_module("fc", nn.Linear(400, 2))
        # self.fc = nn.Linear(400, 2)
        
        # 正确应用数据并行
        if data_parallel:
            self.linear = nn.DataParallel(linear)
        else:
            self.linear = linear
            # self.fc = nn.DataParallel(self.fc)
    def forward(self, x):
        x = self.linear(x)
        return x

