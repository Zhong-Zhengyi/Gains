import torch.nn as nn
import torch.nn.functional as F
import torch

class AmazonFullLSTM(nn.Module):
    def __init__(self, data_parallel=True):
        super(AmazonFullLSTM, self).__init__()
        self.encoder = AmazonLSTM(data_parallel)
        self.classifier = AmazonClassifier(data_parallel)

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

        self.global_projector = nn.Linear(400, 256)
        self.local_projector = nn.Linear(400, 128) 

        self.classifier = AmazonClassifier(data_parallel)

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

        self.reshape = nn.Linear(self.input_size, self.input_size)

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=True
        )

        self.fc = nn.Linear(self.hidden_size * 2, self.hidden_size)

        if data_parallel:
            self.reshape = nn.DataParallel(self.reshape)
            self.lstm = nn.DataParallel(self.lstm)
            self.fc = nn.DataParallel(self.fc)

    def forward(self, x):

        batch_size = x.size(0)
        x = self.reshape(x)
        x = x.unsqueeze(1)

        lstm_out, _ = self.lstm(x)

        lstm_out = lstm_out[:, -1, :]

        feature = self.fc(lstm_out)
        
        return feature


class AmazonClassifier(nn.Module):
    def __init__(self, data_parallel=True):
        super(AmazonClassifier, self).__init__()
        linear = nn.Sequential()
        linear.add_module("fc", nn.Linear(400, 2))
        # self.fc = nn.Linear(400, 2)

        if data_parallel:
            self.linear = nn.DataParallel(linear)
        else:
            self.linear = linear
            # self.fc = nn.DataParallel(self.fc)
    def forward(self, x):
        x = self.linear(x)
        return x

