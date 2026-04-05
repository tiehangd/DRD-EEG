import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


class BaseDecoderSEED(nn.Module):
    def __init__(self):
        super(BaseDecoderSEED, self).__init__()
        num_cls=3
        self.conv1 = nn.Conv2d(3, 16, (1, 37), padding = 0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))
        self.pre_fc = nn.Sequential(nn.Linear(96, 20))
        self.linear = nn.Sequential(nn.Linear(20, num_cls))
        
    def return_hidden(self, x):
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = self.pooling2(x)
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = self.pooling3(x)
        x = x.view(-1, 96)
        x = self.pre_fc(x)
        return x

    def forward(self, x):
        x = self.return_hidden(x)
        x = F.softmax(self.linear(x), dim=1)
        return x

