import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1D(nn.Module):
    def __init__(self, in_channels, out_features):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, 3, padding='same')
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(64, 128, 3, padding='same')
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(128, 64, 3, padding='same')
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(768 + 7 + 1, 256)
        self.fc3 = nn.Linear(256, out_features)

    def forward(self, data, lr, action):
        x = F.relu(self.conv1(data))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.flatten(x)
        x = torch.hstack([x, lr, action])
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x