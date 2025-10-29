import torch.nn as nn
import torch.nn.functional as F


class FashionMNISTNet(nn.Module):
    """CNN for Fashion-MNIST (28x28, 1 channel)"""
    def __init__(self, num_classes=10):
        super(FashionMNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        features = F.relu(self.fc1(x))
        x = self.dropout(features)
        x = self.fc2(x)
        return x
    
    def get_features(self, x):
        """Extract features for analysis"""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        features = F.relu(self.fc1(x))
        return features

class CIFAR10Net(nn.Module):
    """CNN for CIFAR-10 (32x32, 3 channels)"""
    def __init__(self, num_classes=10):
        super(CIFAR10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        features = F.relu(self.fc2(x))
        x = self.dropout(features)
        x = self.fc3(x)
        return x
    
    def get_features(self, x):
        """Extract features for analysis"""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        features = F.relu(self.fc2(x))
        return features