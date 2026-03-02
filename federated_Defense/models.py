import torch.nn as nn
import torch.nn.functional as F


def get_model_size_bytes(model):
    """Return total model parameter size in bytes (float32 = 4 bytes per param)."""
    return sum(p.numel() * p.element_size() for p in model.parameters())


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


class FEMNISTNet(nn.Module):
    """CNN for FEMNIST (28x28, 1 channel, 62 classes)"""
    def __init__(self, num_classes=62):
        super(FEMNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 2048)
        self.fc2 = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        features = F.relu(self.fc1(x))
        x = self.fc2(features)
        return x

    def get_features(self, x):
        """Extract features for analysis"""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        features = F.relu(self.fc1(x))
        return features


class EMNISTNet(nn.Module):
    """CNN for EMNIST (28x28, 1 channel, 47 classes for 'balanced' split)"""
    def __init__(self, num_classes=47):
        super(EMNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 3 * 3)
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
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        features = F.relu(self.fc2(x))
        return features


class ShakespeareNet(nn.Module):
    """2-layer LSTM for Shakespeare next-character prediction (80 classes, seq_len=80)"""
    def __init__(self, num_classes=80, embed_dim=8, hidden_size=100, num_layers=2):
        super(ShakespeareNet, self).__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.long()                        # float indices from client -> long for embedding
        x = self.embedding(x)               # (batch, seq_len, embed_dim)
        output, _ = self.lstm(x)            # (batch, seq_len, hidden_size)
        last_out = output[:, -1, :]         # (batch, hidden_size)
        logits = self.fc(last_out)          # (batch, num_classes)
        return logits

    def get_features(self, x):
        """Extract last LSTM hidden state as features for analysis"""
        x = x.long()
        x = self.embedding(x)
        output, _ = self.lstm(x)
        return output[:, -1, :]


class Sentiment140Net(nn.Module):
    """2-layer LSTM binary classifier for Sentiment140 tweets (CMFL paper Section 6.1.1).
    - Embedding : vocab_size x 300D  (pretrained 300D GloVe [53])
    - LSTM      : 300 -> 256 hidden, 2 layers
    - FC        : 256 -> 2    (negative / positive)
    """
    def __init__(self, vocab_size=30001, embed_dim=300, hidden_size=256, num_classes=2):
        super(Sentiment140Net, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size,
                            num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.long()                        # float word indices -> long for embedding
        x = self.embedding(x)               # (batch, seq_len, 300)
        output, _ = self.lstm(x)            # (batch, seq_len, 256)
        last_out = output[:, -1, :]         # (batch, 256)
        logits = self.fc(last_out)          # (batch, 2)
        return logits

    def get_features(self, x):
        """Extract last LSTM hidden state as features for analysis"""
        x = x.long()
        x = self.embedding(x)
        output, _ = self.lstm(x)
        return output[:, -1, :]

    def load_glove_embeddings(self, glove_path, word_to_idx):
        """Load pretrained 300D GloVe embeddings (paper [53]: Pennington et al.).

        Args:
            glove_path: Path to glove.6B.300d.txt file
            word_to_idx: dict mapping word -> index in our vocabulary
        """
        import numpy as np
        embed_dim = self.embedding.embedding_dim
        pretrained = np.zeros((self.embedding.num_embeddings, embed_dim), dtype=np.float32)
        found = 0
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.rstrip().split(' ')
                word = parts[0]
                if word in word_to_idx:
                    idx = word_to_idx[word]
                    pretrained[idx] = np.array(parts[1:], dtype=np.float32)
                    found += 1
        import torch
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained))
        self.embedding.weight.requires_grad = False  # Freeze pretrained embeddings
        print(f"[GloVe] Loaded {found}/{len(word_to_idx)} word vectors from {glove_path}")