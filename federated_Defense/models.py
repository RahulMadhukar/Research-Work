import torch.nn as nn
import torch.nn.functional as F

def get_model_size_bytes(model):
    """Return total model parameter size in bytes (float32 = 4 bytes per param)."""
    return sum(p.numel() * p.element_size() for p in model.parameters())

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
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        features = F.relu(self.fc1(x))
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
        x = x.long()
        x = self.embedding(x)
        output, _ = self.lstm(x)
        last_out = output[:, -1, :]
        logits = self.fc(last_out)
        return logits

    def get_features(self, x):
        x = x.long()
        x = self.embedding(x)
        output, _ = self.lstm(x)
        return output[:, -1, :]

class Sentiment140Net(nn.Module):
    """2-layer LSTM binary classifier for Sentiment140 tweets."""
    def __init__(self, vocab_size=30001, embed_dim=300, hidden_size=256, num_classes=2):
        super(Sentiment140Net, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size,
                            num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.long()
        x = self.embedding(x)
        output, _ = self.lstm(x)
        last_out = output[:, -1, :]
        logits = self.fc(last_out)
        return logits

    def get_features(self, x):
        x = x.long()
        x = self.embedding(x)
        output, _ = self.lstm(x)
        return output[:, -1, :]

    def load_glove_embeddings(self, glove_path, word_to_idx):
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
        self.embedding.weight.requires_grad = False
        print(f"[GloVe] Loaded {found}/{len(word_to_idx)} word vectors from {glove_path}")