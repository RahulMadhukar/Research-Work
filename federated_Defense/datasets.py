import os
import urllib.request
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np



_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = os.path.join(_THIS_DIR, 'data')

SHAKESPEARE_CHARS = (
    'abcdefghijklmnopqrstuvwxyz'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    '0123456789'
    " \n\t.,;:!?'\"-()/[]&"
)
SHAKESPEARE_SEQ_LEN = 80

class _ShakespeareDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (torch.tensor(self.sequences[idx], dtype=torch.float32),
                torch.tensor(self.targets[idx], dtype=torch.long))

SENTIMENT140_VOCAB_SIZE = 30000
SENTIMENT140_MAX_WORDS  = 25

class _Sentiment140Dataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets   = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (torch.tensor(self.sequences[idx], dtype=torch.float32),
                torch.tensor(self.targets[idx],   dtype=torch.long))

def load_femnist(data_dir=None, download=True):
    import torchvision
    data_dir = data_dir or _DEFAULT_DATA_DIR
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1751,), (0.3332,))
    ])
    trainset = torchvision.datasets.EMNIST(
        root=data_dir, split='byclass', train=True, download=download, transform=transform
    )
    testset = torchvision.datasets.EMNIST(
        root=data_dir, split='byclass', train=False, download=download, transform=transform
    )
    return trainset, testset

def load_shakespeare(data_dir=None, download=True, seq_len=SHAKESPEARE_SEQ_LEN):
    data_dir = data_dir or _DEFAULT_DATA_DIR
    txt_path = os.path.join(data_dir, 'shakespeare', 'tinyshakespeare.txt')
    if not os.path.exists(txt_path):
        if not download:
            raise FileNotFoundError(f"Shakespeare text not found at {txt_path}. Pass download=True to fetch it.")
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        print(f"[INFO] Downloading Shakespeare dataset...")
        urllib.request.urlretrieve(url, txt_path)
        print(f"[INFO] Saved to {txt_path}")
    with open(txt_path, 'r') as f:
        text = f.read()
    char_to_idx = {c: i for i, c in enumerate(SHAKESPEARE_CHARS)}
    indices = np.array([char_to_idx[c] for c in text if c in char_to_idx], dtype=np.int64)
    n_samples = len(indices) - seq_len
    sequences = np.stack([indices[i:i + seq_len] for i in range(n_samples)])
    targets   = indices[seq_len:seq_len + n_samples]
    split = int(0.9 * n_samples)
    trainset = _ShakespeareDataset(sequences[:split], targets[:split])
    testset  = _ShakespeareDataset(sequences[split:], targets[split:])
    print(f"[INFO] Shakespeare — train: {len(trainset)}, test: {len(testset)}, classes: {len(SHAKESPEARE_CHARS)}")
    return trainset, testset

def _import_hf_datasets():
    import sys, importlib
    _here = os.path.dirname(os.path.abspath(__file__))
    _import_hf_datasets._local_mod = sys.modules.pop('datasets', None)
    orig_path = sys.path[:]
    sys.path = [p for p in sys.path if os.path.abspath(p) != _here]
    try:
        hf = importlib.import_module('datasets')
        return hf
    except ImportError:
        if _import_hf_datasets._local_mod is not None:
            sys.modules['datasets'] = _import_hf_datasets._local_mod
        raise ImportError("Sentiment140 requires the HuggingFace 'datasets' package:\n    pip install datasets\n")
    finally:
        sys.path = orig_path

def _restore_local_datasets():
    import sys
    local_mod = getattr(_import_hf_datasets, '_local_mod', None)
    if local_mod is not None:
        sys.modules['datasets'] = local_mod
        _import_hf_datasets._local_mod = None

def load_sentiment140(data_dir=None, download=True,
                      max_words=SENTIMENT140_MAX_WORDS,
                      vocab_size=SENTIMENT140_VOCAB_SIZE):
    import json
    from collections import Counter
    data_dir = data_dir or _DEFAULT_DATA_DIR
    cache_dir  = os.path.join(data_dir, 'sentiment140')
    train_cache = os.path.join(cache_dir, 'train.npz')
    test_cache  = os.path.join(cache_dir, 'test.npz')
    if os.path.exists(train_cache) and os.path.exists(test_cache):
        train_data = np.load(train_cache)
        test_data  = np.load(test_cache)
        trainset = _Sentiment140Dataset(train_data['X'], train_data['y'])
        testset  = _Sentiment140Dataset(test_data['X'],  test_data['y'])
        print(f"[INFO] Sentiment140 loaded from cache — train: {len(trainset)}, test: {len(testset)}")
        return trainset, testset
    if not download:
        raise FileNotFoundError(f"Sentiment140 cache not found in {cache_dir}. Pass download=True to fetch it.")
    hf = _import_hf_datasets()
    hf.disable_progress_bar()
    print("[INFO] Downloading Sentiment140 (1.6M tweets) — this may take several minutes...")
    ds = hf.load_dataset("sentiment140", split="train")
    _restore_local_datasets()
    label_col = 'sentiment' if 'sentiment' in ds.column_names else 'target'
    unique_vals = sorted(set(ds[label_col]))
    neg_val, pos_val = min(unique_vals), max(unique_vals)
    ds = ds.filter(lambda x: x[label_col] in [neg_val, pos_val])
    texts  = ds['text']
    labels = np.array([0 if t == neg_val else 1 for t in ds[label_col]], dtype=np.int64)
    try:
        hf.enable_progress_bar()
    except Exception:
        pass
    print(f"[INFO] {len(texts)} binary samples (neg: {(labels == 0).sum()}, pos: {(labels == 1).sum()})")
    counter = Counter()
    for text in texts:
        counter.update(text.lower().split())
    top_words    = [w for w, _ in counter.most_common(vocab_size)]
    word_to_idx  = {w: i + 1 for i, w in enumerate(top_words)}
    X = np.zeros((len(texts), max_words), dtype=np.int32)
    for i, text in enumerate(texts):
        words = text.lower().split()[:max_words]
        for j, w in enumerate(words):
            X[i, j] = word_to_idx.get(w, 0)
    y = labels
    split_idx   = int(0.9 * len(X))
    X_train, X_test = X[:split_idx],  X[split_idx:]
    y_train, y_test = y[:split_idx],  y[split_idx:]
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(train_cache, X=X_train, y=y_train)
    np.savez(test_cache,  X=X_test,  y=y_test)
    with open(os.path.join(cache_dir, 'vocab.json'), 'w') as f:
        json.dump(top_words, f)
    print(f"[INFO] Sentiment140 cached — train: {len(X_train)}, test: {len(X_test)}, vocab: {vocab_size + 1}")
    return _Sentiment140Dataset(X_train, y_train), _Sentiment140Dataset(X_test, y_test)

def partition_dataset(dataset, num_clients, iid=True, alpha=0.5):
    data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    X, y = next(iter(data_loader))
    X, y = X.numpy(), y.numpy()
    n_samples = len(dataset)
    client_data = []
    if iid:
        indices = np.random.permutation(n_samples)
        splits = np.array_split(indices, num_clients)
        for split in splits:
            client_data.append((X[split], y[split]))
    else:
        unique_classes = np.unique(y)
        num_classes = len(unique_classes)
        class_indices = {c: np.where(y == c)[0] for c in unique_classes}
        min_samples = max(1, n_samples // (num_clients * 10))
        client_indices = [[] for _ in range(num_clients)]
        for c in unique_classes:
            idx = class_indices[c]
            np.random.shuffle(idx)
            proportions = np.random.dirichlet([alpha] * num_clients)
            counts = (proportions * len(idx)).astype(int)
            remainder = len(idx) - counts.sum()
            if remainder > 0:
                fracs = (proportions * len(idx)) - counts
                top_ids = np.argsort(-fracs)[:remainder]
                counts[top_ids] += 1
            start = 0
            for cid in range(num_clients):
                end = start + counts[cid]
                if end > start:
                    client_indices[cid].extend(idx[start:end].tolist())
                start = end
        all_indices_flat = np.arange(n_samples)
        np.random.shuffle(all_indices_flat)
        spare_ptr = 0
        for cid in range(num_clients):
            idx = np.array(client_indices[cid], dtype=int)
            if len(idx) < min_samples:
                need = min_samples - len(idx)
                extra = all_indices_flat[spare_ptr:spare_ptr + need]
                spare_ptr += need
                if len(extra) > 0:
                    idx = np.concatenate([idx, extra])
            client_data.append((X[idx], y[idx]))
    return client_data