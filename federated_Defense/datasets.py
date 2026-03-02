import os
import urllib.request

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np

# ---------------------------------------------------------------------------
# Resolve the default data directory relative to THIS file so that loaders
# work regardless of the caller's working directory.
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = os.path.join(_THIS_DIR, 'data')

# ---------------------------------------------------------------------------
# Shakespeare character vocabulary — exactly 80 classes
#   a-z (26) + A-Z (26) + 0-9 (10) + 18 punctuation/whitespace
# ---------------------------------------------------------------------------
SHAKESPEARE_CHARS = (
    'abcdefghijklmnopqrstuvwxyz'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    '0123456789'
    " \n\t.,;:!?'\"-()/[]&"
)
SHAKESPEARE_SEQ_LEN = 80


class _ShakespeareDataset(Dataset):
    """Internal Dataset wrapper for pre-processed Shakespeare sequences."""
    def __init__(self, sequences, targets):
        self.sequences = sequences  # (N, seq_len) int numpy array
        self.targets = targets      # (N,)        int numpy array

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # sequences stored as float so client.py FloatTensor path works;
        # ShakespeareNet.forward() casts back to long for the embedding lookup.
        return (torch.tensor(self.sequences[idx], dtype=torch.float32),
                torch.tensor(self.targets[idx], dtype=torch.long))

# ---------------------------------------------------------------------------
# Sentiment140 constants
#   vocab: top 30 000 words + index 0 reserved for UNK/PAD  →  30 001 total
#   seq_len: 25 words per tweet (matches LEAF benchmark)
# ---------------------------------------------------------------------------
SENTIMENT140_VOCAB_SIZE = 30000
SENTIMENT140_MAX_WORDS  = 25


class _Sentiment140Dataset(Dataset):
    """Internal Dataset wrapper for pre-processed Sentiment140 word-index sequences."""
    def __init__(self, sequences, targets):
        self.sequences = sequences  # (N, max_words) int numpy array
        self.targets   = targets    # (N,)          int numpy array

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # sequences stored as float so client.py FloatTensor path works;
        # Sentiment140Net.forward() casts back to long for the embedding lookup.
        return (torch.tensor(self.sequences[idx], dtype=torch.float32),
                torch.tensor(self.targets[idx],   dtype=torch.long))


def _import_hf_datasets():
    """Import HuggingFace 'datasets' package safely despite this file also being named datasets.py.

    IMPORTANT: The HF module is kept in sys.modules['datasets'] until
    _restore_local_datasets() is called.  This is necessary because
    HF's dataset scripts (e.g. sentiment140.py) internally do
    ``import datasets`` and need the real HF package, not this file.
    """
    import sys, importlib
    _here = os.path.dirname(os.path.abspath(__file__))
    # Save local module so we can restore it later
    _import_hf_datasets._local_mod = sys.modules.pop('datasets', None)
    # Remove project directory from path so Python finds the installed package
    orig_path = sys.path[:]
    sys.path = [p for p in sys.path if os.path.abspath(p) != _here]
    try:
        hf = importlib.import_module('datasets')
        # Keep HF module in sys.modules so HF internals can find it
        return hf
    except ImportError:
        # Restore local module on failure
        if _import_hf_datasets._local_mod is not None:
            sys.modules['datasets'] = _import_hf_datasets._local_mod
        raise ImportError(
            "Sentiment140 requires the HuggingFace 'datasets' package:\n"
            "    pip install datasets\n")
    finally:
        sys.path = orig_path


def _restore_local_datasets():
    """Restore this file as sys.modules['datasets'] after HF loading is complete."""
    import sys
    local_mod = getattr(_import_hf_datasets, '_local_mod', None)
    if local_mod is not None:
        sys.modules['datasets'] = local_mod
        _import_hf_datasets._local_mod = None


def load_fashion_mnist(data_dir=None, download=True):
    """Load Fashion-MNIST dataset"""
    data_dir = data_dir or _DEFAULT_DATA_DIR
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    trainset = torchvision.datasets.FashionMNIST(
        root=data_dir, train=True, download=download, transform=transform
    )
    testset = torchvision.datasets.FashionMNIST(
        root=data_dir, train=False, download=download, transform=transform
    )
    
    return trainset, testset

def load_cifar10(data_dir=None, download=True):
    """Load CIFAR-10 dataset"""
    data_dir = data_dir or _DEFAULT_DATA_DIR
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=download, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=download, transform=transform
    )

    return trainset, testset

def load_femnist(data_dir=None, download=True):
    """Load FEMNIST dataset (EMNIST byclass split — 62 classes: 0-9, a-z, A-Z)"""
    data_dir = data_dir or _DEFAULT_DATA_DIR
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1751,), (0.3332,))
    ])

    trainset = torchvision.datasets.EMNIST(
        root=data_dir, split='byclass', train=True, download=download, transform=transform
    )
    testset = torchvision.datasets.EMNIST(
        root=data_dir, split='byclass', train=False, download=download, transform=transform
    )

    return trainset, testset

def load_emnist(data_dir=None, download=True, split='balanced'):
    """
    Load EMNIST dataset

    Args:
        data_dir: Directory to store/load data
        download: Whether to download if not present
        split: EMNIST split to use ('balanced', 'byclass', 'bymerge', 'digits', 'letters', 'mnist')
               Default 'balanced' has 47 classes with balanced distribution
    """
    data_dir = data_dir or _DEFAULT_DATA_DIR
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1751,), (0.3332,))
    ])

    trainset = torchvision.datasets.EMNIST(
        root=data_dir, split=split, train=True, download=download, transform=transform
    )
    testset = torchvision.datasets.EMNIST(
        root=data_dir, split=split, train=False, download=download, transform=transform
    )

    return trainset, testset

def load_shakespeare(data_dir=None, download=True, seq_len=SHAKESPEARE_SEQ_LEN):
    """
    Load Shakespeare next-character prediction dataset (80 character classes).

    Downloads tinyshakespeare.txt on first use, then caches locally.
    Each sample: input = 80-char sequence, target = next character.
    Train/test split: 90 / 10.
    """
    data_dir = data_dir or _DEFAULT_DATA_DIR
    txt_path = os.path.join(data_dir, 'shakespeare', 'tinyshakespeare.txt')

    if not os.path.exists(txt_path):
        if not download:
            raise FileNotFoundError(
                f"Shakespeare text not found at {txt_path}. Pass download=True to fetch it.")
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        print(f"[INFO] Downloading Shakespeare dataset...")
        try:
            urllib.request.urlretrieve(url, txt_path)
        except Exception as e:
            raise RuntimeError(f"Download failed: {e}. Place tinyshakespeare.txt in {os.path.dirname(txt_path)} manually.") from e
        print(f"[INFO] Saved to {txt_path}")

    with open(txt_path, 'r') as f:
        text = f.read()

    char_to_idx = {c: i for i, c in enumerate(SHAKESPEARE_CHARS)}

    # Keep only characters in the 80-class vocabulary
    indices = np.array([char_to_idx[c] for c in text if c in char_to_idx], dtype=np.int64)

    # Build (sequence, next_char) pairs
    n_samples = len(indices) - seq_len
    sequences = np.stack([indices[i:i + seq_len] for i in range(n_samples)])
    targets   = indices[seq_len:seq_len + n_samples]

    # 90 / 10 train-test split
    split = int(0.9 * n_samples)
    trainset = _ShakespeareDataset(sequences[:split], targets[:split])
    testset  = _ShakespeareDataset(sequences[split:], targets[split:])

    print(f"[INFO] Shakespeare — train: {len(trainset)}, test: {len(testset)}, classes: {len(SHAKESPEARE_CHARS)}")
    return trainset, testset


def load_sentiment140(data_dir=None, download=True,
                      max_words=SENTIMENT140_MAX_WORDS,
                      vocab_size=SENTIMENT140_VOCAB_SIZE):
    """
    Load Sentiment140 binary tweet-sentiment classification dataset.

    Downloads via HuggingFace 'datasets' on first use, then caches processed
    numpy arrays locally so subsequent loads are instant.

    Labels  : 0 = negative, 1 = positive  (neutral tweets are dropped)
    Input   : word-index sequence of length max_words, padded with 0 (UNK/PAD)
    Vocab   : top *vocab_size* words from training text; index 0 = UNK/PAD
    Split   : 90 % train / 10 % test

    Requires:  pip install datasets     (HuggingFace)
    """
    import json
    from collections import Counter

    data_dir = data_dir or _DEFAULT_DATA_DIR
    cache_dir  = os.path.join(data_dir, 'sentiment140')
    train_cache = os.path.join(cache_dir, 'train.npz')
    test_cache  = os.path.join(cache_dir, 'test.npz')

    # ---------- fast path: load from cache ----------
    if os.path.exists(train_cache) and os.path.exists(test_cache):
        train_data = np.load(train_cache)
        test_data  = np.load(test_cache)
        trainset = _Sentiment140Dataset(train_data['X'], train_data['y'])
        testset  = _Sentiment140Dataset(test_data['X'],  test_data['y'])
        print(f"[INFO] Sentiment140 loaded from cache — train: {len(trainset)}, test: {len(testset)}")
        return trainset, testset

    if not download:
        raise FileNotFoundError(
            f"Sentiment140 cache not found in {cache_dir}. Pass download=True to fetch it.")

    # ---------- download & process ----------
    hf = _import_hf_datasets()

    # Suppress HuggingFace progress bars for all operations (download + filter)
    hf.disable_progress_bar()

    print("[INFO] Downloading Sentiment140 (1.6M tweets) — this may take several minutes...")
    try:
        ds = hf.load_dataset("sentiment140", split="train")
    finally:
        # Restore local datasets.py as the 'datasets' module now that HF loading is done
        _restore_local_datasets()

    # Detect label column name: HF uses 'sentiment'; fallback to 'target'
    label_col = 'sentiment' if 'sentiment' in ds.column_names else 'target'

    # Detect label values — original CSV uses 0/2/4, some HF versions remap to 0/1/2
    unique_vals = sorted(set(ds[label_col]))
    neg_val, pos_val = min(unique_vals), max(unique_vals)

    # Keep only negative and positive; drop neutral; remap → 0 / 1
    print("[INFO] Filtering to binary labels (negative / positive)...")
    ds = ds.filter(lambda x: x[label_col] in [neg_val, pos_val])

    texts  = ds['text']
    labels = np.array([0 if t == neg_val else 1 for t in ds[label_col]], dtype=np.int64)

    # Re-enable progress bars now that all HF operations are complete
    try:
        hf.enable_progress_bar()
    except Exception:
        pass
    print(f"[INFO] {len(texts)} binary samples (neg: {(labels == 0).sum()}, pos: {(labels == 1).sum()})")

    # ---------- build vocabulary ----------
    print("[INFO] Building vocabulary (top {} words)...".format(vocab_size))
    counter = Counter()
    for text in texts:
        counter.update(text.lower().split())
    top_words    = [w for w, _ in counter.most_common(vocab_size)]
    word_to_idx  = {w: i + 1 for i, w in enumerate(top_words)}  # 0 = UNK / PAD

    # ---------- tokenise ----------
    print("[INFO] Tokenising tweets...")
    X = np.zeros((len(texts), max_words), dtype=np.int32)
    for i, text in enumerate(texts):
        words = text.lower().split()[:max_words]
        for j, w in enumerate(words):
            X[i, j] = word_to_idx.get(w, 0)
    y = labels

    # ---------- 90 / 10 split ----------
    split_idx   = int(0.9 * len(X))
    X_train, X_test = X[:split_idx],  X[split_idx:]
    y_train, y_test = y[:split_idx],  y[split_idx:]

    # ---------- cache to disk ----------
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(train_cache, X=X_train, y=y_train)
    np.savez(test_cache,  X=X_test,  y=y_test)
    with open(os.path.join(cache_dir, 'vocab.json'), 'w') as f:
        json.dump(top_words, f)

    print(f"[INFO] Sentiment140 cached — train: {len(X_train)}, test: {len(X_test)}, vocab: {vocab_size + 1}")
    return _Sentiment140Dataset(X_train, y_train), _Sentiment140Dataset(X_test, y_test)


def partition_dataset(dataset, num_clients, iid=True, alpha=0.5):
    """
    Partition dataset among clients with IID or Non-IID distribution
    
    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
        iid: Whether to use IID partitioning
        alpha: Dirichlet parameter for Non-IID (lower = more heterogeneous)
    """
    # Convert dataset to numpy arrays
    data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    X, y = next(iter(data_loader))
    X, y = X.numpy(), y.numpy()
    
    n_samples = len(dataset)
    client_data = []
    
    if iid:
        # IID partitioning: random shuffle and split
        indices = np.random.permutation(n_samples)
        splits = np.array_split(indices, num_clients)
        
        for split in splits:
            client_data.append((X[split], y[split]))
    
    else:
        # Non-IID partitioning using Dirichlet distribution (per-class split).
        #
        # Standard approach (FedML / Flower): for each class independently,
        # draw a Dirichlet vector of length num_clients to decide what
        # fraction of that class each client receives.  This guarantees:
        #   1. ALL samples are allocated (no data loss)
        #   2. No early-client / late-client bias
        #   3. Dirichlet heterogeneity is preserved exactly
        unique_classes = np.unique(y)
        num_classes = len(unique_classes)
        class_indices = {c: np.where(y == c)[0] for c in unique_classes}

        # Minimum samples per client (safeguard for very small datasets)
        min_samples = max(1, n_samples // (num_clients * 10))

        # Collect indices per client
        client_indices = [[] for _ in range(num_clients)]

        for c in unique_classes:
            idx = class_indices[c]
            np.random.shuffle(idx)

            # Dirichlet proportions for this class across all clients
            proportions = np.random.dirichlet([alpha] * num_clients)

            # Convert to integer counts
            counts = (proportions * len(idx)).astype(int)

            # Distribute the remainder (from int truncation) to random clients
            remainder = len(idx) - counts.sum()
            if remainder > 0:
                # Give extra samples to clients with largest fractional parts
                fracs = (proportions * len(idx)) - counts
                top_ids = np.argsort(-fracs)[:remainder]
                counts[top_ids] += 1

            # Assign samples slice-by-slice (no overlap, no loss)
            start = 0
            for cid in range(num_clients):
                end = start + counts[cid]
                if end > start:
                    client_indices[cid].extend(idx[start:end].tolist())
                start = end

        # Build client data; ensure every client gets at least min_samples
        all_indices_flat = np.arange(n_samples)
        np.random.shuffle(all_indices_flat)
        spare_ptr = 0  # pointer into shuffled spare pool

        for cid in range(num_clients):
            idx = np.array(client_indices[cid], dtype=int)
            if len(idx) < min_samples:
                # Top up from spare pool (samples not yet given to anyone
                # are unlikely here since Dirichlet covers all, but handle
                # edge cases like num_clients >> n_samples)
                need = min_samples - len(idx)
                extra = all_indices_flat[spare_ptr:spare_ptr + need]
                spare_ptr += need
                if len(extra) > 0:
                    idx = np.concatenate([idx, extra])
            client_data.append((X[idx], y[idx]))
    
    return client_data
