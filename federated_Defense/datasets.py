import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

def load_fashion_mnist(data_dir='./data', download=True):
    """Load Fashion-MNIST dataset"""
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

def load_cifar10(data_dir='./data', download=True):
    """Load CIFAR-10 dataset"""
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

def load_mnist(data_dir='./data', download=True):
    """Load MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=download, transform=transform
    )
    testset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=download, transform=transform
    )

    return trainset, testset

def load_emnist(data_dir='./data', download=True, split='balanced'):
    """
    Load EMNIST dataset

    Args:
        data_dir: Directory to store/load data
        download: Whether to download if not present
        split: EMNIST split to use ('balanced', 'byclass', 'bymerge', 'digits', 'letters', 'mnist')
               Default 'balanced' has 47 classes with balanced distribution
    """
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
        # Non-IID partitioning using Dirichlet distribution
        num_classes = len(np.unique(y))
        class_indices = [np.where(y == i)[0] for i in range(num_classes)]

        # IMPORTANT: Ensure minimum samples per client
        min_samples_per_client = max(50, n_samples // (num_clients * 5))  # At least 50 samples or ~20% of average

        # Generate Dirichlet distribution for each client
        client_class_dist = np.random.dirichlet([alpha] * num_classes, num_clients)

        # Normalize to ensure minimum samples: boost low probabilities
        for client_id in range(num_clients):
            total_expected = sum(int(client_class_dist[client_id][c] * len(class_indices[c]))
                               for c in range(num_classes))

            # If expected samples are too low, boost this client's probabilities
            if total_expected < min_samples_per_client:
                boost_factor = min_samples_per_client / max(total_expected, 1)
                client_class_dist[client_id] *= boost_factor

        # Re-normalize each class across clients to sum to 1.0
        for class_id in range(num_classes):
            class_total = sum(client_class_dist[c][class_id] for c in range(num_clients))
            if class_total > 0:
                for client_id in range(num_clients):
                    client_class_dist[client_id][class_id] /= class_total

        for client_id in range(num_clients):
            client_indices = []

            for class_id in range(num_classes):
                # Number of samples from this class for this client
                n_class_samples = int(client_class_dist[client_id][class_id] * len(class_indices[class_id]))

                if n_class_samples > 0:
                    # Randomly select samples from this class
                    available = min(n_class_samples, len(class_indices[class_id]))
                    if available > 0:
                        selected = np.random.choice(
                            class_indices[class_id],
                            available,
                            replace=False
                        )
                        client_indices.extend(selected)

                        # Remove selected indices to avoid overlap
                        class_indices[class_id] = np.setdiff1d(class_indices[class_id], selected)

            if len(client_indices) >= min_samples_per_client:
                client_data.append((X[client_indices], y[client_indices]))
            else:
                # Fallback: give minimum samples to each client from remaining data
                remaining_indices = np.concatenate([ci for ci in class_indices if len(ci) > 0])
                if len(remaining_indices) >= min_samples_per_client:
                    selected = np.random.choice(remaining_indices, min_samples_per_client, replace=False)
                    # Add already allocated samples
                    if len(client_indices) > 0:
                        selected = np.concatenate([client_indices, selected])
                    client_data.append((X[selected], y[selected]))
                else:
                    # Very last resort - use all remaining
                    all_indices = client_indices if len(client_indices) > 0 else remaining_indices[:min_samples_per_client]
                    client_data.append((X[all_indices], y[all_indices]))
    
    print(f"Dataset partitioned among {num_clients} clients ({'IID' if iid else 'Non-IID'})")
    print(f"Client data sizes: {[len(data[1]) for data in client_data]}")
    
    return client_data
