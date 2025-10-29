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
        
        # Generate Dirichlet distribution for each client
        client_class_dist = np.random.dirichlet([alpha] * num_classes, num_clients)
        
        for client_id in range(num_clients):
            client_indices = []
            
            for class_id in range(num_classes):
                # Number of samples from this class for this client
                n_class_samples = int(client_class_dist[client_id][class_id] * len(class_indices[class_id]))
                
                if n_class_samples > 0:
                    # Randomly select samples from this class
                    selected = np.random.choice(
                        class_indices[class_id], 
                        min(n_class_samples, len(class_indices[class_id])), 
                        replace=False
                    )
                    client_indices.extend(selected)
                    
                    # Remove selected indices to avoid overlap
                    class_indices[class_id] = np.setdiff1d(class_indices[class_id], selected)
            
            if len(client_indices) > 0:
                client_data.append((X[client_indices], y[client_indices]))
            else:
                # Fallback: give at least some data to each client
                fallback_size = max(1, n_samples // (num_clients * 10))
                remaining_indices = np.concatenate(class_indices)
                if len(remaining_indices) >= fallback_size:
                    selected = np.random.choice(remaining_indices, fallback_size, replace=False)
                    client_data.append((X[selected], y[selected]))
                else:
                    # Very last resort
                    client_data.append((X[:fallback_size], y[:fallback_size]))
    
    print(f"Dataset partitioned among {num_clients} clients ({'IID' if iid else 'Non-IID'})")
    print(f"Client data sizes: {[len(data[1]) for data in client_data]}")
    
    return client_data
