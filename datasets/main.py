from torch.utils.data import DataLoader, Subset
import torch
from torchvision import datasets, transforms

def get_mnist_loaders(data_dir, batch_size=64, val_split=0.2):

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(degrees=15),  # Rotación aleatoria de la imagen
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Traslación aleatoria de la imagen
        # transforms.RandomHorizontalFlip(),  # Volteo horizontal aleatorio de la imagen
        transforms.Normalize((0.1307,), (0.3081,))  # Normalización de la imagen
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


    train_dataset = datasets.MNIST(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=train_transform,
        )

    val_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=False,
        transform=eval_transform,
        )
    
    test_dataset = datasets.MNIST(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=eval_transform,
        )

    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(train_dataset)),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
        )

    train_dataset = Subset(train_dataset, train_indices.indices)
    val_dataset = Subset(val_dataset, val_indices.indices)

    print(f"Train size: {train_size}, Validation size: {val_size}, Test size: {len(test_dataset)}")


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
    

    return train_loader, val_loader, test_loader