from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

DEFAULT_BATCH_SIZE = 128
DEFAULT_NUM_WORKERS = 10
DEFAULT_SPLIT_RATIO = 0.8

def get_data_loader(dataset_path, batch_size=DEFAULT_BATCH_SIZE, num_workers=DEFAULT_NUM_WORKERS):
    """
    Prepares a DataLoader for a dataset.

    Args:
        dataset_path (str): Path to the dataset directory.
        batch_size (int): Batch size.
        num_workers (int): Number of workers for data loading.

    Returns:
        tuple: A tuple containing:

        - **train_loader (DataLoader)**:
          DataLoader for the training set.

        - **classes (list)**:
          List of classes in the dataset.

    """
    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
    ])

    # Loading the dataset
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    # Creating the DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

    return loader, dataset.classes

def get_split_data_loaders(dataset_path, batch_size=DEFAULT_BATCH_SIZE, num_workers=DEFAULT_NUM_WORKERS, split_ratio=DEFAULT_SPLIT_RATIO):
    """
    Prepares DataLoaders for the training and test sets with automatic splitting.

    Args:
        dataset_path (str): Path to the dataset directory.
        batch_size (int): Batch size.
        split_ratio (float): Percentage of data used for the training set (default 0.8).
        num_workers (int): Number of workers for data loading.

    Returns:
        tuple: A tuple containing:

        - **train_loader (DataLoader)**:
          DataLoader for the training set.

        - **test_loader (DataLoader)**:
          DataLoader for the test set.

        - **classes (list)**:
          List of classes in the dataset.
    """
    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
    ])

    # Loading the dataset
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    # Splitting the dataset
    train_size = int(len(dataset) * split_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Creating the DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                              num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                             num_workers=num_workers)

    return train_loader, test_loader, dataset.classes
