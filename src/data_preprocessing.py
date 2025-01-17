from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import kornia.augmentation as k

DEFAULT_BATCH_SIZE = 256
DEFAULT_NUM_WORKERS = 8
DEFAULT_SPLIT_RATIO = 0.8

class GPUTransformDataset(Dataset):
    """
    Dataset wrapper to apply Kornia transformations on the GPU.

    Args:
        dataset (Dataset): The original dataset (e.g., ImageFolder).
        transform (torch.nn.Module): Kornia transformations for the GPU.
    """
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = self.to_tensor(img)  # Convert PIL.Image to torch.Tensor
        img = img.unsqueeze(0)  # Kornia requires a batch dimension
        img = self.transform(img).squeeze(0)  # Apply transformations and remove batch dimension
        return img, label


def get_gpu_transform():
    """
    Returns a Kornia transformation pipeline for GPU usage.

    Returns:
        torch.nn.Module: A sequential module containing Kornia transformations.
    """
    return torch.nn.Sequential(
        k.Normalize(mean=torch.tensor([0.5, 0.5, 0.5]),
                    std=torch.tensor([0.5, 0.5, 0.5]))
    )


def get_data_loader(
        dataset_path,
        batch_size=DEFAULT_BATCH_SIZE,
        num_workers=DEFAULT_NUM_WORKERS,
        custom_transform=None
):
    """
    Prepares a DataLoader for a dataset.

    Args:
        dataset_path (str): Path to the dataset directory.
        batch_size (int): Batch size.
        num_workers (int): Number of workers for data loading.
        custom_transform (torch.nn.Module): Custom transformation pipeline.

    Returns:
        tuple: A tuple containing:

        - **train_loader (DataLoader)**:
          DataLoader for the training set.

        - **classes (list)**:
          List of classes in the dataset.

    """

    # Loading the dataset
    raw_dataset = datasets.ImageFolder(root=dataset_path)

    # Use custom transformations if provided, otherwise default to get_gpu_transform
    transform = custom_transform if custom_transform else get_gpu_transform()

    # Wrap dataset with GPU transformations
    dataset = GPUTransformDataset(raw_dataset, transform)

    # Creating the DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

    return loader, raw_dataset.classes

def get_split_data_loaders(
        dataset_path,
        batch_size=DEFAULT_BATCH_SIZE,
        num_workers=DEFAULT_NUM_WORKERS,
        split_ratio=DEFAULT_SPLIT_RATIO,
        custom_transform=None
):
    """
    Prepares DataLoaders for the training and test sets with automatic splitting.

    Args:
        dataset_path (str): Path to the dataset directory.
        batch_size (int): Batch size.
        split_ratio (float): Percentage of data used for the training set (default 0.8).
        num_workers (int): Number of workers for data loading.
        custom_transform (torch.nn.Module): Custom transformation pipeline.

    Returns:
        tuple: A tuple containing:

        - **train_loader (DataLoader)**:
          DataLoader for the training set.

        - **test_loader (DataLoader)**:
          DataLoader for the test set.

        - **classes (list)**:
          List of classes in the dataset.
    """

    # Loading the dataset
    raw_dataset = datasets.ImageFolder(root=dataset_path)

    # Splitting the dataset
    train_size = int(len(raw_dataset) * split_ratio)
    test_size = len(raw_dataset) - train_size
    train_dataset, test_dataset = random_split(raw_dataset, [train_size, test_size])

    # Use custom transformations if provided, otherwise default to get_gpu_transform
    transform = custom_transform if custom_transform else get_gpu_transform()

    # Wrap datasets with GPU transformations
    train_dataset = GPUTransformDataset(train_dataset, transform)
    test_dataset = GPUTransformDataset(test_dataset, transform)

    # Creating the DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                              num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                             num_workers=num_workers)

    return train_loader, test_loader, raw_dataset.classes
