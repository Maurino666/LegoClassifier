from src.model import SimpleCNN
from src.model import DeeperCNN
from train_and_evaluate import train_and_evaluate
import torch
import kornia.augmentation as k

def main():

    # Parameters
    dataset_path = "Dataset/D1"  # Path to the dataset
    batch_size = 256  # Batch size for data loading
    num_workers = 8  # Number of workers for data loading
    num_epochs = 20  # Number of epochs for training

    custom_transform = torch.nn.Sequential(
        k.RandomHorizontalFlip(p=0.5),
        k.RandomRotation(degrees=15),
        k.Normalize(mean=torch.tensor([0.5, 0.5, 0.5]), std=torch.tensor([0.5, 0.5, 0.5]))
    )

    # Train and evaluate the model with no custom transformations
    train_and_evaluate(dataset_path, batch_size, num_workers, num_epochs, DeeperCNN, custom_transform=custom_transform)


if __name__ == "__main__":
    main()

