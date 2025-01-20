from src.model import SimpleCNN, DeeperCNN
from train_and_evaluate import train_and_evaluate
import torch
import kornia.augmentation as k

def create_optimizer_factory(optimizer_class=torch.optim.Adam, **kwargs):
    """Factory for creating an optimizer."""
    def optimizer_factory(model):
        return optimizer_class(model.parameters(), **kwargs)
    return optimizer_factory

def create_scheduler_factory(scheduler_class=torch.optim.lr_scheduler.StepLR, **kwargs):
    """Factory for creating a scheduler."""
    def scheduler_factory(optimizer):
        return scheduler_class(optimizer, **kwargs)
    return scheduler_factory

def main():
    # Parameters
    dataset_path = "Dataset/D1"  # Path to the dataset
    batch_size = 256  # Batch size for data loading
    num_workers = 8  # Number of workers for data loading
    num_epochs = 20  # Number of epochs for training

    # Define custom transformations
    custom_transform = torch.nn.Sequential(
        k.RandomHorizontalFlip(p=0.5),
        k.RandomRotation(degrees=15),
        k.Normalize(mean=torch.tensor([0.5, 0.5, 0.5]), std=torch.tensor([0.5, 0.5, 0.5]))
    )

    # Define optimizer and scheduler factories
    optimizer_factory = create_optimizer_factory(torch.optim.Adam, lr=0.001)
    scheduler_factory = create_scheduler_factory(
        torch.optim.lr_scheduler.StepLR, step_size=10, gamma=0.1
    )

    # Train and evaluate the model
    train_and_evaluate(
        dataset_path=dataset_path,
        batch_size=batch_size,
        num_workers=num_workers,
        num_epochs=num_epochs,
        model_class=DeeperCNN,
        custom_transform=custom_transform,
        criterion=None,  # Use default criterion
        optimizer_factory=optimizer_factory,  # Pass the optimizer factory
        scheduler_factory=scheduler_factory  # Pass the scheduler factory
    )

if __name__ == "__main__":
    main()
