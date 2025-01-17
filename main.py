from src.model import SimpleCNN
from src.model import DeeperCNN
from train_and_evaluate import train_and_evaluate


def main():

    # Parameters
    dataset_path = "Dataset/D1"  # Path to the dataset
    batch_size = 256  # Batch size for data loading
    num_workers = 8  # Number of workers for data loading
    num_epochs = 20  # Number of epochs for training

    #Train and evaluate deeper model with no custom transformations
    train_and_evaluate(dataset_path, batch_size, num_workers, num_epochs, DeeperCNN)

    # Parameters
    dataset_path = "Dataset/D1"  # Path to the dataset
    batch_size = 256  # Batch size for data loading
    num_workers = 12  # Number of workers for data loading
    num_epochs = 20  # Number of epochs for training

    # Train and evaluate the model with no custom transformations
    train_and_evaluate(dataset_path, batch_size, num_workers, num_epochs, SimpleCNN)


if __name__ == "__main__":
    main()

