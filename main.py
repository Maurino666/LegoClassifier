from src.model import SimpleCNN
from train_and_evaluate import train_and_evaluate


def main():

    # Parameters
    dataset_path = "Dataset/D1"  # Path to the dataset
    batch_size = 256  # Batch size for data loading
    num_workers = 12  # Number of workers for data loading
    num_epochs = 1  # Number of epochs for training

    train_and_evaluate(dataset_path, batch_size, num_workers, num_epochs, SimpleCNN)

if __name__ == "__main__":
    main()

