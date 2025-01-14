import torch
from src.data_preprocessing import get_split_data_loaders
from src.model import SimpleCNN
from src.train import train_model
from src.utils import save_model
from src.evaluation import evaluate_model, save_results

def main():
    # Parameters
    dataset_path = "Dataset/D1"  # Path to the dataset
    batch_size = 256  # Batch size for data loading
    num_epochs = 5  # Number of epochs for training

    # Use GPU with CUDA if available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load the dataset and split into training and test sets
    train_loader, test_loader, classes = get_split_data_loaders(dataset_path, batch_size)
    print("Classes loaded...")

    # Initialize the model with the number of output classes
    model = SimpleCNN(num_classes=len(classes)).to(device)
    print("Model initialized...")

    # Train the model on the training set
    train_model(model, train_loader, device, num_epochs)
    print("Training completed!")

    # Save the trained model to a file
    save_model(model, path="simple_cnn.pth")

    # Evaluate the model on the test set
    accuracy, report, labels, predictions = evaluate_model(model, test_loader, device, classes)
    print(f"Test Accuracy: {accuracy:.2f}")
    print(report)

    # Save the evaluation results to a file
    save_results(labels, predictions, classes, output_path="evaluation_results.txt")

if __name__ == "__main__":
    main()

