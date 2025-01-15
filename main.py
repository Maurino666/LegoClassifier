import os
import torch
from datetime import datetime
from src.data_preprocessing import get_split_data_loaders
from src.model import SimpleCNN
from src.train import train_model
from src.utils import save_model
from src.evaluation import evaluate_model, save_results

def main():

    # Parameters
    dataset_path = "Dataset/D1"  # Path to the dataset
    batch_size = 512  # Batch size for data loading
    num_epochs = 20  # Number of epochs for training

    # Use GPU with CUDA if available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Create a directory for saving the trained models and evaluations
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("trained_models", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving outputs to: {output_dir}")

    # Load the dataset and split into training and test sets
    train_loader, test_loader, classes = get_split_data_loaders(dataset_path, batch_size)
    print("Classes loaded...")

    # Initialize the model with the number of output classes
    model = SimpleCNN(num_classes=len(classes)).to(device)
    print("Model initialized...")

    # Train the model on the training set
    train_model(model, train_loader, device, num_epochs)
    print("Training completed!")

    # Save the trained model to the output directory
    model_path = os.path.join(output_dir, "simple_cnn.pth")
    save_model(model, path=model_path)

    # Evaluate the model on the test set
    accuracy, report, labels, predictions = evaluate_model(model, test_loader, device, classes)
    print(f"Test Accuracy: {accuracy:.2f}")
    print(report)

    # Save the evaluation results to the output directory
    evaluation_path = os.path.join(output_dir, "evaluation_results.txt")
    save_results(labels, predictions, classes, output_path=evaluation_path)

if __name__ == "__main__":
    main()

