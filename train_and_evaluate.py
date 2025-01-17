import os
import torch
from datetime import datetime
from time import time
from src.data_preprocessing import get_split_data_loaders
from src.train import train_model
from src.utils import save_model
from src.evaluation import evaluate_model

def train_and_evaluate(
        dataset_path,
        batch_size,
        num_workers,
        num_epochs,
        model_class,
        custom_transform=None,
        criterion=None,
        optimizer=None
):

    # Use GPU with CUDA if available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Create a directory for saving the trained models and evaluations
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("trained_models", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving outputs to: {output_dir}")

    # Load the dataset and split into training and test sets
    train_loader, test_loader, classes = get_split_data_loaders(
        dataset_path,
        batch_size,
        num_workers,
        custom_transform = custom_transform
    )
    print("Classes loaded...")

    # Initialize the model with the number of output classes
    model = model_class(num_classes=len(classes)).to(device)
    print("Model initialized...")

    # Train the model on the training set
    start_time = time()
    epoch_stats = train_model(model, train_loader, device, num_epochs, criterion, optimizer)
    training_time = time() - start_time
    print("Training completed!")

    # Save the trained model to the output directory
    model_path = os.path.join(output_dir, f"{model_class.__name__}.pth")
    save_model(model, path=model_path)
    print(f"Model saved to {model_path}")

    # Evaluate the model on the test set
    accuracy, report, labels, predictions = evaluate_model(model, test_loader, device, classes)
    print(f"Evaluation Complete!/nTest Accuracy: {accuracy:.2f}")


    # Save the evaluation results to the output directory
    evaluation_path = os.path.join(output_dir, "evaluation_results.txt")
    with open(evaluation_path, "w") as f:
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Model: {model_class.__name__}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Number of Workers: {num_workers}\n")
        f.write(f"Number of Epochs: {num_epochs}\n")
        f.write(f"Training Time: {training_time:.2f} seconds\n")
        f.write(f"Test Accuracy: {accuracy:.2f}\n")
        f.write(report)
        f.write("\nTraining Epoch Statistics:\n")
        for epoch, (loss, acc) in enumerate(epoch_stats, start=1):
            f.write(f"Epoch {epoch}: Loss={loss:.4f}, Accuracy={acc:.2f}%\n")

    print(f"Results saved to {evaluation_path}")