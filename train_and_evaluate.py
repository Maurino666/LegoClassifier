import os
import torch
from datetime import datetime
from time import time
import matplotlib.pyplot as plt
import seaborn as sns
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
        optimizer_factory=None,
        scheduler_factory=None
):
    """
        Trains and evaluates a model.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Batch size for data loading.
            num_workers (int): Number of workers for data loading.
            num_epochs (int): Number of epochs for training.
            model_class (type): Class of the model to be instantiated.
            custom_transform (torch.nn.Module, optional): Custom data augmentation transformations.
            criterion (torch.nn.Module, optional): Loss function. If None, CrossEntropyLoss is used.
            optimizer_factory (callable, optional): Factory function for creating the optimizer.
                Must take the model as input and return an optimizer.
            scheduler_factory (callable, optional): Factory function for creating the scheduler.
                Must take the optimizer as input and return a scheduler.
    """
    # Use GPU with CUDA if available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Create a directory for saving the trained models and evaluations
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("trained_models", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving outputs to: {output_dir}")

    # Validate factory inputs
    if optimizer_factory and not callable(optimizer_factory):
        raise ValueError("optimizer_factory must be a callable that takes a model and returns an optimizer.")
    if scheduler_factory and not callable(scheduler_factory):
        raise ValueError("scheduler_factory must be a callable that takes an optimizer and returns a scheduler.")

    # Load the dataset and split into training and test sets
    train_loader, test_loader, classes = get_split_data_loaders(
        dataset_path,
        batch_size,
        num_workers,
        custom_transform=custom_transform
    )
    print("Classes loaded...")

    # Initialize the model with the number of output classes
    model = model_class(num_classes=len(classes)).to(device)
    print("Model initialized...")

    # Define default criterion, optimizer, and scheduler
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()
    optimizer = optimizer_factory(model) if callable(optimizer_factory) else torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = scheduler_factory(optimizer) if callable(scheduler_factory) else None

    # Train the model on the training set
    start_time = time()
    try:
        epoch_stats = train_model(
            model,
            train_loader,
            device,
            num_epochs,
            criterion,
            optimizer,
            scheduler
        )
        training_time = time() - start_time
        print("Training completed!")
    except Exception as e:
        print(f"Error during training: {e}")
        raise

    # Save the trained model to the output directory
    model_path = os.path.join(output_dir, f"{model_class.__name__}.pth")
    save_model(model, path=model_path)
    print(f"Model saved to {model_path}")

    # Evaluate the model on the test set
    accuracy, report, labels, predictions, cm = evaluate_model(model, test_loader, device, classes)
    print(f"Evaluation Complete!\nTest Accuracy: {accuracy:.2f}")

    # Save the evaluation results to the output directory with all the information
    evaluation_path = os.path.join(output_dir, "evaluation_results.txt")
    with open(evaluation_path, "w") as f:
        # Write the training and evaluation details
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Model: {model_class.__name__}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Number of Workers: {num_workers}\n")
        f.write(f"Number of Epochs: {num_epochs}\n")
        f.write(f"Training Time: {training_time:.2f} seconds\n")

        # Write model details
        f.write("\nModel Details:\n")
        f.write(str(model) + "\n")
        f.write(f"Total Parameters: {sum(p.numel() for p in model.parameters())}\n")

        # Write custom transformations, loss function, optimizer and scheduler if provided
        f.write("\nCustom Transformations:\n")
        f.write(f"{custom_transform}\n" if custom_transform else "Default\n")
        f.write("\nLoss Function:\n")
        criterion_info = str(criterion.__class__.__name__)
        criterion_params = getattr(criterion, "__dict__", "Default parameters")
        f.write(f"{criterion_info} with parameters: {criterion_params}\n")
        f.write("\nOptimizer:\n")
        f.write(f"{optimizer}\n")
        f.write("\nScheduler:\n")
        if scheduler:
            scheduler_info = str(scheduler.__class__.__name__)  # Get scheduler type
            scheduler_params = scheduler.state_dict()  # Get scheduler parameters
            f.write(f"{scheduler_info} with parameters:\n{scheduler_params}\n")
        else:
            f.write("Default (None)\n")

        # Write the evaluation metrics
        f.write("\nEvaluation Results:\n")
        f.write(f"Test Accuracy: {accuracy:.2f}\n")
        f.write(report)

        # Write the epoch-wise training statistics
        f.write("\nTraining Epoch Statistics:\n")
        for epoch, (loss, acc) in enumerate(epoch_stats, start=1):
            f.write(f"Epoch {epoch}: Loss={loss:.4f}, Accuracy={acc:.2f}%\n")

    print(f"Results saved to {evaluation_path}")

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.ylabel("True Labels")
    plt.xlabel("Predicted Labels")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")
