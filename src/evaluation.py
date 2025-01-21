import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from src.utils import load_model


def load_model_for_evaluation(model_class, model_path, device):
    """
    Loads a pre-trained model for evaluation using the utility function.

    Args:
        model_class (callable): A callable that returns an instance of the model class.
        model_path (str): The path to the saved model file.
        device (torch.device): The device (CPU or GPU) to load the model onto.

    Returns:
        torch.nn.Module: The loaded model ready for evaluation.
    """
    model = load_model(model_class, model_path, device)
    model.eval()  # Set the model to evaluation mode
    return model

def evaluate_model(model, test_loader, device, class_names):
    """
    Evaluates the model on the test dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): The device (CPU or GPU) to use for evaluation.
        class_names (list): List of class names in the dataset.

    Returns:
        tuple: A tuple containing:

        - accuracy (float): Overall accuracy of the model.
        - report (str): Classification report.
        - labels (list): True labels from the test dataset.
        - predictions (list): Predicted labels from the model.
    """
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions, target_names=class_names)
    cm = confusion_matrix(all_labels, all_predictions)
    return accuracy, report, all_labels, all_predictions, cm

def visualize_predictions(test_loader, class_names, model, device, num_images=4):
    """
    Visualizes a few images with their predictions.

    Args:
        test_loader (DataLoader): DataLoader for the test dataset.
        class_names (list): List of class names.
        model (torch.nn.Module): The model to use for predictions.
        device (torch.device): The device (CPU or GPU) to use for predictions.
        num_images (int): Number of images to display.
    """
    model.eval()
    with torch.no_grad():
        inputs, labels = next(iter(test_loader))
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)

        inputs = inputs.cpu()
        predictions = predictions.cpu()
        labels = labels.cpu()

        # Display images
        for i in range(min(num_images, len(inputs))):
            img = inputs[i].permute(1, 2, 0) * 0.5 + 0.5  # De-normalize
            plt.imshow(img.numpy())
            plt.title(f"Pred: {class_names[predictions[i]]}, True: {class_names[labels[i]]}")
            plt.show()

def save_results(labels, predictions, class_names, output_path="results.txt"):
    """
    Saves the evaluation results to a file.

    Args:
        labels (list): True labels.
        predictions (list): Predicted labels.
        class_names (list): List of class names.
        output_path (str): Path to the output file.
    """
    with open(output_path, "w") as f:
        for true_label, pred_label in zip(labels, predictions):
            f.write(f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}\n")
    print(f"Results saved to {output_path}")