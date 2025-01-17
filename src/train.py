from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn

DEFAULT_NUM_EPOCHS = 20

def train_model(
        model,
        train_loader,
        device,
        num_epochs = DEFAULT_NUM_EPOCHS,
        criterion=None,
        optimizer=None
):
    """
    Trains the model using the specified training dataset and hyperparameters.

    Args:
        model (nn.Module): The neural network model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        device (torch.device): The device (CPU or GPU) to run the training on.
        num_epochs (int): Number of epochs for training.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training
    """
    # Define the loss function (default: Cross-Entropy Loss)
    criterion = criterion if criterion else nn.CrossEntropyLoss()

    # Define the optimizer (default: Adam)
    optimizer = optimizer if optimizer else optim.Adam(model.parameters(), lr=0.0001)

    # List to store loss and accuracy for each epoch
    epoch_stats = []

    # Training loop for the specified number of epochs
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        # Progress bar for visualizing training progress
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", total=len(train_loader))

        for inputs, labels in progress_bar:
            # Move input data and labels to the specified device (e.g., GPU)
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass: compute model predictions
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, labels)

            # Backward pass: compute gradients and update weights
            loss.backward()
            optimizer.step()

            # Update metrics for the current batch
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update the progress bar with loss and accuracy
            progress_bar.set_postfix({
                "Loss": f"{running_loss / (total or 1):.4f}",
                "Acc": f"{100. * correct / (total or 1):.2f}%"
            })
        epoch_stats.append((running_loss / len(train_loader), 100. * correct / total))

    return epoch_stats