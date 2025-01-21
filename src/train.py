from tqdm import tqdm
import torch.nn as nn
import torch

def train_model(
        model,
        train_loader,
        device,
        num_epochs,
        criterion,
        optimizer,
        scheduler=None  # Scheduler è opzionale e di default è None
):
    """
    Trains the model using the specified training dataset and hyperparameters.

    Args:
        model (nn.Module): The neural network model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        device (torch.device): The device (CPU or GPU) to run the training on.
        num_epochs (int): Number of epochs for training.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
            If None, no scheduler is used (default behavior).
    """
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
            # Move input data and labels to the specified device
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
                "Acc": f"{100. * correct / (total or 1):.2f}%",
                "LR": f"{optimizer.param_groups[0]['lr']:.6f}"
            })

        # Calculate average loss for the epoch
        avg_loss = running_loss / len(train_loader)

        # Store the loss, accuracy and learning rate for the epoch
        epoch_stats.append((avg_loss, 100. * correct / total, optimizer.param_groups[0]['lr']))

        # Update the scheduler, if provided
        if scheduler:  # Check if a scheduler is provided
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_loss)
            else:
                scheduler.step()

    return epoch_stats