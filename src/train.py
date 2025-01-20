from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import torch

DEFAULT_NUM_EPOCHS = 20

def train_model(
        model,
        train_loader,
        device,
        num_epochs = DEFAULT_NUM_EPOCHS,
        criterion=None,
        optimizer=None,
        scheduler=None
):
    """
    Trains the model using the specified training dataset and hyperparameters.

    Args:
        model (nn.Module): The neural network model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        device (torch.device): The device (CPU or GPU) to run the training on.
        num_epochs (int, optional): Number of epochs for training. Default is 20.
        criterion (torch.nn.Module, optional): Loss function. If None, the default is `torch.nn.CrossEntropyLoss`.
        optimizer (torch.optim.Optimizer, optional): Optimizer for training. If None, the default is
            `torch.optim.Adam` with a learning rate of 0.001.
        scheduler (torch.optim.lr_scheduler._LRScheduler or torch.optim.lr_scheduler.ReduceLROnPlateau, optional):
            Learning rate scheduler. Can be any PyTorch learning rate scheduler.
            If None, the default is `torch.optim.lr_scheduler.StepLR` with `step_size=10` and `gamma=0.1`.
            Note that the scheduler must be initialized using the same optimizer passed to this function.

    Notes:
        - All parameters except `model`, `train_loader`, and `device` are optional and have default values:
            - **num_epochs:** `20`
            - **Criterion:** `torch.nn.CrossEntropyLoss`
            - **Optimizer:** `torch.optim.Adam` with `lr=0.001`
            - **Scheduler:** `torch.optim.lr_scheduler.StepLR` with `step_size=10` and `gamma=0.1`
        - The `scheduler` can be any PyTorch learning rate scheduler, such as:
            - `torch.optim.lr_scheduler.StepLR`
            - `torch.optim.lr_scheduler.MultiStepLR`
            - `torch.optim.lr_scheduler.ReduceLROnPlateau`
            - `torch.optim.lr_scheduler.CosineAnnealingLR`
            - `torch.optim.lr_scheduler.ExponentialLR`
        - The `scheduler`, if provided, must be initialized with the same `optimizer` passed to this function.

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        >>> train_model(model, train_loader, device, num_epochs=20, criterion=None, optimizer=optimizer, scheduler=scheduler)
    """

    # Define the loss function (default: Cross-Entropy Loss)
    criterion = criterion if criterion else nn.CrossEntropyLoss()

    # Define the optimizer (default: Adam)
    optimizer = optimizer if optimizer else optim.Adam(model.parameters(), lr=0.001)

    scheduler = scheduler if scheduler else torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

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

        # Calculate average loss for the epoch
        avg_loss = running_loss / len(train_loader)

        # Store the loss, accuracy and learning rate for the epoch
        epoch_stats.append((avg_loss, 100. * correct / total, optimizer.param_groups[0]['lr']))

        # Update the scheduler, if provided
        try:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_loss)
            else:
                scheduler.step()
        except TypeError as e:
            raise ValueError(f"Error updating scheduler: {e}. Ensure the correct arguments are passed.")

    return epoch_stats