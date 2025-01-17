import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        """
        Initializes the SimpleCNN model.

        Args:
            num_classes (int): Number of output classes for classification.
        """
        super(SimpleCNN, self).__init__()

        # First convolutional layer: input channels = 3 (RGB), output channels = 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)

        # Max pooling layer: reduces spatial dimensions by a factor of 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional layer: input channels = 32, output channels = 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Fully connected layer 1: maps flattened feature maps to 128 neurons
        self.fc1 = nn.Linear(64 * 16 * 16, 128)

        # Fully connected layer 2: maps 128 neurons to the number of output classes
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the SimpleCNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 64, 64).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        # Apply first convolutional layer followed by ReLU activation and pooling
        x = self.pool(torch.relu(self.conv1(x)))

        # Apply second convolutional layer followed by ReLU activation and pooling
        x = self.pool(torch.relu(self.conv2(x)))

        # Flatten the feature maps to a 1D vector for the fully connected layers
        x = x.view(-1, 64 * 16 * 16)

        # Apply the first fully connected layer with ReLU activation
        x = torch.relu(self.fc1(x))

        # Apply the second fully connected layer to produce class scores
        x = self.fc2(x)

        return x

class DeeperCNN(nn.Module):
    def __init__(self, num_classes):
        """
        Initializes the DeeperCNN model with additional convolutional layers.

        Args:
            num_classes (int): Number of output classes for classification.
        """
        super(DeeperCNN, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third convolutional block
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        Defines the forward pass of the DeeperCNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 64, 64).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        # First block
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool1(x)

        # Second block
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool2(x)

        # Third block
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))
        x = self.pool3(x)

        # Flatten the feature maps
        x = x.view(-1, 512 * 8 * 8)

        # Fully connected layers with dropout
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x
