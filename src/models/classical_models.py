# src/models/classical_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    """A simple Multi-Layer Perceptron for baseline comparison."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # Logits for CrossEntropyLoss
        return x
    
class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network for image tasks like CIFAR-10.
    """
    def __init__(self, input_channels: int = 3, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        # Input: (batch_size, 3, 32, 32)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        # -> (batch_size, 32, 32, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # -> (batch_size, 32, 16, 16)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # -> (batch_size, 64, 16, 16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # -> (batch_size, 64, 8, 8)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # -> (batch_size, 128, 8, 8)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # -> (batch_size, 128, 4, 4)
        
        # Flatten the output for the fully connected layer
        self.flatten = nn.Flatten()
        
        # Calculate the size of the flattened features
        # 128 channels * 4x4 image size
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # This model expects image input, e.g., (batch_size, 3, 32, 32)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # Logits
        return x

# In src/models/classical_models.py

class DecoupledMLP(nn.Module):
    """
    An MLP with a decoupled head and body, for federated representation learning.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 10):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.head = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        features = self.body(x)
        return self.head(features)