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