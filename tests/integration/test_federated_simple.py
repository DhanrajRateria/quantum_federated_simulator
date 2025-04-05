    # A simplified version of federated learning with MNIST
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import os

from src.federated.server import FederatedServer
from src.federated.client import FederatedClient
from src.federated.aggregation import FedAvg
from src.federated.utils import FederatedDataset, setup_logger, set_seed

# Set up logging
logger = setup_logger(name="mnist_demo", level="INFO")

# Define a simple CNN model for MNIST
class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def main():
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create data directory if it doesn't exist
    os.makedirs('./data', exist_ok=True)
    
    # Prepare MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        './data', train=False, download=True, transform=transform
    )
    
    # Partition dataset for clients (using IID partitioning)
    num_clients = 5
    client_datasets = FederatedDataset.iid_partition(train_dataset, num_clients)
    
    # Initialize global model
    global_model = MnistCNN()
    
    # Initialize server
    server = FederatedServer(
        model=global_model,
        aggregation_strategy=FedAvg(),
        evaluation_dataset=test_dataset
    )
    
    # Initialize and register clients
    for i in range(num_clients):
        # Create client
        client = FederatedClient(
            client_id=f"client_{i}",
            model=MnistCNN(),  # Each client needs its own model instance
            dataset=client_datasets[i],
            batch_size=64,
            learning_rate=0.01,
            optimizer_class=torch.optim.SGD,
            optimizer_kwargs={'momentum': 0.9}
        )
        
        # Register client with server
        server.register_client(client)
        logger.info(f"Client {i} has {len(client_datasets[i])} samples")
    
    # Run federated training
    num_rounds = 5
    logger.info(f"Starting federated training with {num_clients} clients for {num_rounds} rounds")
    server.train(
        num_rounds=num_rounds,
        local_epochs=1,
        client_fraction=1.0
    )
    
    # Evaluate final model
    final_metrics = server.evaluate()
    logger.info(f"Final evaluation metrics: {final_metrics}")
    
    # Save model
    os.makedirs('./models', exist_ok=True)
    server.save_model("./models/mnist_federated_simple.pt")
    logger.info("Model saved to ./models/mnist_federated_simple.pt")

if __name__ == "__main__":
    main()