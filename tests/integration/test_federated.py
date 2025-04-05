import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import logging
import argparse
import os

from src.federated.server import FederatedServer
from src.federated.client import FederatedClient
from src.federated.aggregation import FedAvg
from src.federated.utils import (
    FederatedDataset, 
    setup_logger, 
    set_seed, 
    ConfigUtils
)

# Set up logging
logger = setup_logger(name="mnist_test", level=logging.INFO, console_output=True)

# Define a simple CNN model for MNIST
class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def main(args):
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Load configurations
    server_config = ConfigUtils.load_yaml_config(args.server_config) or {}
    client_config = ConfigUtils.load_yaml_config(args.client_config) or {}

    if not server_config:
        logger.warning(f"Server config file '{args.server_config}' not found. Using defaults.")
    if not client_config:
        logger.warning(f"Client config file '{args.client_config}' not found. Using defaults.")

    # Make sure data directory exists
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

    if args.fast:
        # Use smaller dataset
        train_indices = torch.randperm(len(train_dataset))[:5000]
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        
        test_indices = torch.randperm(len(test_dataset))[:1000]
        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
        
        # Override number of rounds if not specified
        if args.rounds is None:
            args.rounds = 3
    # Partition dataset for clients
    num_clients = args.num_clients
    if args.iid:
        # IID partitioning (uniform and random)
        client_datasets = FederatedDataset.iid_partition(train_dataset, num_clients)
        logger.info(f"Created {num_clients} clients with IID data partitioning")
    else:
        # Non-IID partition based on labels (each client gets a subset of classes)
        client_datasets = FederatedDataset.non_iid_label_partition(
            train_dataset, num_clients, num_classes=10, classes_per_client=args.classes_per_client
        )
        logger.info(f"Created {num_clients} clients with non-IID data partitioning "
                   f"({args.classes_per_client} classes per client)")
    
    # Initialize model
    global_model = MnistCNN()
    
    # Initialize server
    server = FederatedServer(
        model=global_model,
        aggregation_strategy=FedAvg(),
        evaluation_dataset=test_dataset
    )
    
    # Initialize clients
    for i in range(num_clients):
        # Extract optimization settings from client config
        optim_config = client_config.get('optimization', {})
        lr = optim_config.get('learning_rate', 0.01)
        optimizer_name = optim_config.get('optimizer', {}).get('name', 'SGD')
        
        if optimizer_name.lower() == 'sgd':
            optimizer_class = torch.optim.SGD
            momentum = optim_config.get('optimizer', {}).get('momentum', 0.9)
            weight_decay = optim_config.get('optimizer', {}).get('weight_decay', 0.0001)
            optimizer_kwargs = {'momentum': momentum, 'weight_decay': weight_decay}
        else:
            optimizer_class = torch.optim.Adam
            optimizer_kwargs = {}
        
        # Create client with its own model instance
        client = FederatedClient(
            client_id=f"client_{i}",
            model=MnistCNN(),  # Each client needs its own model instance
            dataset=client_datasets[i],
            batch_size=client_config.get('client', {}).get('batch_size', 32),
            learning_rate=lr,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs
        )
        
        # Register client with server
        server.register_client(client)
        
        # Log client dataset size
        logger.debug(f"Client {i} has {len(client_datasets[i])} training samples")
    
    # Training settings from server config
    num_rounds = args.rounds or server_config.get('server', {}).get('num_rounds', 10)
    client_fraction = server_config.get('client_management', {}).get('client_fraction', 1.0)
    local_epochs = client_config.get('client', {}).get('local_epochs', 1)
    
    # FedProx mu parameter if needed
    proximal_mu = client_config.get('regularization', {}).get('proximal_mu', 0.0)
    
    # Run federated training
    logger.info(f"Starting federated training with {num_clients} clients for {num_rounds} rounds")
    logger.info(f"Client fraction: {client_fraction}, Local epochs: {local_epochs}")
    
    if proximal_mu > 0:
        logger.info(f"Using FedProx with mu={proximal_mu}")
        
    results = server.train(
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        client_fraction=client_fraction,
        proximal_term=proximal_mu
    )
    
    # Evaluate final model
    final_metrics = server.evaluate()
    logger.info(f"Final evaluation metrics: {final_metrics}")
    
    # Save model if requested
    if args.save_model:
        os.makedirs('./models', exist_ok=True)
        save_path = args.model_path or server_config.get('model', {}).get('save_path', './models/federated_mnist.pt')
        server.save_model(save_path)
        logger.info(f"Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Learning with MNIST')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--num-clients', type=int, default=10, help='number of clients')
    parser.add_argument('--rounds', type=int, default=None, help='number of training rounds')
    parser.add_argument('--iid', action='store_true', help='use IID data partitioning')
    parser.add_argument('--classes-per-client', type=int, default=2, 
                        help='number of classes per client for non-IID partitioning')
    parser.add_argument('--save-model', action='store_true', help='save final model')
    parser.add_argument('--model-path', type=str, default=None, help='path to save model')
    parser.add_argument('--server-config', type=str, default='server_config.yaml',
                        help='path to server configuration')
    parser.add_argument('--client-config', type=str, default='client_config.yaml',
                        help='path to client configuration')
    parser.add_argument('--fast', action='store_true', 
                        help='Use reduced dataset and fewer rounds for quick testing')
    args = parser.parse_args()
    
    main(args)