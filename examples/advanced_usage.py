import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt  # Added missing import
from neurotune import OptiBrain, AdaptiveMomentum, ElasticLR, ConvergenceTracker
from neurotune.utils import setup_logging, initialize_parameters

class ComplexNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def compare_optimizers(model, train_loader, valid_loader, epochs=50):
    """Compare different optimizers on the same task."""
    optimizers = {
        'OptiBrain': OptiBrain(model.parameters(), lr=0.001),
        'AdaptiveMomentum': AdaptiveMomentum(model.parameters(), lr=0.001),
        'ElasticLR': ElasticLR(model.parameters(), lr=0.001)
    }
    
    results = {}
    for name, optimizer in optimizers.items():
        print(f"\nTraining with {name}")
        # Create a fresh model for each optimizer to ensure fair comparison
        model_copy = ComplexNet(model.network[0].in_features, 
                              [64, 32], 
                              model.network[-1].out_features)
        initialize_parameters(model_copy, method='xavier')
        
        tracker = ConvergenceTracker()
        train_losses = []
        valid_losses = []
        
        for epoch in range(epochs):
            # Training
            model_copy.train()
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = model_copy(X_batch)
                loss = nn.MSELoss()(output, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            model_copy.eval()
            valid_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in valid_loader:
                    output = model_copy(X_batch)
                    loss = nn.MSELoss()(output, y_batch)
                    valid_loss += loss.item()
            
            avg_valid_loss = valid_loss / len(valid_loader)
            valid_losses.append(avg_valid_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, "
                      f"Valid Loss = {avg_valid_loss:.4f}")
        
        results[name] = {
            'train_losses': train_losses,
            'valid_losses': valid_losses
        }
    
    return results

def main():
    # Setup logging
    setup_logging()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Generate synthetic dataset
    N_SAMPLES = 1000
    INPUT_DIM = 20
    X = torch.randn(N_SAMPLES, INPUT_DIM)
    y = torch.randn(N_SAMPLES, 1)
    
    # Split data into train and validation sets
    train_size = int(0.8 * N_SAMPLES)
    X_train, X_valid = X[:train_size], X[train_size:]
    y_train, y_valid = y[:train_size], y[train_size:]
    
    # Create data loaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    valid_loader = DataLoader(TensorDataset(X_valid, y_valid), batch_size=32)
    
    # Create model
    model = ComplexNet(INPUT_DIM, [64, 32], 1)
    initialize_parameters(model, method='xavier')
    
    # Compare optimizers
    results = compare_optimizers(model, train_loader, valid_loader)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    for name, result in results.items():
        plt.plot(result['train_losses'], label=name)
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for name, result in results.items():
        plt.plot(result['valid_losses'], label=name)
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('optimizer_comparison.png')
    plt.close()

if __name__ == "__main__":
    main()