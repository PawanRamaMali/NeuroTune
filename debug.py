import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class SimpleNet(nn.Module):
    def __init__(self, input_size=1, hidden_size=10, output_size=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

def generate_data(n_samples=1000):
    # Generate synthetic data: y = 2x + sin(x) + noise
    X = torch.linspace(-5, 5, n_samples).reshape(-1, 1)
    y = 2 * X + torch.sin(X) + torch.randn_like(X) * 0.1
    return X, y

def train_model(model, optimizer, train_loader, val_loader, epochs=100, print_freq=10):
    criterion = nn.MSELoss()
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                output = model(X_batch)
                val_loss += criterion(output, y_batch).item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        if epoch % print_freq == 0:
            print(f'Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
    
    return train_losses, val_losses

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Generate data
    X, y = generate_data()
    
    # Split data into train and validation sets
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Create data loaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)
    
    # Test optimizers
    optimizers = {
        'OptiBrain': lambda p: OptiBrain(p, lr=0.001, betas=(0.9, 0.999)),
        'Adam': lambda p: torch.optim.Adam(p, lr=0.001, betas=(0.9, 0.999))
    }
    
    results = {}
    
    plt.figure(figsize=(15, 5))
    
    for name, opt_fn in optimizers.items():
        print(f"\nTraining with {name}")
        model = SimpleNet()
        optimizer = opt_fn(model.parameters())
        
        train_losses, val_losses = train_model(model, optimizer, train_loader, val_loader)
        results[name] = {'train': train_losses, 'val': val_losses}
        
        # Plot training curves
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label=f'{name} (train)')
        plt.subplot(1, 2, 2)
        plt.plot(val_losses, label=f'{name} (val)')
    
    plt.subplot(1, 2, 1)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('optimizer_comparison.png')
    plt.close()

if __name__ == "__main__":
    main()