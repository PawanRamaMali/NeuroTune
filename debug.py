import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Simple model for testing
class SimpleNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Generate synthetic dataset
    N_SAMPLES = 1000
    INPUT_DIM = 20
    X = torch.randn(N_SAMPLES, INPUT_DIM)
    # Create a non-linear target function
    y = torch.sin(X[:, 0]) + torch.cos(X[:, 1]) + torch.randn(N_SAMPLES) * 0.1
    y = y.view(-1, 1)
    
    # Split data
    train_size = int(0.8 * N_SAMPLES)
    X_train, X_valid = X[:train_size], X[train_size:]
    y_train, y_valid = y[:train_size], y[train_size:]
    
    # Create data loaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    valid_loader = DataLoader(TensorDataset(X_valid, y_valid), batch_size=32)
    
    # Create model and optimizer
    model = SimpleNet(INPUT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Training loop
    train_losses = []
    valid_losses = []
    
    print("Initial model parameters:")
    for name, param in model.named_parameters():
        print(f"{name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")
    
    for epoch in range(50):
        # Training
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            
            # Print gradient information for first epoch
            if epoch == 0 and batch_idx == 0:
                print("\nInitial gradients:")
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(f"{name} grad: mean={param.grad.mean():.4f}, std={param.grad.std():.4f}")
            
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_train_loss = epoch_loss / batch_count
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        valid_loss = 0
        batch_count = 0
        
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                output = model(X_batch)
                loss = criterion(output, y_batch)
                valid_loss += loss.item()
                batch_count += 1
        
        avg_valid_loss = valid_loss / batch_count
        valid_losses.append(avg_valid_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Valid Loss = {avg_valid_loss:.4f}")
    
    print("\nFinal model parameters:")
    for name, param in model.named_parameters():
        print(f"{name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_debug.png')
    plt.close()

if __name__ == "__main__":
    main()