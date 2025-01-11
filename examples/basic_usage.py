import torch
import torch.nn as nn
from neurotune import OptiBrain, ConvergenceTracker
import matplotlib.pyplot as plt

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

def main():
    # Create model and dummy data
    model = SimpleNet()
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    
    # Initialize optimizer and convergence tracker
    optimizer = OptiBrain(model.parameters(), lr=0.01)
    tracker = ConvergenceTracker()
    
    # Lists to store metrics for plotting
    losses = []
    convergence_scores = []
    
    # Training loop
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(X)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        
        # Update parameters and track convergence
        optimizer.step()
        metrics = tracker.update(loss.item(), model.parameters())
        
        # Store metrics
        losses.append(loss.item())
        if metrics.get('gradient_stability'):
            convergence_scores.append(metrics['gradient_stability'])
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
            if metrics:
                print(f"Convergence Metrics: {metrics}")
    
    # Plot training progress
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(convergence_scores)
    plt.title('Gradient Stability')
    plt.xlabel('Epoch')
    plt.ylabel('Stability Score')
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.close()

if __name__ == "__main__":
    main()
