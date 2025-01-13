import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import math

# Optimizer implementations
class OptiBrain(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                # Get parameters for this group
                lr = group['lr']
                beta1, beta2 = group['betas']
                eps = group['eps']

                # Update step count
                state['step'] += 1

                # Update moving averages
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias corrections
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Update parameters
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                denom = exp_avg_sq.sqrt().add_(eps)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

# Simple test model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1, bias=True)
    
    def forward(self, x):
        return self.layer(x)

def test_optimizer(optimizer_class, lr=0.01, epochs=100):
    # Create simple dataset: y = 2x + 1 with some noise
    X = torch.linspace(-5, 5, 100).reshape(-1, 1)
    y = 2 * X + 1 + torch.randn_like(X) * 0.1
    
    # Create model and optimizer
    model = SimpleNet()
    optimizer = optimizer_class(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Training loop
    losses = []
    print(f"\nTesting {optimizer_class.__name__}")
    print("Initial parameters:", dict(model.named_parameters()))
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        
        # Print gradients for first epoch
        if epoch == 0:
            print("Initial gradients:")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"{name} grad: {param.grad.mean().item():.4f}")
        
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    print("Final parameters:", dict(model.named_parameters()))
    print(f"Final loss: {losses[-1]:.4f}")
    return losses

def main():
    # Test OptiBrain
    plt.figure(figsize=(10, 5))
    
    losses = test_optimizer(OptiBrain)
    plt.plot(losses, label='OptiBrain')
    
    # Compare with Adam
    losses_adam = test_optimizer(torch.optim.Adam)
    plt.plot(losses_adam, label='Adam (reference)')
    
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.savefig('optimizer_test_results.png')
    plt.close()

if __name__ == "__main__":
    main()