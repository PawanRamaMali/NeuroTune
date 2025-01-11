import torch
import torch.nn as nn
from neurotune import Optimizer
from neurotune.utils import setup_logging

class CustomOptimizer(Optimizer):
    """
    Example of implementing a custom optimizer using the NeuroTune framework.
    """
    def __init__(self, params, lr=1e-3, momentum=0.9):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)
        
        # Initialize optimizer-specific buffers
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['momentum_buffer'] = torch.zeros_like(p.data)
    
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = p.grad.data
                state = self.state[p]
                
                # Update momentum buffer
                momentum_buffer = state['momentum_buffer']
                momentum_buffer.mul_(group['momentum']).add_(d_p)
                
                # Update parameters
                p.data.add_(momentum_buffer, alpha=-group['lr'])
        
        return loss

def main():
    # Setup logging
    setup_logging()
    
    # Create a simple model and data
    model = nn.Linear(10, 1)
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    
    # Initialize custom optimizer
    optimizer = CustomOptimizer(model.parameters(), lr=0.01)
    
    # Training loop
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(X)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

if __name__ == "__main__":
    main()
