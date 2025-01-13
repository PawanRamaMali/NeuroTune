import torch
from torch.optim import Optimizer
from typing import List, Optional, Dict, Any
import math  # Added missing import

class OptiBrain(Optimizer):
    """
    OptiBrain optimizer: A sophisticated optimizer that adapts learning rates
    based on parameter importance and gradient history.
    """
    def __init__(
        self,
        params: List[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        importance_sampling: bool = True
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            importance_sampling=importance_sampling
        )
        super().__init__(params, defaults)
        
        self.init_buffers()
    
    def init_buffers(self):
        """Initialize optimizer buffers for gradient history and momentum."""
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
                if group['importance_sampling']:
                    state['importance_score'] = torch.ones_like(p)

    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """
        Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                # Update step count
                state['step'] += 1
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                # Update momentum and variance
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Compute bias-corrected moments
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Apply importance sampling if enabled
                if group['importance_sampling']:
                    importance_score = state['importance_score']
                    grad_importance = torch.abs(grad) / (torch.std(grad) + group['eps'])
                    importance_score.mul_(0.9).add_(grad_importance, alpha=0.1)
                    step_size = group['lr'] * importance_score
                else:
                    step_size = group['lr']
                
                # Update parameters
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                p.data.addcdiv_(exp_avg / bias_correction1, denom, value=-step_size)
        
        return loss

class AdaptiveMomentum(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9, variance_window=10):
        defaults = dict(lr=lr, momentum=momentum, variance_window=variance_window)
        super().__init__(params, defaults)
        
        # Initialize optimizer state
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['momentum_buffer'] = torch.zeros_like(p.data)
                state['variance_history'] = []
    
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
                
                # Update variance history
                if len(state['variance_history']) >= group['variance_window']:
                    state['variance_history'].pop(0)
                state['variance_history'].append(grad.var().item())
                
                # Compute adaptive learning rate
                if len(state['variance_history']) > 0:
                    variance = sum(state['variance_history']) / len(state['variance_history'])
                    adaptive_lr = group['lr'] / (1 + variance)**0.5
                else:
                    adaptive_lr = group['lr']
                
                # Update momentum buffer
                momentum_buffer = state['momentum_buffer']
                momentum_buffer.mul_(group['momentum']).add_(grad)
                
                # Update parameters
                p.data.add_(momentum_buffer, alpha=-adaptive_lr)
        
        return loss

class ElasticLR(Optimizer):
    def __init__(self, params, lr=1e-3, curvature_window=5):
        defaults = dict(lr=lr, curvature_window=curvature_window)
        super().__init__(params, defaults)
        
        # Initialize optimizer state
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['prev_grad'] = torch.zeros_like(p.data)
                state['step'] = 0
    
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
                state['step'] += 1
                
                # Compute curvature estimate
                grad_diff = grad - state['prev_grad']
                curvature = torch.abs(grad_diff).mean().item()
                
                # Update learning rate based on curvature
                adaptive_lr = group['lr'] / (1 + curvature)
                
                # Update parameters
                p.data.add_(grad, alpha=-adaptive_lr)
                
                # Store current gradient for next iteration
                state['prev_grad'] = grad.clone()
        
        return loss