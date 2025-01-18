import torch
from torch.optim import Optimizer
import math

class OptiBrain(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Get grad
                grad = p.grad.data
                
                # Handle weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias correction terms
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute step size with bias correction
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Compute denominator with epsilon
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                # Update parameters
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
class AdaptiveMomentum(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)
        
        # Initialize state
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['momentum_buffer'] = torch.zeros_like(p.data)
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
                
                # Get momentum buffer
                momentum_buffer = state['momentum_buffer']
                
                # Update momentum buffer
                momentum_buffer.mul_(group['momentum']).add_(grad)
                
                # Compute adaptive learning rate
                lr = group['lr'] / (1 + 0.1 * state['step'])
                
                # Update parameters
                p.data.add_(momentum_buffer, alpha=-lr)
        
        return loss

class ElasticLR(Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        
        # Initialize state
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['prev_grad'] = torch.zeros_like(p.data)
    
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
                
                # Compute gradient change
                grad_diff = grad - state['prev_grad']
                grad_norm = grad.norm().item()
                
                # Compute adaptive learning rate
                lr = group['lr'] / (1 + grad_norm)
                
                # Update parameters
                p.data.add_(grad, alpha=-lr)
                
                # Store current gradient
                state['prev_grad'] = grad.clone()
        
        return loss