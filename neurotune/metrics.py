import numpy as np
from typing import List, Dict, Any
import torch

class ConvergenceTracker:
    """
    Tracks and analyzes model convergence metrics during training.
    """
    def __init__(self, window_size: int = 100, threshold: float = 1e-4):
        self.window_size = window_size
        self.threshold = threshold
        self.loss_history: List[float] = []
        self.gradient_norms: List[float] = []
        
    def update(self, loss: float, model_parameters: List[torch.Tensor]) -> Dict[str, Any]:
        """
        Updates tracking metrics with new loss value and model parameters.
        
        Args:
            loss: Current loss value
            model_parameters: List of model parameters
            
        Returns:
            Dict containing convergence metrics
        """
        self.loss_history.append(loss)
        
        # Calculate gradient norms
        total_norm = 0.0
        for param in model_parameters:
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        self.gradient_norms.append(total_norm)
        
        return self.compute_metrics()
    
    def compute_metrics(self) -> Dict[str, Any]:
        """Computes convergence metrics based on collected history."""
        if len(self.loss_history) < 2:
            return {}
            
        metrics = {
            'loss_trend': self._compute_loss_trend(),
            'gradient_stability': self._compute_gradient_stability(),
            'is_converged': self._check_convergence()
        }
        
        return metrics
    
    def _compute_loss_trend(self) -> float:
        """Computes the trend in loss values over recent iterations."""
        if len(self.loss_history) < self.window_size:
            return 0.0
            
        recent_losses = self.loss_history[-self.window_size:]
        return np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
    
    def _compute_gradient_stability(self) -> float:
        """Computes the stability of gradients over recent iterations."""
        if len(self.gradient_norms) < self.window_size:
            return 0.0
            
        recent_norms = self.gradient_norms[-self.window_size:]
        return np.std(recent_norms) / (np.mean(recent_norms) + 1e-8)
    
    def _check_convergence(self) -> bool:
        """Checks if the model has converged based on defined criteria."""
        if len(self.loss_history) < self.window_size:
            return False
            
        recent_losses = self.loss_history[-self.window_size:]
        loss_std = np.std(recent_losses)
        return loss_std < self.threshold

class LossAnalyzer:
    """
    Analyzes loss landscape characteristics to guide optimization.
    """
    def __init__(self):
        self.loss_samples = []
        
    def add_sample(self, parameters: Dict[str, torch.Tensor], loss: float):
        """Adds a new sample point to the loss landscape analysis."""
        # Implementation details...
        pass
        
    def compute_curvature(self) -> float:
        """Computes the local curvature of the loss landscape."""
        # Implementation details...
        pass
