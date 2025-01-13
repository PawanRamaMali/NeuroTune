import logging
from typing import Dict, Any
import torch

def setup_logging(level: str = "INFO") -> None:
    """Sets up logging configuration for the package.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def initialize_parameters(
    model: torch.nn.Module,
    method: str = "xavier",
    gain: float = 1.0
) -> None:
    """Initializes model parameters using specified initialization method.
    
    Args:
        model: PyTorch model
        method: Initialization method ('xavier', 'kaiming', 'normal')
        gain: Scaling factor for initialization
    """
    for param in model.parameters():
        if len(param.shape) > 1:  # For weight matrices
            if method == "xavier":
                torch.nn.init.xavier_uniform_(param, gain=gain)
            elif method == "kaiming":
                torch.nn.init.kaiming_uniform_(param, a=gain)
            elif method == "normal":
                torch.nn.init.normal_(param, std=gain)
        else:  # For bias terms
            torch.nn.init.zeros_(param)