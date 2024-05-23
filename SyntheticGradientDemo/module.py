import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import *
from numbers import Number

import torch.utils
import torch.utils.data

class SyntheticGradient(nn.Module):
    """
        A simple one-layer perceptron/neural network/linear transformation
        for generating synthetic gradient
    """
    def __init__(self, input_dim, output_dim):
        super(SyntheticGradient, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)


    def forward(self, x):
        return self.fc(x)



class ModuleWithSyntheticGradient(nn.Module):
    """
        A wrapping module which aims to utilize synthetic gradient instead of true gradients to update the weights. 
        
        The initialization process won't add any layer into this module. One should call `add_module` method to add layers into this module. Please be aware that layers added will be executed sequentially.
        
        In order to train this module, one should first call `enable_training` first and use `train_weight` method.
    """
    def __init__(self):
        super(ModuleWithSyntheticGradient, self).__init__()
        
        self.sequential: nn.Sequential = nn.Sequential()
        self.synthetic_gradient: SyntheticGradient = None
        
        self.weight_optim: optim.Optimizer = None
        self.synthetic_gradient_optim: optim.Optimizer = None
        
        self.synthetic_grad: torch.Tensor = None
        
        
    def add_module(self, module: nn.Module, name: str = None):
        """
            Add an nn.Module to this module. Modules added will be executed sequentially.

            Args:
                module (nn.Module): The module/layer being added.
                name (str, optional): The name of the given module/layer. If None is given, the name will be 'layer_{i}'
        """
        if not name:
            name = f'layer_{len(self.sequential)}'
        self.sequential.add_module(name, module)
        
    
    def enable_training(self, dim: int, lr: float = 1e-4):
        """
            Initialize the optimizer for weights.

            Args:
                dim (int): The dimension of synthetic gradient.
                lr (float): The learning rate. Default to 1e-4.
        """
        self.weight_optim = optim.Adam(
            self.sequential.parameters(),
            lr = lr
        )
        
        self.synthetic_gradient = SyntheticGradient(dim, dim)
        self.synthetic_gradient_optim = optim.Adam(
            self.synthetic_gradient.parameters(),
            lr = lr
        )
        
    
    def train_weight(self, x: torch.Tensor) -> torch.Tensor:
        """
            Update model weights. Note that we don't need to know what the true label is. The synthetic gradient will take over that.

            Args:
                x (torch.Tensor): Input tensor

            Returns:
                torch.Tensor: The output of this module.
        """
        # forward pass
        x = self.sequential(x)
        
        # apply synthetic gradient
        self.synthetic_grad = self.synthetic_gradient(x.detach())
        x.backward(self.synthetic_grad)
        self.weight_optim.step()
        self.weight_optim.zero_grad()
        
        # detach the output from computational graph
        # set requires_grad to True to handle true grad later
        x = x.detach()
        x.requires_grad = True
        
        return x
    
    
    def train_synthetic_grad(self, true_grad: torch.Tensor) -> Number:
        """
            Update synthetic gradient generator weights. 

            Args:
                true_grad (torch.Tensor): True gradient from the rest of network

            Returns:
                Number: The mse loss between synthetic gradient and true gradient.
        """
        loss = F.mse_loss(self.synthetic_grad, true_grad)
        loss.backward()
        
        self.synthetic_gradient_optim.step()
        self.synthetic_gradient_optim.zero_grad()
        
        return loss.item()
    
    
    def forward(self, x):
        return self.sequential(x)
    
    

class RegularModule(nn.Module):
    """
        A wrapping module which aims to utilize normal gradient to update weights.
        
        The initialization process won't add any layer into this module. One should call `add_module` method to add layers into this module. Please be aware that layers added will be executed sequentially.
        
        In order to train this module, one should first call `enable_training` first and use `train_weight` method.
    """
    def __init__(self) -> None:
        super(RegularModule, self).__init__()
        
        self.sequential = nn.Sequential()
        self.weight_optim = None
        
        
    def add_module(self, module: nn.Module, name: str = None):
        """
            Add an nn.Module to this module. Modules added will be executed sequentially.

            Args:
                module (nn.Module): The module/layer being added.
                name (str, optional): The name of the given module/layer. If None is given, the name will be 'layer_{i}'
        """
        if not name:
            name = f'layer_{len(self.sequential)}'
        self.sequential.add_module(name, module)
        
    
    def enable_training(self, lr: float = 1e-4):
        """
            Initialize the optimizer for weights.

            Args:
                lr (float): The learning rate. Default to 1e-4.
        """
        self.weight_optim = optim.Adam(
            self.sequential.parameters(),
            lr = lr
        )
        
        
    def train_weight(
        self,
        x: torch.Tensor,
        label: torch.Tensor,
        loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]\
            = F.cross_entropy
    ) -> Number:
        """
            Update model weights.

            Args:
                x (torch.Tensor): Input tensor
                label (torch.Tensor): Label tensor
                loss_func (Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional): Loss function. Defaults to F.cross_entropy.

            Returns:
                Number: Current trainning loss.
        """
        # forward pass
        x = self.sequential(x)
        
        # apply gradient descent
        loss = loss_func(x, label)
        loss.backward()
        self.weight_optim.step()
        self.weight_optim.zero_grad()
        
        return loss.item()

    
    def forward(self, x):
        return self.sequential(x)