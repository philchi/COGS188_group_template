import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import *

class SyntheticGradient(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SyntheticGradient, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)


    def forward(self, x):
        return self.fc(x)



class ModuleWithSyntheticGradient(nn.Module):
    def __init__(self) -> None:
        super(ModuleWithSyntheticGradient, self).__init__()
        
        self.sequential = nn.Sequential()
        self.synthetic_gradient = None
        
        self.weight_optim = None
        self.synthetic_gradient_optim = None
        
        self.synthetic_grad = None
        
        
    def add_module(self, name, module):
        """
            Add an nn.Module to this module. Modules added will be executed sequentially.
        """
        self.sequential.add_module(name, module)
        
    
    def enable_training(self, dim):
        """
            Initialize the optimizer for weights
        """
        self.weight_optim = optim.Adam(
            self.sequential.parameters(),
            lr = 1e-4
        )
        
        self.synthetic_gradient = SyntheticGradient(dim, dim)
        self.synthetic_gradient_optim = optim.Adam(
            self.synthetic_gradient.parameters(),
            lr = 1e-4
        )
        
    
    def train_weight(self, x):
        """
            Update the weight
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
    
    
    def train_synthetic_grad(self, true_grad):
        """
            Update syenthetic gradient generator
        """
        loss = F.mse_loss(self.synthetic_grad, true_grad)
        loss.backward()
        
        self.synthetic_gradient_optim.step()
        self.synthetic_gradient_optim.zero_grad()
        
        return loss.item()
    
    
    def forward(self, x):
        return self.sequential(x)
    
    

class RegularModule(nn.Module):
    def __init__(self) -> None:
        super(RegularModule, self).__init__()
        
        self.sequential = nn.Sequential()
        self.weight_optim = None
        
        
    def add_module(self, name, module):
        """
            Add an nn.Module to this module. Modules added will be executed sequentially.
        """
        self.sequential.add_module(name, module)
        
    
    def enable_training(self):
        """
            Initialize the optimizer for weights
        """
        self.weight_optim = optim.Adam(
            self.sequential.parameters(),
            lr = 1e-4
        )
        
        
    def train_weight(self, x, label, loss = F.cross_entropy):
        """
            Update the weight
        """
        # forward pass
        x = self.sequential(x)
        
        # apply gradient descent
        _loss = loss(x, label)
        _loss.backward()
        self.weight_optim.step()
        self.weight_optim.zero_grad()
        
        return _loss.item()
    
    
    def forward(self, x):
        return self.sequential(x)