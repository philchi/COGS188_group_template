from module import ModuleWithSyntheticGradient as Syn
from module import RegularModule as Reg

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

if __name__ == "__main__":
    '''
        Initialize the neural network
    '''
    conv = Syn()

    conv.add_module('conv_1', nn.Conv2d(3, 32, kernel_size=3, padding=1))
    conv.add_module('relu_1', nn.ReLU())
    conv.add_module('maxpooling_1', nn.MaxPool2d(kernel_size=2, stride=2))

    conv.add_module('conv_2', nn.Conv2d(32, 64, kernel_size=3, padding=1))
    conv.add_module('relu_2', nn.ReLU())
    conv.add_module('maxpooling_2', nn.MaxPool2d(kernel_size=2, stride=2))

    conv.add_module('conv_3', nn.Conv2d(64, 64, kernel_size=3, padding=1))
    conv.add_module('relu_3', nn.ReLU())
    conv.add_module('maxpooling_3', nn.MaxPool2d(kernel_size=2, stride=2))

    conv.add_module('flatten', nn.Flatten())

    conv.enable_training(1024) # final output will be 64(C) * 4(H) * 4(W)


    dense = Reg()

    dense.add_module('fc_1', nn.Linear(1024, 512))
    dense.add_module('relu_1', nn.ReLU())

    dense.add_module('fc_2', nn.Linear(512, 10))

    dense.enable_training()


    '''
        Prepare data
    '''
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    
    '''
        Finalize preparation
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conv.to(device)
    dense.to(device)
    
    
    '''
        Trainning process
    '''
    for epoch in range(1,11):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            conv_output = conv.train_weight(data)
            loss = dense.train_weight(conv_output, target)
            synthetic_grad_loss = conv.train_synthetic_grad(conv_output.grad)
        
        print(f'Train Epoch: {epoch} \tLoss: {loss:.6f}\tSynthetic Loss: {synthetic_grad_loss:.6f}')