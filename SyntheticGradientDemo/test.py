from module import ModuleWithSyntheticGradient as Syn
from module import RegularModule as Reg

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

if __name__ == "__main__":
    '''
        Conv part:
        input(32x32x3) -> conv_1(3x3 32channels) -> max_pooling (16x16 32channels)
        -> conv_2(3x3 64channels) -> max_pooling (8x8 64channels)
        -> conv_2(3x3 64channels) -> max_pooling (4x4 64channels)
        -> flatten (1024)
        
        MLP:
        flattened vector -> Linear(1024->512) -> Linear(512->10)
    '''
    
    
    '''
        Initialize the neural network
    '''
    conv = Syn()

    conv.add_module(name='conv_1', module=nn.Conv2d(3, 32, kernel_size=3, padding=1))
    conv.add_module(name='relu_1', module=nn.ReLU())
    conv.add_module(name='maxpooling_1', module=nn.MaxPool2d(kernel_size=2, stride=2))

    conv.add_module(name='conv_2', module=nn.Conv2d(32, 64, kernel_size=3, padding=1))
    conv.add_module(name='relu_2', module=nn.ReLU())
    conv.add_module(name='maxpooling_2', module=nn.MaxPool2d(kernel_size=2, stride=2))

    conv.add_module(name='conv_3', module=nn.Conv2d(64, 64, kernel_size=3, padding=1))
    conv.add_module(name='relu_3', module=nn.ReLU())
    conv.add_module(name='maxpooling_3', module=nn.MaxPool2d(kernel_size=2, stride=2))

    conv.add_module(name='flatten', module=nn.Flatten())

    conv.enable_training(1024) # final output will be 64(C) * 4(H) * 4(W)


    dense = Reg()

    dense.add_module(name='fc_1', module=nn.Linear(1024, 512))
    dense.add_module(name='relu_1', module=nn.ReLU())

    dense.add_module(name='fc_2', module=nn.Linear(512, 10))

    dense.enable_training()


    '''
        Prepare data
    '''
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    '''
        Finalize preparation
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conv.to(device)
    dense.to(device)
    
    
    '''
        Trainning process
    '''
    for epoch in range(1,51):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            conv_output = conv.train_weight(data)
            loss = dense.train_weight(conv_output, target)
            synthetic_grad_loss = conv.train_synthetic_grad(conv_output.grad)
        
        with torch.no_grad():
            correct_num = 0
            total_num = 0
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                conv_output = conv(data)
                mlp_output = dense(conv_output)
                result = torch.argmax(mlp_output, dim=-1)
                print(result == target)
                correct_num += torch.sum(result == target).item()
                total_num += result.shape[0]
                
        print(f'Train Epoch: {epoch}\tLoss: {loss:.6f}\tSynthetic Loss: {synthetic_grad_loss:.6f}\tAccuracy: {correct_num/total_num}')