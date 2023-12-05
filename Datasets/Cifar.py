from torchvision import datasets
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
import numpy as np
import torch 

def get_classes_cifar(data):
    targets = torch.tensor(data.targets)
    indexes = (targets == 0 ) | (targets == 1)
    data.targets = targets[indexes]
    data.data = data.data[indexes]
    data.targets[data.targets == 0] = 0
    data.targets[data.targets == 1] = 1
    return data

    

def get_data_car_plane():

    default = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5),),
    ])
    train_data = datasets.CIFAR10(root='./data', train=True, transform=default, download=True)
    test_data = datasets.CIFAR10(root='./data', train=False, transform=default, download=True)
    # train_data = datasets.MNIST(root='./data',train=True, download=True,transform=default)
    # test_data = datasets.MNIST(root='./data',train=False,download=True,transform=default)
    train_data = get_classes_cifar(train_data)
    test_data = get_classes_cifar(test_data)

    train_loader = DataLoader(train_data, batch_size=500000,num_workers=4, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=500000, num_workers=4, shuffle=True)

    for X, y in train_loader:
        X_train, y_train = np.array(X), np.array(y)
    for X,y in test_loader:
        X_test, y_test = np.array(X), np.array(y)

    return X_train, y_train, X_test, y_test


