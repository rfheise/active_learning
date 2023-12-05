from torchvision import datasets
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
import numpy as np
import torch 

def get_data():

    """
    generic dataset func
    gets dataset for training a model 
    returns 4 numpy arrays (X_train, y_train, X_test, y_test)
    click on MNIST.py folder to view example
    """
    X_train,y_train, X_test, y_test = (None,None,None,None)
    return X_train, y_train, X_test, y_test


class DataSetFromTensor(Dataset):

    def __init__(self,X,y=None, transform=None):
        self.data = X
        self.targets = y
        self.transform=transform

    def __getitem__(self, index):
        
        new_data = self.data[index]
        if self.transform:
            new_data = self.transform(new_data)
        
        if self.targets is not None:
            return new_data,self.targets[index]
        
        return new_data

    def __len__(self):
        return self.data.shape[0]