import numpy as np
from .LeNet import LeNetAL
import torch 
from torchvision import models 
from torch import nn
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader


class ResNet18AL(LeNetAL):

    def __init__(self):
        
        super().__init__()
        self.model = self.get_model()
        self.model = self.model.to(LeNetAL.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.001) 
        self.loss = torch.nn.CrossEntropyLoss()
        self.initial_weights = self.get_model()
        self.initial_weights.load_state_dict(self.model.state_dict())

    @staticmethod
    def get_model():
        model = models.resnet18()
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10), 
            nn.Softmax(dim=1)
        )
        return model
    
    @staticmethod
    def clone(other):
      net = ResNet18AL()
      net.model.load_state_dict(other.model.state_dict())
      net.initial_weights.load_state_dict(other.model.state_dict())
      return net
