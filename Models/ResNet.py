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
        
        self.model_loss = torch.nn.CrossEntropyLoss()
        self.initial_weights = self.get_model()
        self.initial_weights.load_state_dict(self.model.state_dict())

        self.epochs = 50 
        self.train_batch_size = 1000

        self.transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomGrayscale(.1),
        transforms.RandomHorizontalFlip(.5),
        transforms.RandomVerticalFlip(.25),
        transforms.Resize(36),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5),),
    ])

    def set_optim(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.0001) 
        # self.scheduler = None 
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30,60,120], gamma=0.1)
    
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
        return model.to(LeNetAL.device)
    
    @staticmethod
    def clone(other):
      net = ResNet18AL()
      net.model.load_state_dict(other.model.state_dict())
      net.initial_weights.load_state_dict(other.model.state_dict())
      return net
    
    def clear(self):
        # self.model.load_state_dict(self.initial_weights.state_dict())
        self.model = self.get_model()
        self.set_optim()

    def set_hyper_params(self):

        self.hyper_params = {
            "model": "Resnet",
            "epochs":self.epochs, 
            "train_batch_size":self.train_batch_size, 
            "loss":"CE",
            "scheduler": "torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30,60,120], gamma=0.2)",
            "optim":"torch.optim.Adam(self.model.parameters(), lr=.0001) ",
            "clear": "new model",
            "num_fitting":self.num_models
        }
