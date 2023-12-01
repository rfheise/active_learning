import numpy as np
from LeNet import LeNet, BalancedLoss
import torch 
from torchvision import models 
from torch import nn
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from Cifar import DataSetFromTensor

class Model():

    def __init__(self):
        pass

    def fit(self,X,y):

        # train model
        pass

    def pred(self,X):

        # make predictions on X
        return np.argmax(self.pred_proba(self, X))
    
    def pred_proba(self, X):

        # make probability predictions for X 
        return None
    
    def clear(self):

        # clear params
        pass
    
    @staticmethod
    def clone():
        pass

class LeNetAL():

    device = "cuda"
    default_transform = transforms.Compose([
        transforms.Normalize((0.5,),(0.5,),),
    ])
    epochs = 150

    def __init__(self):
        self.model = LeNet().to(LeNetAL.device)
        
        # self.loss = torch.nn.CrossEntropyLoss()
        self.loss = BalancedLoss(10,LeNetAL.device)
        self.initial_weights = LeNet()
        self.initial_weights.load_state_dict(self.model.state_dict())
    
    def set_optim(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.001) 
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30,60,120], gamma=0.1)
        
    def to_data_loader(self, train, X,y, batch_size=50):
        
        X = torch.tensor(X)
        if y is not None:
            y = torch.tensor(y)

        return DataLoader(DataSetFromTensor(X,y, LeNetAL.default_transform), batch_size=batch_size, num_workers=4,shuffle=train)

    def fit(self,X,y):

        self.model.train()
        loader = self.to_data_loader(True, X,y)
        
        loss = 0
        acc = 0

        for i in range(LeNetAL.epochs):
            rolling_acc = 0
            rolling_loss = 0
            count = 0
            count_batches = 0
            for X,y in loader:

                X = X.to(LeNetAL.device)
                y = y.to(LeNetAL.device)
                self.optimizer.zero_grad()
                
                preds = self.model(X)
                
                loss = self.loss(preds, y)
                loss.backward()
                self.optimizer.step() 
                rolling_acc += (preds.argmax(dim=1) == y).sum().item()
                count += X.shape[0]
                rolling_loss += loss.item()
                count_batches += 1
            loss, acc = rolling_acc/count, rolling_loss/count_batches
            self.scheduler.step()

        return loss, acc

    def pred_proba(self,X):

        self.model.eval()

        loader = self.to_data_loader(False, X, None, 60000)
        preds = []
    
        with torch.no_grad():
            for X in loader:
                X = X.to(LeNetAL.device)
                preds.append(self.model(X))
        return torch.cat(preds,0).cpu().numpy()
    
    @staticmethod
    def clone(other):
      net = LeNetAL()
      net.model.load_state_dict(other.model.state_dict())
      net.initial_weights.load_state_dict(net.model.state_dict())
      net.set_optim()
      return net

    def get_stat_val(self, X, y):
        
        y = torch.tensor(y)
        proba = torch.tensor(self.pred_proba(X))
        loss = self.loss(proba, y).item()
        acc = (proba.argmax(dim=1) == y).to(torch.float).mean().item()
        return acc, loss
            


    def clear(self):
        self.model.load_state_dict(self.initial_weights.state_dict())
        self.model = LeNet().to(LeNetAL.device)
        self.set_optim()


class ResNet18AL(LeNetAL):

    def __init__(self):
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
