import numpy as np
import torch 
from torchvision import models 
from torch import nn
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from .Model import Model
from ..Datasets import DataSetFromTensor
import time 

class LeNetAL(Model):

    device = "cuda"
    default_transform = transforms.Compose([
        # transforms.Normalize((0.5,),(0.5,),),
    ])

    def __init__(self):
        super().__init__()
        self.model = LeNet().to(LeNetAL.device)
        self.model_loss = torch.nn.CrossEntropyLoss()
        # self.model_loss = BalancedLoss(10,LeNetAL.device)
        self.initial_weights = LeNet()
        self.initial_weights.load_state_dict(self.model.state_dict())
        self.train_batch_size = 250
        self.num_models = 10
        self.epochs = 75
        self.transform_train = None
    
    def set_optim(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.001) 
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30,60,120], gamma=0.1)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, self.epochs - 1)

    def to_data_loader(self, train, X,y, batch_size=100):
        
        X = torch.tensor(X)
        if y is not None:
            y = torch.tensor(y)

        if train and self.transform_train is not None:
              return DataLoader(DataSetFromTensor(X,y, self.transform_train), batch_size=batch_size, num_workers=4,shuffle=train)
        else:
            return DataLoader(DataSetFromTensor(X,y, LeNetAL.default_transform), batch_size=batch_size, num_workers=4,shuffle=train)

    def fit(self,X,y):

        models = []
        # loader = self.to_data_loader(True, X,y, self.train_batch_size)
        indexes = torch.randperm(X.shape[0])
        X_orig = torch.tensor(X).to(LeNetAL.device)
        y_orig = torch.tensor(y).to(LeNetAL.device)
        s = time.time()
        for j in range(self.num_models):
            self.model.train()
            self.set_optim()
            loss = 0
            acc = 0
            for i in range(self.epochs):
                rolling_acc = 0
                rolling_loss = 0
                count = 0
                count_batches = 0
                # print("epoch:",i,"\n")
                # for X,y in loader:
                indexes = torch.randperm(X_orig.shape[0])
                X_perm = X_orig[indexes]
                y_perm = y_orig[indexes]
                for k in range(X_orig.shape[0]//self.train_batch_size + 1):
                    X = X_perm[self.train_batch_size * k: self.train_batch_size * (k+1)]
                    y = y_perm[self.train_batch_size * k: self.train_batch_size * (k+1)]
                    if X.shape[0] == 0:
                        break
                    # X = X.to(LeNetAL.device)
                    # y = y.to(LeNetAL.device)
                    
                    self.optimizer.zero_grad()
                    
                    preds = self.model(X)
                    
                    loss = self.model_loss(preds, y)
                    loss.backward()
                    self.optimizer.step() 
                    rolling_acc += (preds.argmax(dim=1) == y).sum().item()
                    count += X.shape[0]
                    rolling_loss += loss.item()
                    count_batches += 1
                    
                acc, loss = rolling_acc/count, rolling_loss/count_batches
                # print(acc, loss, "\n\n\n")
                if self.scheduler is not None:
                    self.scheduler.step()
            models.append((self.model, acc, loss))
            self.clear() 
        e = time.time()
        # print(e - s)
        min_index = 0
        for j in range(self.num_models):
            # print(models[j][1], models[j][2])
            if models[min_index][2] > models[j][2]:
                min_index = j
        self.model = models[min_index][0]
        # acc, loss
        return models[min_index][1], models[min_index][2]

    def pred_proba(self,X):

        self.model.eval()

        # loader = self.to_data_loader(False, X, None, 60000)
        preds = []
        X = torch.tensor(X)
        with torch.no_grad():
            # for X in loader:
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
    
    def loss(self, y_true, y_pred):
        
        return self.model_loss(torch.tensor(y_pred), torch.tensor(y_true)).item()
            


    def clear(self):
        # self.model.load_state_dict(self.initial_weights.state_dict())
        self.model = LeNet().to(LeNetAL.device)
        self.set_optim()

    def set_hyper_params(self):

        self.hyper_params = {
            "model": "LeNet Fast",
            "epochs":self.epochs, 
            "train_batch_size":self.train_batch_size, 
            "loss":"CE",
            "scheduler": "torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30,60,120], gamma=0.1)",
            "optim":"default adam",
            "clear": "new model",
            "num_fitting":self.num_models
        }

        
class BalancedLoss(nn.Module):

    def __init__(self, num_classes, device):
        super().__init__()
        self.num_classes = num_classes 
        self.device = device

    def forward(self, preds, y):
        preds = preds.to(self.device)
        y = y.to(self.device)
        weights = torch.unique(torch.cat((y,torch.arange(self.num_classes).to(self.device)),0), return_counts=True)[1].to(self.device)
        weights = 1/weights
        weights = weights/weights.sum()
        return nn.functional.cross_entropy(preds, y, weight=weights)

class LeNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5),stride=(1,1)),
            nn.MaxPool2d(kernel_size=(5,5),stride=(1,1)),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5),stride=(1,1)),
            nn.MaxPool2d(kernel_size=(5,5),stride=(1,1)),
            nn.Conv2d(in_channels=16, out_channels=20, kernel_size=(4,4),stride=(1,1)),
            nn.MaxPool2d(kernel_size=(3,3),stride=(1,1)),
            nn.Conv2d(in_channels=20, out_channels=32, kernel_size=(4,4),stride=(1,1)),
            nn.MaxPool2d(kernel_size=(2,2),stride=(1,1)),
            nn.Flatten(),
            nn.Linear(288, 256), 
            nn.ReLU(), 
            nn.Linear(256,64),
            nn.ReLU(), 
            nn.Linear(64, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, X):

        return self.layers(X)