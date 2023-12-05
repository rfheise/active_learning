import numpy as np
import torch 
from torchvision import models 
from torch import nn
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from ..Logger import Logger


class Model():

    def __init__(self):
        self.hyper_params = {}

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

    def set_hyper_params(self):
        #optional method to update hyper_params dict 
        #used for logging 
        pass

    def log_hyper_parameters(self):
        
        #call logger with hyperparameters
        self.set_hyper_params()
        Logger.log_hyper_parameters(**self.hyper_params)
    

