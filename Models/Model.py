import numpy as np
import torch 
from torchvision import models 
from torch import nn
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from ..Logger import Logger


class Model():

    def __init__(self):
        """
            initialize your model
            click on LeNet.py to view an example
        """
        self.hyper_params = {}

    def fit(self,X,y):

        """trains model with training data 
        X: training data
        y: targets

        returns: model accuracy, loss on training data
        """
        acc = 0 
        loss = 0
        return acc, loss

    def pred(self,X):

        # make predictions on X
        return np.argmax(self.pred_proba(X),axis=1)
    
    def pred_proba(self, X):

        """
        predicts class probabilities 
        returns: numpy array of class predictions should be shape (m,c)
            m - number of training examples 
            c - number of classes
        """
        # make probability predictions for X 
        return None
    
    def clear(self):

        """
        resets model for another round of training
        just reset the models params
        """
        pass
    
    @staticmethod
    def clone(other):
        """
        creates clone of other with same params
        returns another cloned model
        """
        return Model()

    def set_hyper_params(self):
        #optional method to update hyper_params dict 
        #used for logging 

        """
        optional method used to update hyper_params dict for logging
        used to log modified hyper-parameters for each test
        """
        self.hyper_params = {}

    def log_hyper_parameters(self):
        
        #call logger with hyperparameters
        self.set_hyper_params()
        Logger.log_hyper_parameters(**self.hyper_params)
    
    def loss(self, y_true, y_pred):
        """custom loss function

        Args:
            y_true (_type_): true values
            y_pred (_type_): predicted probabilities
        Returns:
            (int): loss value
        """
        return 0


