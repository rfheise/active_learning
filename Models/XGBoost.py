import numpy as np
import torch 
from torchvision import models 
from torch import nn
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from ..Logger import Logger
from xgboost import XGBClassifier
from sklearn.metrics import log_loss
from .Model import Model

class XGBoost(Model):

    def __init__(self):
        """
            initialize your model
            click on LeNet.py to view an example
        """
        super().__init__()
        self.model = XGBClassifier()
        self.loss = log_loss
    def fit(self,X,y):

        """trains model with training data 
        X: training data
        y: targets

        returns: model accuracy, loss on training data
        """
        self.model.fit(X,y)
        preds = self.model.predict_proba(X)
        acc = (preds.argmax(axis=1) == y).astype(np.float64).sum()/y.shape[0]
        loss = self.loss(y, preds)
        return acc, loss
    
    def pred_proba(self, X):

        """
        predicts class probabilities 
        returns: numpy array of class predictions should be shape (m,c)
            m - number of training examples 
            c - number of classes
        """
        # make probability predictions for X 
        return self.model.predict_proba(X)
    
    def clear(self):

        """
        resets model for another round of training
        just reset the models params
        """
        self.model = XGBClassifier()
    
    @staticmethod
    def clone(other):
        """
        creates clone of other with same params
        returns another cloned model
        """

        return XGBoost()

    def set_hyper_params(self):
        #optional method to update hyper_params dict 
        #used for logging 

        """
        optional method used to update hyper_params dict for logging
        used to log modified hyper-parameters for each test
        """
        self.hyper_params = {
            "model": "XGBoost",
            "loss":"CE",
            "clear": "new model"
        }


