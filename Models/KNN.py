import numpy as np
import torch 
from torchvision import models 
from torch import nn
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from ..Logger import Logger
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss
from .Model import Model

class KNN_AL(Model):
    def __init__(self):
        super().__init__()
        self.k = 1
        self.model = KNeighborsClassifier(n_neighbors=self.k)
        
        self.total_classes = 10
        self.classes_searched = []

    def loss(self, y_true,y_preds ):
        return log_loss(y_true, y_preds, labels=np.arange(self.total_classes))
    
    def fit(self,X,y):

        """trains model with training data 
        X: training data
        y: targets

        returns: model accuracy, loss on training data
        """
        self.model.fit(X,y)
        preds = self.model.predict_proba(X)
        # acc = (preds.argmax(axis=1) == y).astype(np.float64).sum()/y.shape[0]
        # loss = self.loss(y, preds)
        for y_i in y:
            y_i = int(y_i)
            if len(self.classes_searched) == 0:
                self.classes_searched.append(y_i)
            elif y_i not in self.classes_searched:
                self.classes_searched.append(y_i)
        #print(self.classes_searched)
        self.classes_searched.sort()
        return 0,0
    
    def pred_proba(self, X):
        pred_prob = self.model.predict_proba(X)
        #print(self.classes_searched)
        new_pred_proba = np.zeros(shape=[len(X), self.total_classes])
        for i in range(len(self.classes_searched)):
            idx = self.classes_searched[i]
            for j in range(len(pred_prob)):
                new_pred_proba[j,idx] = pred_prob[j,i]
        #print(new_pred_proba[0])
        # print(new_pred_proba.shape)
        """
        predicts class probabilities 
        returns: numpy array of class predictions should be shape (m,c)
            m - number of training examples 
            c - number of classes
        """
        # make probability predictions for X 
        return new_pred_proba
    
    def clear(self):

        """
        resets model for another round of training
        just reset the models params
        """
        self.model = KNeighborsClassifier(n_neighbors=self.k)
    
    @staticmethod
    def clone(other):
        """
        creates clone of other with same params
        returns another cloned model
        """

        return KNN_AL()
    
    def initializer(X, y):
        y_u = np.unique(y, return_index=True)
        for y_i in y_u:
            pass
        return None

    def set_hyper_params(self):
        #optional method to update hyper_params dict 
        #used for logging 

        """
        optional method used to update hyper_params dict for logging
        used to log modified hyper-parameters for each test
        """
        self.hyper_params = {
            "model": "KNN",
            "loss":"CE",
            "clear": "new model"
        }