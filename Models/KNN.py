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
import copy

class KNN_AL(Model):
    def __init__(self):
        super().__init__()
        self.k = 1
        self.model = KNeighborsClassifier(n_neighbors=self.k)
        
        self.total_classes = 10
        self.classes_searched = []

    def loss(self, y_true,y_preds):
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
        """
        predicts class probabilities 
        returns: numpy array of class predictions should be shape (m,c)
            m - number of training examples 
            c - number of classes
        """
        # make probability predictions for X 
        pred_prob = self.model.predict_proba(X)
        new_pred_proba = np.zeros(shape=[len(X), self.total_classes])
        for i in range(len(self.classes_searched)):
            idx = self.classes_searched[i]
            for j in range(len(pred_prob)):
                new_pred_proba[j,idx] = pred_prob[j,i]
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
    
    def initializer(X, y, num_of_points):
        bias_y = np.where(y<3.0)
        bias_y = np.flip(bias_y)
        init_X = []
        init_y = []
        unlbld_X = copy.deepcopy(X)
        unlbld_y = copy.deepcopy(y)
        for i in range(num_of_points):
            init_X.append(X[bias_y[0,i]])
            init_y.append(y[bias_y[0,i]])
            np.delete(unlbld_X, bias_y[0,i])
            np.delete(unlbld_y, bias_y[0,i])
        labeled_data = [np.array(init_X), np.array(init_y)]
        return labeled_data, unlbld_X, unlbld_y

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