from torchvision import datasets
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
import numpy as np
import torch 
from sklearn.model_selection import train_test_split
import os
import pandas as pd

def get_knn_data():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_path, "simple_2d")
    file_path = os.path.join(data_path, "knn_data2.csv")

    reader = pd.read_csv(file_path)
    X = reader.iloc[:,:-1].to_numpy() 
    y = reader.iloc[:, -1].to_numpy()
    
    rand_choice = np.random.choice(X.shape[0], int(X.shape[0] * .2))
    X_test = X[rand_choice]
    y_test = y[rand_choice]
    X_train = np.delete(X, rand_choice, axis=0)
    y_train = np.delete(y, rand_choice,axis=0)
   
    # X_train = np.array(X_train)
    # y_train = np.array(y_train)
    # X_test = np.array(X_test)
    # y_test = np.array(y_test)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

    return X_train, y_train, X_test, y_test