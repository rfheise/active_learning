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
    #data_path = os.path.join(dir_path, "Datasets")
    data_path = os.path.join(dir_path, "simple_2d")
    file_path = os.path.join(data_path, "knn_data8.csv")

    reader = pd.read_csv(file_path)
    X = reader.iloc[:,:-1]
    y = reader.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    
    """for i in range(len(x_train)):
        x_train[i] = np.array(x_train[i])
        y_train[i] = np.array(y_train[i])

    for j in range(len(x_test)):
        x_test[i] = np.array(x_test[i])
        y_test[i] = np.array(y_test[i])"""
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    #print(x_train)
    #print(f"Train length: {len(y_train)}, Test length: {len(y_test)}")

    return X_train, y_train, X_test, y_test