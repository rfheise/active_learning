from .Models import LeNetAL, XGBoost, LogisticAL,ResNet18AL, KNN_AL
from .Datasets import get_mnist_data,get_titanic_data, get_mnist_flat_data,get_cifar_data, get_knn_data 
from .train import test_all_strats
from .metrics import accuracy_metric, loss_metric, f1_metric
import numpy as np

def initializer(X,y, num_points):

        labeled_data = [X[:num_points], y[:num_points]]
        random_perm = np.arange(X.shape[0] - num_points) + num_points
        return labeled_data, X[random_perm], y[random_perm]

def main():
    
    # runs all querying strategies on mnist dataset with LeNet 
    # you must change the test id for each test or it will overwrite the previous logs 
    # uses 500 initial training points 
    # uses a budget of 10000
    # uses a step size of 500
    # to run your test change the args of the function below
    metrics = {
        "acc":accuracy_metric,
        "loss":loss_metric,
        "f1":f1_metric
    }
    
    # test_all_strats(LeNetAL, get_mnist_data, metrics=metrics,test_id=6045,num_init=500,budget=8050, k=250)
    # for i in range(4):
    test_all_strats(LogisticAL, get_titanic_data, metrics=metrics, test_id=6047,num_init=10, k=5,budget=1000)
    # test_all_strats(KNN_AL, get_knn_data,initializer=initializer, metrics=metrics, test_id=11003, num_init=10,budget=1000, k=5 )

if __name__ == "__main__":
    main()
