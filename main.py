from .Models import LeNetAL, XGBoost, LogisticAL,ResNet18AL
from .Datasets import get_mnist_data,get_titanic_data, get_mnist_flat_data,get_cifar_data
from .train import test_all_strats
from .metrics import accuracy_metric, loss_metric, f1_metric
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
    # test_all_strats(LeNetAL, get_mnist_data, test_id=9001,num_init=500,budget=8050, k=250)
    # for i in range(4):
    
    test_all_strats(LogisticAL, get_titanic_data, metrics=metrics, test_id=6018, num_init=10,budget=5000, k=5 )

if __name__ == "__main__":
    main()
