from .Models import LeNetAL, XGBoost, LogisticAL,ResNet18AL
from .Datasets import get_mnist_data,get_titanic_data, get_mnist_flat_data,get_cifar_data
from .train import test_all_strats

def main():
    
    # runs all querying strategies on mnist dataset with LeNet 
    # you must change the test id for each test or it will overwrite the previous logs 
    # uses 500 initial training points 
    # uses a budget of 10000
    # uses a step size of 500
    # to run your test change the args of the function below
    test_all_strats(LeNetAL, get_mnist_data, test_id=8021,num_init=500,budget=60000, k=500)
    # for i in range(4):
    # test_all_strats(LogisticAL, get_mnist_data, test_id=6090, num_init=500,budget=60000, k=500 )

if __name__ == "__main__":
    main()