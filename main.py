from .Models import LeNetAL 
from .Datasets import get_mnist_data
from .train import test_all_strats

def main():
    
    # runs all querying strategies on mnist dataset with LeNet 
    # you must change the test id for each test or it will overwrite the previous logs 
    # uses 500 initial training points 
    # uses a budget of 10000
    # uses a step size of 500
    # to run your test change the args of the function below
    test_all_strats(LeNetAL, get_mnist_data, test_id=7001,num_init=500, budget=5000, k=500)


if __name__ == "__main__":
    main()