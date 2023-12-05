from .Models import LeNetAL 
from .Datasets import get_mnist_data
from .train import test_all_strats

def main():
    
    test_all_strats(LeNetAL, get_mnist_data, test_id=7)


if __name__ == "__main__":
    main()