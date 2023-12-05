from .Models import LeNetAL 
from .Datasets import get_mnist_data


def main():
    
    test_all_strats(LeNetAL, data_func)


if __name__ == "__main__":
    main()