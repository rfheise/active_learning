import os
import csv
from .Log import Log

class TextLog(Log):

    def __init__(self, test_name, test_id, root="logs"):
        super().__init__(test_name, test_id)
        self.test_name = test_name
        self.test_id = test_id
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.root = os.path.join(dir_path, root, f"test{test_id}",test_name)
        self.init = False
        if not os.path.exists(self.root):
            os.makedirs(self.root)

    def log_hyper_parameters(self, **kwargs):
        
        fname = os.path.join(self.root, "hyper_params.txt")
        with open(fname, "a") as f:
            for kwarg in kwargs:
                f.write(f"{kwarg}:{kwargs[kwarg]}\n")

    def log_training_data(self,**kwargs):
        
        fname = os.path.join(self.root, "test_data.csv")
        self.f = open(fname, "a", newline="")
        csvwriter = csv.DictWriter(self.f, fieldnames=kwargs.keys())
        if self.init == False:
            csvwriter.writeheader()
            self.init = True
        csvwriter = csv.DictWriter(self.f, fieldnames=kwargs.keys())
        csvwriter.writerow(kwargs)
        self.f.close()
        self.f = None

        