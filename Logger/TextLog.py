import os
import csv

class TextLog():

    def __init__(self, test_name, test_id, root="logs"):
        self.test_name = test_name
        self.test_id = test_id
        self.root = os.path.join(root, f"test{test_id}",test_name)
        self.f = None
        if not os.path.exists(self.root):
            os.makedirs(self.root)

    def log_hyper_params(self, **kwargs):
        
        fname = os.path.join(self.root, "hyper_params.txt")
        with open(fname, "a") as f:
            for kwarg in kwargs:
                f.write(f"{kwarg}:{kwargs[kwarg]}\n")

    def log_training_data(self,**kwargs):
        
        if self.f is None:
            fname = os.path.join(self.root, "test_data.csv")
            self.f = open(fname, "a", newline="")
            csvwriter = csv.DictWriter(self.f, fieldnames=kwargs.keys())
            csvwriter.writeheader()
        csvwriter = csv.DictWriter(self.f, fieldnames=kwargs.keys())
        csvwriter.writerow(kwargs)

    def __del__(self):

        if self.f is not None:
            self.f.close()