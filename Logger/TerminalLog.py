from .Log import Log



class TerminalLog(Log):

    def __init__(self, test_name, test_id):
        super().__init__(test_name, test_id)
        self.test_name = test_name
        self.test_id = test_id
        print(test_name, test_id)

    def log_hyper_parameters(self, **kwargs):
        pass 

    def log_training_data(self,**kwargs):
        
        data = {}
        step_size = None
        for kwarg in kwargs:
            if kwarg == "labeled_data_points":
                step_size = kwargs[kwarg]
            else:
                data[kwarg] = kwargs[kwarg]

        print(f"\n\n-------------------\nLabeled Data Points {step_size}\n-------------------\n\n")
        for d in data:
            print(f"{d}:{data[d]}")