



class Log():

    def __init__(self, test_name, test_id):
        self.test_name = test_name
        self.test_id = test_id

    def log_hyper_parameters(self, **kwargs):
        pass 

    def log_training_data(self,**kwargs):
        pass