import wandb
from .Log import Log

class WBLog(Log):

    def __init__(self, test_name, test_id):
        super().__init__(test_name, test_id)
        self.table = None
        wandb.init(project="active learning playground 2",name=f"{test_name}-{test_id}",reinit=True,
        config = {
            "dataset": test_id, 
            "method": test_name
        })


    def log_hyper_parameters(self, **kwargs):
        
        if self.table is None:
            self.table = wandb.Table(columns=["hyper param","value"])
        
        for kwarg in kwargs:
            self.table.add_data(str(kwarg), str(kwargs[kwarg]))

    def log_training_data(self, **kwargs):
        self.flush_table()
    
        data = {}
        step_size = None
        for kwarg in kwargs:
            if kwarg == "labeled_data_points":
                step_size = kwargs[kwarg]
            else:
                data[kwarg] = kwargs[kwarg]
        wandb.log(data, step=step_size)

    def flush_table(self):
        
        if self.table is not None:
            wandb.log({"hyper-params":self.table})
            self.table = None

    def __del__(self):

        self.flush_table()

        
        
