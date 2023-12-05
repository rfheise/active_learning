import wandb


class WBLog():

    def __init__(self, test_name, test_id):
        super().__init__(test_id)
        self.table = None
        wandb.init(project="active learning playground",name=f"{test_name}-{test_id}",reinit=True,
        config = {
            "dataset": test_id, 
            "method": test_name
        })


    def log_hyper_parameters(self, **kwargs):
        
        if self.table is None:
            self.wandb_table = wandb.Table(columns=["hyper param","value"])
        
        for kwarg in kwargs:
            self.wandb_table.add_data(kwarg, kwargs[kwarg])

    def log_training_data(self, **kwargs):
        self.flush_table()
        wandb.log(kwargs)

    def flush_table(self):
        
        if self.table is not None:
            wandb.log({"hyper-params":self.table})
            self.table = None

    def __del__(self):

        self.flush_table()

        
        
