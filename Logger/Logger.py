from .WBLog import WBLog 
from .TextLog import TextLog



class Logger():

    test_id = None
    test_name = None
    logs = []
    wandb = False 
    text = True

    @staticmethod 
    def initialize_logger(test_name, test_id):
        Logger.test_id = test_id 
        Logger.test_name = test_name

        if Logger.wandb:
            Logger.logs.append(WBLog(test_id))
        if Logger.text:
            Logger.logs.append(TextLog(test_id))        
        
    
    @staticmethod 
    def log_hyper_parameters(**kwargs):
        
        for log in Logger.logs:
            log.log_hyper_parameters(**kwargs)

    @staticmethod 
    def log_training_data(**kwargs):

        for log in Logger.logs:
            log.log_training_data(**kwargs)