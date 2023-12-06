from .WBLog import WBLog 
from .TextLog import TextLog
from .TerminalLog import TerminalLog


class Logger():

    test_id = None
    test_name = None
    logs = []

    # runs wandb logging
    wandb = False
    # runs text based logging into csv 
    # puts in logs folder
    text = True
    # prints logs to terminal
    terminal = True

    @staticmethod 
    def initialize_logger(test_name, test_id):
        Logger.test_id = test_id 
        Logger.test_name = test_name
        Logger.logs = []
        if Logger.wandb:
            Logger.logs.append(WBLog(test_name, test_id))
        if Logger.text:
            Logger.logs.append(TextLog(test_name, test_id))     
        if Logger.terminal:
            Logger.logs.append(TerminalLog(test_name, test_id))   
        
    
    @staticmethod 
    def log_hyper_parameters(**kwargs):
        
        for log in Logger.logs:
            log.log_hyper_parameters(**kwargs)

    @staticmethod 
    def log_training_data(**kwargs):

        for log in Logger.logs:
            log.log_training_data(**kwargs)