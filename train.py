import numpy as np
from .Datasets import get_mnist_data
from .Models import LeNetAL, ResNet18AL
from .query_strat import uncertainty_pool, uncertainty_stream, query_by_comittee, default_random, default_increment
from .Logger import Logger



def generate_test(name, model, query_strat):

    return {
        "name":name, 
        "model":model, 
        "query_strat":query_strat
    }

def test_all_strats(ModelClass, data_func,test_id=1, num_init=500, budget=10000, k=500):
    initial_weights = ModelClass()
    tests = [
        generate_test("uncert-pool",[ModelClass.clone(initial_weights)],uncertainty_pool),
        generate_test("baseline-random",[ModelClass.clone(initial_weights)],default_random),
        generate_test("uncert-stream",[ModelClass.clone(initial_weights)],uncertainty_stream),
        generate_test("baseline-increment",[ModelClass.clone(initial_weights)],default_increment),
        generate_test("query-by-committee",[ModelClass(), ModelClass(), ModelClass(), ModelClass(), ModelClass()],query_by_comittee),
        
    ]
    X_train, y_train, X_test, y_test = data_func()
    for test in tests:
        print(f"\n\n Running: {test['name']}\n\n")
        Logger.initialize_logger(test['name'], test_id)
        train_query_samp(test["model"],X_train, y_train, test["query_strat"],num_init, budget, k, (X_test, y_test))

def test_model(models, X,y, val):
    train_query_samp(models, X,y, default_random, initialize=10e8, budget=1, step_size=10, val=val)


def train_query_samp(models, X,y, query_strat, initialize=100, budget=100, step_size=10, val=None):
    
    #initialize starting labels
    labeled_data = [X[:initialize], y[:initialize]]
    unlabeled_data = X[initialize:]
    y = y[initialize:]
    
    #log hyper params
    for model in models:
        model.log_hyper_parameters()

    Logger.log_hyper_parameters(budget=budget, k=step_size, init_pts=initialize)

    for i in range(0,budget,step_size):

        train_acc, train_loss, val_acc, val_loss = (0,0,0,0)
        for model in models:
            # model.clear()
            train_acc, train_loss = model.fit(labeled_data[0], labeled_data[1])
            
        if val is not None:
            val_acc, val_loss = model.get_stat_val(val[0], val[1])

        Logger.log_training_data(
            **{"train/acc":train_acc,
            "train/loss":train_loss, 
            "val/loss":val_loss,
            "val/acc":val_acc,
            "labeled_data_points":labeled_data[0].shape[0]
            }
        )
        
        if unlabeled_data.shape[0] == 0:
            break

        labeled_data_new, unlabeled_data, y = query_strat(step_size, models, unlabeled_data, y)
        labeled_data[0] = np.append(labeled_data[0],labeled_data_new[0], axis=0)
        labeled_data[1] = np.append(labeled_data[1],labeled_data_new[1], axis=0)


    return model


