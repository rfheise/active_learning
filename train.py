import numpy as np
from Cifar import get_data 
from Model import LeNetAL
from query_strat import uncertainty_pool, uncertainty_stream, query_by_comittee, default_random, default_increment
import wandb



def generate_test(name, model, query_strat):
    return {
        "name":name, 
        "model":model, 
        "query_strat":query_strat
    }

def main():
    
    # wandb.init(project="active learning")

    X_train, y_train, X_test, y_test = get_data()
    models = [LeNetAL()]
    # models = [LeNetAL(), LeNetAL(), LeNetAL(), LeNetAL(), LeNetAL()]
    # query_strat = uncertainty_pool 
    # query_strat = default_increment
    query_strat = uncertainty_stream
    # query_strat = query_by_comittee
    train(models, X_train, y_train, query_strat, 500, 9500, 500)

def test_all():
    for i in range(1,5):
        initial_weights = LeNetAL()
        tests = [
          generate_test("uncert-pool",[LeNetAL.clone(initial_weights)],uncertainty_pool),
          generate_test("uncert-stream",[LeNetAL.clone(initial_weights)],uncertainty_stream),
          generate_test("query-by-committee",[LeNetAL(), LeNetAL(), LeNetAL(), LeNetAL(), LeNetAL()],query_by_comittee),
          generate_test("baseline-increment",[LeNetAL.clone(initial_weights)],default_increment),
          generate_test("baseline-random",[LeNetAL.clone(initial_weights)],default_random),
        ]
        X_train, y_train, X_test, y_test = get_data()
        for test in tests:
            print(f"\n\n Running: {test['name']}\n\n")
            wandb.init(project="active learning playground",name=f"{test['name']}-{i}",reinit=True,
                    config = {
                        "method":test["name"],
                        "dataset": i
                    })
            train(test["model"],X_train, y_train, test["query_strat"],100, 5100, 100, (X_test, y_test))






def train(models, X,y, query_strat, initialize=100, budget=100, step_size=10, val=None):
    
    #initialize starting labels
    labeled_data = [X[:initialize], y[:initialize]]
    unlabeled_data = X[initialize:]
    y = y[initialize:]

    for i in range(0,budget,step_size):

        train_acc, train_loss, val_acc, val_loss = (0,0,0,0)
        for model in models:
            model.clear()
            train_acc, train_loss = model.fit(labeled_data[0], labeled_data[1])
            
        if val is not None:
            val_acc, val_loss = model.get_stat_val(val[0], val[1])

        wandb.log(
            {"train/acc":train_acc,
                "train/loss":train_loss, 
                "val/loss":val_loss,
                "val/acc":val_acc},
                step=labeled_data[0].shape[0]
        )

        print(f"\n\n-------------------\nLabeled Data Points {labeled_data[0].shape[0]}\n-------------------\n\n")
        print(f"train acc: {train_acc}  train loss: {train_loss}")
        print(f"val loss: {val_acc}  val loss: {val_loss}")
        print(f"Unlabeled Data Points:{unlabeled_data.shape[0]}")
        
        if unlabeled_data.shape[0] == 0:
            break

        labeled_data_new, unlabeled_data, y = query_strat(step_size, models, unlabeled_data, y)
        labeled_data[0] = np.append(labeled_data[0],labeled_data_new[0], axis=0)
        labeled_data[1] = np.append(labeled_data[1],labeled_data_new[1], axis=0)


    return model

if __name__ == "__main__":
    test_all()
    


