import numpy as np
from Cifar import get_data, get_data_cat_dog 
from Model import LeNetAL, ResNet18AL
from query_strat import uncertainty_pool, uncertainty_stream, query_by_comittee, default_random, default_increment
import wandb



def generate_test(name, model, query_strat):
    return {
        "name":name, 
        "model":model, 
        "query_strat":query_strat
    }

def main():
    
    wandb.init(project="active learning playground",name="gg2")

    X_train, y_train, X_test, y_test = get_data()
    models = [LeNetAL()]
    # models = [LeNetAL(), LeNetAL(), LeNetAL(), LeNetAL(), LeNetAL()]
    query_strat = uncertainty_pool 
    # query_strat = default_increment
    # query_strat = uncertainty_stream
    # query_strat = query_by_comittee
    train(models, X_train, y_train, query_strat, 500, 5000, 500, (X_test, y_test))

def test_all(ModelClass, data_func):
    for i in range(83,84):
        initial_weights = ModelClass()
        tests = [
          generate_test("uncert-pool",[ModelClass.clone(initial_weights)],uncertainty_pool),
          generate_test("uncert-stream",[ModelClass.clone(initial_weights)],uncertainty_stream),
          generate_test("baseline-increment",[ModelClass.clone(initial_weights)],default_increment),
          generate_test("baseline-random",[ModelClass.clone(initial_weights)],default_random),
          generate_test("query-by-committee",[ModelClass(), ModelClass(), ModelClass(), ModelClass(), ModelClass()],query_by_comittee),
          
        ]
        X_train, y_train, X_test, y_test = data_func()
        for test in tests:
            print(f"\n\n Running: {test['name']}\n\n")
            wandb.init(project="active learning playground",name=f"{test['name']}-{i}",reinit=True,
                    config = {
                        "method":test["name"],
                        "dataset": i
                    })
            train(test["model"],X_train, y_train, test["query_strat"],1000, 4200, 200, (X_test, y_test))





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
    # test_all(ResNet18AL, get_data_cat_dog)
    test_all(LeNetAL, get_data)
    # main()
    


