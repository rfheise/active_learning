import numpy as np

def query_wrapper(query_strat):

    def query_new(num_points, models, unlabeled_data, y):

        if num_points >= len(unlabeled_data):
            return (unlabeled_data, y), np.array([]), np.array([]) 

        if len(models) == 1:
            return query_strat(num_points, models[0], unlabeled_data, y)
        else:
            return query_strat(num_points, models, unlabeled_data, y)
    
    return query_new

def ret_labels_from_indexes(unlabeled_data, y, index_list):

    new_labels = (unlabeled_data[index_list], y[index_list])
    
    return new_labels,\
        np.delete(unlabeled_data, index_list, axis=0),\
        np.delete(y, index_list, axis=0)

def get_entropy(proba):
    ent = -1 * proba * np.log(proba)
    ent = np.sum(ent, axis=1)
    return ent

@query_wrapper
def uncertainty_pool(num_points, model, unlabeled_data, y):

    preds = model.pred_proba(unlabeled_data)

    unlabeled_candidates = np.argsort(-1 * get_entropy(preds))[:num_points]

    return ret_labels_from_indexes(unlabeled_data, y, unlabeled_candidates)
   
@query_wrapper
def default_random(num_points, model, unlabeled_data, y):

    index_list = np.random.choice(np.arange(unlabeled_data.shape[0]), num_points, replace=False)
    
    return ret_labels_from_indexes(unlabeled_data, y, index_list)

@query_wrapper 
def default_increment(num_points, model, unlabeled_data, y):

    index_list = np.arange(unlabeled_data.shape[0])[:num_points]

    return ret_labels_from_indexes(unlabeled_data, y, index_list)

@query_wrapper
def uncertainty_stream(num_points, model, unlabeled_data, y, threshold=3):

    # gets tensor of probabilities of target class
    
    preds = model.pred_proba(unlabeled_data)

    # chooses the first <num_points> of classes below the threshold
    # could implement a different sampling strat later
    index_list = []
    entropy = get_entropy(preds)
    while len(index_list) < num_points:
        index_list = np.where(entropy > threshold)[0][:num_points]
        np.random.shuffle(index_list)
        threshold -= .01
        if threshold < 0:
            index_list = np.arange(len(entropy))[:num_points]
    
    # preds = np.max(preds, axis=1)
    # while len(index_list) < num_points:
    #     index_list = np.where(preds < threshold)[0][:num_points]
    #     threshold += .01

    # returns new labeled data & unlabeled sets from the list
    return ret_labels_from_indexes(unlabeled_data, y, index_list)

def vote_ent(c):

    def func(arr):
        votes = np.unique(arr, return_counts=True)[1]
        return (-1 * votes/c * np.log(votes/c)).sum(axis=0)
    return func

@query_wrapper
def query_by_comittee(num_points, models, unlabeled_data, y):

    preds = []
    for model in models:
        proba = model.pred_proba(unlabeled_data)
        preds.append(np.argmax(proba, axis=1))
    preds = np.array(preds)
    vote_entropy = np.apply_along_axis(vote_ent(len(models)),0,preds)
    index_list = np.argsort(-1 * vote_entropy)[:num_points]
    return ret_labels_from_indexes(unlabeled_data, y, index_list)

