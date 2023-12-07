from sklearn.metrics import f1_score


def accuracy_metric(model, y_true, y_proba):

    y_preds = y_proba.argmax(axis=1)
    acc = (y_preds  == y_true).mean()
    return acc

def f1_metric(model, y_true, y_proba):
    
    y_preds = y_proba.argmax(axis=1)
    return f1_score(y_true,y_preds, average='micro')

def loss_metric(model, y_true, y_proba):

    return model.loss(y_true,y_proba)

def compute_metrics(metrics,dataset, model, X, y):

    preds = model.pred_proba(X)
    ret = {}
    for metric in metrics:
        ret[f"{dataset}/{metric}"] = metrics[metric](model, y, preds)
    return ret
