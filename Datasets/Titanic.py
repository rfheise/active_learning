import pandas as pd
import os


def get_titanic_data():
    cdir =  os.path.dirname(os.path.realpath(__file__))
    train_f = os.path.join(cdir,"titanic","titanic_train.csv")
    test_f = os.path.join(cdir,"titanic","titanic_test.csv")
    train = pd.read_csv(train_f).sample(frac=1)
    test = pd.read_csv(test_f).sample(frac=1)
    y_train = train["Survived"].to_numpy()
    X_train = train.drop('Survived',axis=1).to_numpy()
    y_test = test["Survived"].to_numpy()
    X_test = test.drop('Survived',axis=1).to_numpy()
    return X_train, y_train, X_test, y_test
    