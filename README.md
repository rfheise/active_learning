# Before you begin #
  ## Must Reads ##
  *  Model.py in Models
      * generic Model class used to implement a model
  * Logger.py in Logger
      * Used to log stuff
      * Not that important but just know to set bools to True/False to turn of that specific logger
  * Data.py in Datasets
      * Example of generic dataset function
  * main.py
    * main file where you will run your tests

  To create your own model create a new file in the Models folder please see LeNet.py for an example. To create your own dataset implement a generic data function please see MNIST.py for an example. Whenever adding a file remeber to update the README and __init__.py file in each folder when applicable. I implemented it as a python module so you have to import and run everything as a part of the python module i.e.
  `python3 -m <dir>.main`


# File Break Down
  * requirements.txt 
    * python dependencies
  * main.py
    * main file where you run your tests
  * plot_test.py
    * plots data from the csv files generated from the text based logger
  * metrics.py
    * file that allows you to produce custom data metrics to be logged by the logger
  * playground_ryan.py
    * file where I was just messing with things feel free to make your own 
  * query_strat.py 
    * file where I implemented all of the query strategies 
  * train.py
    * file where all of the training logic is implemented 
  * Logger
    * Log.py 
      * generic logging class to create a logging system 
    * WBLog.py
      * wandb logger 
    * TerminalLog.py
      * terminal logger 
    * TextLog.py
      * text logger 
    * Logger.py 
      * calls all set loggers when training 
  * Datasets
    * simple_2d
      * gen_data.py
        * generates data for knn test
    * titanic
      * contains pre-processed files of titanic dataset data
    * Data.py 
      * helper functions and generic data function example
    * Cifar.py
      * cifar dataset 
    * MNIST.py 
      * MNIST dataset
    * KNN.py
      * KNN dataset
    * Titanic.py
      * Titanic dataset
  * Models
    * LeNet.py
      * CNN model implementation using pytorch
    * LogisticRegression.py
      * Logisitc Regression model implemenation with sklearn
    * XGBoost.py
      * XGBoost model implementation using xgboost python library
    * KNN.py
    *   KNN model implementation using sklearn
    * ResNet.py
      * ResNet model implementation using pytorch
    * Model.py
      * generic Model class used to implement a model
