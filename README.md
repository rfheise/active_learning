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
    * Data.py 
      * helper functions and generic data function example
    * Cifar.py
      * cifar dataset 
    * MNIST.py 
      * MNIST dataset 
  * Models
    * LeNet.py
      * LeNet model implementation
    * ResNet.py
      * ResNet model implementation
    * Model.py
      * generic Model class used to implement a model
