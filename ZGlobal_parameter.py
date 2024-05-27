from enum import Enum,auto

class Error_Map(Enum):
    TYPE_ERROR_INPUT            = auto() #wrong input
    TYPE_ERROR_INVALID_TYPE     = auto() #unable to lower smth different from string
    
    
class Model_Map(Enum):
    MODEL_NN   =    auto()  #Neural Network
    MODEL_SVM  =    auto()  #SVM model
    MODEL_KNN  =    auto()  #KNN model
    MODEL_RF   =    auto()
    MODEL_RESULTS = auto() #Solution to the project
    
