from enum import Enum,auto

class Error_Map(Enum):
    TYPE_ERROR_INPUT = auto() #wrong input
    
    
class Model_Map(Enum):
    MODEL_NN   =   auto()  #Neural Network
    MODEL_SVM  =   auto()  #SVM model
    
