from enum import Enum,auto

class Error_Map(Enum):
    """
        Enumeration for various error types in the project.

        Attributes:
            TYPE_ERROR_INPUT (int): Error type for wrong input.
            TYPE_ERROR_INVALID_TYPE (int): Error type for invalid input type.
    """
    TYPE_ERROR_INPUT            = auto() #wrong input
    TYPE_ERROR_INVALID_TYPE     = auto() #unable to lower smth different from string
    
    
class Model_Map(Enum):
    """
        Enumeration for different models used in the project.

        Attributes:
            MODEL_NN (int): Neural Network model.
            MODEL_SVM (int): Support Vector Machine model.
            MODEL_KNN (int): K-Nearest Neighbors model.
            MODEL_RF (int): Random Forest model.
            MODEL_RESULTS (int): Solution to the project combining multiple models.
    """
    MODEL_NN   =    auto()  #Neural Network
    MODEL_SVM  =    auto()  #SVM model
    MODEL_KNN  =    auto()  #KNN model
    MODEL_RF   =    auto()
    MODEL_RESULTS = auto() #Solution to the project
    
