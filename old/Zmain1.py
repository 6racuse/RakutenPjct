






        
        
def main():
    filterwarnings("ignore")
    clean_console()
    from ZTFIDFVectorization import Vectorize_matrix
    while True:
        clean_console()
        Reload_data = input("Reload dataset - mandatory if the entire folder hasn't been imported - (yes/no) : ")
        clean_console()
        print("Collecting dataset ...")
        
        if (Reload_data=='yes' or Reload_data=='no'):
            X_data_design,X_data_descrip,y_data,X_test_design,X_test_descrip = Preprocess_dataset(Reload_data=='yes')
            break;
        else:
            print("wrong input")

    
    
    clean_console()
    
    
    print("Dataset collected, select a model :")
    print("     1 - Neural Network (f1-score 0.79) ")
    print("     2 - SVM (f1-score 0.82) ")
    print("")
    submit = int(input("Choice : "))
    clean_console()
    
    X_tfidf = Vectorize_matrix(X_data_design)
    
    if (Model_Map.MODEL_NN.value==submit):
        from ZNN2 import LaunchNN_Model,Get_NN_Prediction
        from ZTFIDFVectorization import Vectorize_matrix
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from ZNNPIPLINE import CategoricalEncoder,KerasClassifier,MustRelloadNN
        
        
       
        NN_model = LaunchNN_Model(X_data_design,y_data)
        if (NN_model==Error_Map.TYPE_ERROR_INPUT):
            return 0
        y_test_pred = Get_NN_Prediction(NN_model,X_test_design)
        Save_label_output(y_test_pred,len(X_data_design))
                #=========================
                
        # X_tfidf = Vectorize_matrix(X_data_design)
        
        # label_encoder = CategoricalEncoder()
        # y_categorical = label_encoder.fit_transform(y_data)
        
        
        # Bypass_bool = input("Bypass learning process - this directly loads the best_model_nn from memory - (yes/no) : ")
        # if(not Bypass_bool or MustRelloadNN()):   
             
        #     pipeline = Pipeline([
        #     ('tfidf', TfidfVectorizer()),
        #     ('keras', KerasClassifier())
        #     ])
        #     pipeline.fit(X_data_design, y_categorical)
        #     dump(pipeline, 'pipeline.pkl')
            
        # else:
        #     pipeline = load('pipeline.pkl')
        #     pipeline.named_steps['keras'].model.load_weights('.\models\\best_model.keras')
            
            
        # y_pred = pipeline.predict(X_test_design)
        # Save_label_output(y_pred,len(X_data_design))       
        #================
    elif (Model_Map.MODEL_SVM.value==int(submit)):
        from ZSVM import LaunchSVM_ModelnGet_SVM_Prediction
        SVM_model = LaunchSVM_Model(X_data_design,y_data)
        if (NN_model==Error_Map.TYPE_ERROR_INPUT):
            return 0
        y_test_pred = Get_SVM_Prediction(SVM_model,X_test_design)
        Save_label_output(y_test_pred,len(X_data_design))
    else:
        print("No model chosen, Ending session ")
        
    return 0


if __name__ == '__main__':
    from ZManageData import clean_console,Save_label_output
    from ZGlobal_parameter import Error_Map,Model_Map
    from joblib import dump,load
    clean_console()
    from warnings import filterwarnings
    import tensorflow as tf
    from ZManageData import Preprocess_dataset
    
    
    main()