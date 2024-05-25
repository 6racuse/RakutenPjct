






        
        
def main():
    filterwarnings("ignore")
    clean_console()
    
    while True:
        clean_console()
        Reload_data = input("Reload dataset - mandatory if the entire folder hasn't been imported - (yes/no) : ")
        clean_console()
        print("Collecting dataset ...")
        
        if Reload_data=='yes' or Reload_data=='no':
            X_data,y_data,X_test = Preprocess_dataset(Reload_data=='yes')
            break
        else:
            print("wrong input")
        
    clean_console()
    print("Dataset collected, select a model :")
    print("     1 - Neural Network (f1-score 0.79) ")
    print("     2 - SVM (f1-score 0.82) ")
    print("")
    submit = int(input("Choice : "))
    clean_console()
    
    tfidf = TfidfVectorizer()
    
    X_train_tfidf = tfidf.fit_transform(X_data)
    X_test_tfidf = tfidf.transform(X_test)
        
    if (Model_Map.MODEL_NN.value==submit):
        from ZNN import train__,f1_m,predict_labels
        from sklearn.pipeline import Pipeline
        from os.path import exists
        from sklearn.preprocessing import LabelEncoder
        
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_data)
        
        DoReload = input("Reload neural network model - mandatory if best_model.keras doesn't exist - (yes/no) ? : ")
        
        if DoReload=='no' and exists('./models/nn_model.keras'):
            print("Loading pre-trained NN model...")
            best_model = tf.keras.models.load_model('./models/nn_model.keras', custom_objects={'f1_m': f1_m})
            
        elif DoReload=='yes':    
            nn_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('model', train__())
            ])
            
            nn_pipeline.fit(X_data, y_train_encoded)
            best_model = nn_pipeline.named_steps['model']
        else:
            return 0
        
        y_test_pred_nn = predict_labels(best_model, X_test_tfidf, label_encoder)
        Save_label_output(y_test_pred_nn,len(X_data),'output_nn.csv')
        
    if (Model_Map.MODEL_SVM.value==int(submit)):
        from ZSVM import train_model
        from joblib import load,dump
        
        
        DoReload = input("Reload SVM model - mandatory if svm_model.keras doesn't exist - (yes/no) ? : ")
        
        if DoReload=='no' and exists('./models/svm_model.joblib'):
            print("Loading pre-trained NN model...")
            best_model = load('./models/svm_model.joblib')
            
        elif DoReload=='yes':    
            best_model = train_model(X_train_tfidf,y_data)
            dump(best_model,'./models/svm_model.joblib')
        else:
            return 0
        y_test_pred_svm = best_model.predict(X_test_tfidf)
        Save_label_output(y_test_pred_svm,len(X_data),'output_svm.csv')


        
    return 0




if __name__ == '__main__':
    from ZManageData import clean_console,Save_label_output
    from ZGlobal_parameter import Error_Map,Model_Map
    clean_console()
    from warnings import filterwarnings
    import tensorflow as tf
    from ZManageData import Preprocess_dataset
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    main()