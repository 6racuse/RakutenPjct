def print_title():
    print("  _______          __               _                           _      _____      _______                   _               _    ")
    print(" |_   __ \        [  |  _          / |_                        / \    |_   _|    |_   __ \                 (_)             / |_  ")
    print("   | |__) |  ,--.  | | / ] __   _ `| |-'.---.  _ .--.         / _ \     | |        | |__) | .--.  .--.     __ .---.  .---.`| |-' ")
    print("   |  __ /  `'_\ : | '' < [  | | | | | / /__\\\[ `.-. |       / ___ \    | |        |  ___[ `/'`\] .'`\ \  [  / /__\\\/ /'`\\]| |   ")
    print("  _| |  \ \_// | |,| |`\ \ | \_/ |,| |,| \__., | | | |     _/ /   \ \_ _| |_      _| |_   | |   | \__. |_  | | \__.,| \__. | |,  ")
    print(" |____| |___\\'-;__[__|  \_]'.__.'_/\__/ '.__.'[___||__]   |____| |____|_____|    |_____| [___]   '.__.'[ \_| |'.__.''.___.'\__/  ")
    print("                                                                                                        \____/                   ")
    return 0
def print_choice():
    
    print("")
    print("Choisir un modèle à éxécuter : ")
    print("")
    print("     1 - Neural Network (f1-score 0.808) ")
    print("     2 - SVM (f1-score 0.8256) ")
    print("     3 - KNN (f1-score 0.7113)")
    print("     4 - RF  (f1-score 0,7920)")
    print("     5 - Solution to the project (f1-score 0.8118)")
    print("")
    submit = int(input("Choix : "))
    return submit
def main():
    filterwarnings("ignore")
    # clean_console()
       
    X_train,y_data,X_test = Preprocess_dataset()
    tfidf = TfidfVectorizer()
    
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
       
    print_title()
    submit = print_choice()
    clean_console()


    if (Model_Map.MODEL_NN.value==submit):
        from ZNN import train__,f1_m,predict_labels
        from sklearn.pipeline import Pipeline
        
        from sklearn.preprocessing import LabelEncoder
        
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_data)
        
        DoReload = input("Reload neural network model - mandatory if nn_model.keras doesn't exist - (yes/no) ? : ")
        
        if DoReload=='no' and path.exists('./models/nn_model.keras'):
            print("Loading pre-trained NN model...")
            best_model = tf.keras.models.load_model('./models/nn_model.keras', custom_objects={'f1_m': f1_m})
            
        elif DoReload=='yes':    
            nn_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('model', train__(X_train_tfidf, y_train_encoded))
            ])
            
            best_model = nn_pipeline.named_steps['model']
        else:
            return 0
        
        y_test_pred_nn = predict_labels(best_model, X_test_tfidf, label_encoder)
        Save_label_output(y_test_pred_nn,len(X_train),'./output/output_nn.csv')
        
    if (Model_Map.MODEL_SVM.value==int(submit)):
        from ZSVM import train_model
        from joblib import load,dump
        
        DoReload = input("Reload SVM model - mandatory if svm_model.joblib doesn't exist - (yes/no) ? : ")
        
        if DoReload=='no' and path.exists('./models/svm_model.joblib'):
            print("Loading pre-trained SVM model...")
            best_model = load('./models/svm_model.joblib')
            
        elif DoReload=='yes':    
            best_model = train_model(X_train_tfidf,y_data)
            dump(best_model,'./models/svm_model.joblib')
        else:
            return 0
        y_test_pred_svm = best_model.predict(X_test_tfidf)
        Save_label_output(y_test_pred_svm,len(X_train),'./output/output_svm.csv')

    if (Model_Map.MODEL_KNN.value == int(submit)):
        from ZKNN import train_knn
        from joblib import load, dump

        DoReload = input("Reload kNN model - mandatory if knn_model.joblib doesn't exist - (yes/no) ? : ")

        if DoReload == 'no' and path.exists('./models/knn_model.joblib'):
            print("Loading pre-trained kNN model...")
            best_model = load('./models/knn_model.joblib')

        elif DoReload == 'yes':
            best_model = train_knn(X_train_tfidf, y_data)
            dump(best_model, './models/knn_model.joblib')
        else:
            return 0
        y_test_pred_knn = best_model.predict(X_test_tfidf)
        Save_label_output(y_test_pred_knn, len(X_train), './output/output_knn.csv')
        
    if (Model_Map.MODEL_RF.value==int(submit)):
            from ZRF import train_rf
            from joblib import dump,load
            DoReload = input("Reload RF model - mandatory if rf_model.joblib doesn't exist - (yes/no) ? : ")
            
            if DoReload=='no' and path.exists('./models/rf_model.joblib'):
                print("Loading pre-trained RF model...")
                best_model = load('./models/rf_model.joblib')
                
            elif DoReload=='yes':    
                best_model = train_rf(X_train_tfidf,y_data)
                dump(best_model,'./models/rf_model.joblib')
            else:
                return 0
            y_pred_rf = best_model.predict(X_test_tfidf)
            Save_label_output(y_pred_rf, len(X_train), './output/output_rf.csv')
        
    if (Model_Map.MODEL_RESULTS.value == int(submit)):
        from joblib import load, dump
        from ZNN import train__, f1_m, predict_labels
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import LabelEncoder
        from ZSVM import train_model
        from joblib import load, dump
        from ZKNN import train_knn
        from joblib import load, dump
        from ZSolutionPjct import weighted_vote_prediction

        #NN
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_data)

        DoReload_nn = input("Reload neural network model - mandatory if nn_model.keras doesn't exist - (yes/no) ? : ")

        if DoReload_nn == 'no' and path.exists('./models/nn_model.keras'):
            print("Loading pre-trained NN model...")
            best_model = tf.keras.models.load_model('./models/nn_model.keras', custom_objects={'f1_m': f1_m})

        elif DoReload_nn == 'yes':
            nn_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('model', train__(X_train_tfidf, y_train_encoded))
            ])

            best_model = nn_pipeline.named_steps['model']
        else:
            return 0

        y_test_pred_nn = predict_labels(best_model, X_test_tfidf, label_encoder)
        Save_label_output(y_test_pred_nn, len(X_train), './output/output_nn.csv')

        #SVM
        DoReload_svm = input("Reload SVM model - mandatory if svm_model.joblib doesn't exist - (yes/no) ? : ")
        if DoReload_svm == 'no' and path.exists('./models/svm_model.joblib'):
            print("Loading pre-trained SVM model...")
            best_model = load('./models/svm_model.joblib')

        elif DoReload_svm == 'yes':
            best_model = train_model(X_train_tfidf, y_data)
            dump(best_model, './models/svm_model.joblib')
        else:
            return 0
        y_test_pred_svm = best_model.predict(X_test_tfidf)
        Save_label_output(y_test_pred_svm, len(X_train), './output/output_svm.csv')

        #kNN
        DoReload_kNN = input("Reload kNN model - mandatory if knn_model.joblib doesn't exist - (yes/no) ? : ")
        if DoReload_kNN == 'no' and path.exists('./models/knn_model.joblib'):
            print("Loading pre-trained kNN model...")
            best_model = load('./models/knn_model.joblib')

        elif DoReload_kNN == 'yes':
            best_model = train_knn(X_train_tfidf, y_data)
            dump(best_model, './models/knn_model.joblib')
        else:
            return 0

        y_test_pred_knn = best_model.predict(X_test_tfidf)
        Save_label_output(y_test_pred_knn, len(X_train), './output/output_knn.csv')

        #RF
        from ZRF import train_rf
        from joblib import dump,load
        DoReload = input("Reload RF model - mandatory if rf_model.joblib doesn't exist - (yes/no) ? : ")
        
        if DoReload=='no' and path.exists('./models/rf_model.joblib'):
            print("Loading pre-trained RF model...")
            best_model = load('./models/rf_model.joblib')
            
        elif DoReload=='yes':    
            best_model = train_rf(X_train_tfidf,y_data)
            dump(best_model,'./models/rf_model.joblib')
        else:
            return 0
        y_pred_rf = best_model.predict(X_test_tfidf)
        Save_label_output(y_pred_rf, len(X_train), './output/output_rf.csv')

        #détermination des meilleurs labels
        Save_label_output(
            weighted_vote_prediction(y_test_pred_nn, y_test_pred_svm, y_test_pred_knn),
            len(X_train),
            './output/output_weighted_vote.csv'
        )
        
    return 0




if __name__ == '__main__':
    from ZManageData import clean_console,Save_label_output
    from ZGlobal_parameter import Error_Map,Model_Map
    clean_console()
    from warnings import filterwarnings
    import tensorflow as tf
    from os import path
    from ZManageData import Preprocess_dataset
    from sklearn.feature_extraction.text import TfidfVectorizer
    main()