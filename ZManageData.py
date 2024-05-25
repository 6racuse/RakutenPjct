def clean_console():
    import os
    if os.name == 'nt':  # Pour Windows
        os.system('cls')
    else:  # Pour Linux et macOS
        os.system('clear')
        
        







def Get_dataset():
    """récupère le jeu de données train

    Returns:
        x_data,y_data
    """
    from pandas import read_csv

    Xtrain_filename = ".\data\X_train_update.csv"   
    raw_data_train = read_csv(Xtrain_filename,index_col=0)
    
    Y_train_filename = ".\data\Y_train_CVw08PX.csv"
    y_data = read_csv(Y_train_filename,index_col=0)
    
    Xtest_filename = ".\data\X_test_update.csv"   
    raw_data_test = read_csv(Xtest_filename,index_col=0)
    
    return raw_data_train,y_data,raw_data_test
        

def Save_label_output(y_pred_labels,len_X_train,filename):
    from csv import writer

    
    # Écriture de la liste dans le fichier CSV
    with open(filename, mode='w', newline='') as file:
        writer = writer(file, lineterminator='\n')
        writer.writerow(['','prdtypecode'])
        k=0
        for value in y_pred_labels:
            writer.writerow([len_X_train+k,value])
            k+=1

def Preprocess_dataset():   
    from nltk import download
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from string import punctuation
    from numpy import ravel 
    
    download("punkt")
    download("stopwords")
    stop_words = set(stopwords.words('french'))
    
    raw_data_train,y_data,raw_data_test = Get_dataset()
  
    X_data_design_train = []
    design = raw_data_train['designation']

    for k in range(len(raw_data_train)):
        
        tokens = word_tokenize(normalize_accent(design[k].lower()),language='french')
        tokens = [word for word in tokens if word not in punctuation and word not in stop_words]
        X_data_design_train.append(tokens)
        
        progress_bar(k + 1,len(raw_data_train), prefix='Récupération X_train:', suffix='Complété', length=50)
    
    X_data_design_test = []
    design_test = raw_data_test['designation']
    
    for k in range(len(raw_data_test)):
        
        tokens = word_tokenize(normalize_accent(design_test[k+len(raw_data_train)].lower()),language='french')
        tokens = [word for word in tokens if word not in punctuation and word not in stop_words]
        X_data_design_test.append(tokens)
        
        progress_bar(k + 1,len(raw_data_test), prefix='Récupération X_test: ', suffix='Complété', length=50)
    
    
    X_data_design_train = [' '.join(tokens) for tokens in X_data_design_train]
    X_data_design_test = [' '.join(tokens) for tokens in X_data_design_test]
    
    clean_console()
    return X_data_design_train,y_data,X_data_design_test
 

def progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """Fonction d'affichage de progress bar, récupérée sur le Notebook du TD1 de Data Sciences de Myriam Tami
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()
                
def raw_to_tokens(raw_string, spacy_nlp):
        # Write code for lower-casing
    string = raw_string.lower()

    # Write code to normalize the accents
    string = normalize_accent(string)

    # Write code to tokenize
    spacy_tokens = spacy_nlp(string)
    

    # Write code to remove punctuation tokens and create string tokens
    string_tokens = [token.orth_ for token in spacy_tokens if not token.is_punct if not token.is_stop]
    # Write code to join the tokens back into a single string
    clean_string = " ".join(string_tokens)

    return clean_string


def normalize_accent(string):
    string = string.replace('á', 'a')
    string = string.replace('â', 'a')
    string = string.replace('à', 'a')
    
    string = string.replace('é', 'e')
    string = string.replace('è', 'e')
    string = string.replace('ê', 'e')
    string = string.replace('ë', 'e')

    string = string.replace('î', 'i')
    string = string.replace('ï', 'i')
    string = string.replace('n°','n')
    string = string.replace('ö', 'o')
    string = string.replace('ô', 'o')
    string = string.replace('ò', 'o')
    string = string.replace('ó', 'o')

    string = string.replace('ù', 'u')
    string = string.replace('û', 'u')
    string = string.replace('ü', 'u')

    string = string.replace('ç', 'c')

    return string


