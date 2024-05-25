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
    raw_data = read_csv(Xtrain_filename,index_col=0)
    
    Y_train_filename = ".\data\Y_train_CVw08PX.csv"
    y_data = read_csv(Y_train_filename,index_col=0)
    return raw_data,y_data
        

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



def Preprocess_dataset(Force_Reload):
    
    from os.path import exists
    from pandas import read_csv,isna
    from scipy.io import loadmat
    from spacy import load
    from scipy.io import savemat
    from sys import getsizeof
    from numpy import where
            
    mat_filename = '.\data\X_data.mat'
    Must_Reload = not exists(mat_filename)
    spacy_nlp = load("fr_core_news_sm")
           
    if Must_Reload or Force_Reload:
        X_data_design,X_data_descrip = [],[]
        raw_data,y_data = Get_dataset()
        design = raw_data['designation']
        # descrip = raw_data['description']
        
        # nan_mask = descrip.isna()
        # nan_indices = where(nan_mask)[0]

        # nan_indices_list = nan_indices.tolist()
        for k in range(len(raw_data)):
            # if k not in nan_indices_list:
            #     X_data_descrip.append(raw_to_tokens(descrip[k],spacy_nlp))
            # else:
            #     X_data_descrip.append("")
                
            X_data_design.append(raw_to_tokens(design[k],spacy_nlp))
            progress_bar(k + 1,len(raw_data), prefix='Récupération X_train:', suffix='Complété', length=50)
        mdic = {"data": X_data_design}
        savemat(mat_filename,mdic)
    else: 
        X_data_design = loadmat(mat_filename)['data']
        Y_train_filename = ".\data\Y_train_CVw08PX.csv"
        y_data = read_csv(Y_train_filename,index_col=0)
        
    mat_filename_t = '.\data\X_test.mat'
    Must_Reload = not exists(mat_filename_t)
    
    
    if Must_Reload or Force_Reload:
        Xtest_filename = ".\data\X_test_update.csv"   
        X_data_test = []
        raw_data_test = read_csv(Xtest_filename,index_col=0)
        design_test = raw_data_test['designation']
        #la boucle suivante prend bcp de temps
        for k in range(len(raw_data_test)):
            X_data_test.append(raw_to_tokens(design_test[len(X_data_design)+k],spacy_nlp))
            progress_bar(k + 1,len(raw_data_test), prefix='Récupération X_test:', suffix='Complété', length=50)
        mdic = {"data": X_data_test}
        savemat(mat_filename_t,mdic)       
    else:
        X_data_test = loadmat(mat_filename_t)['data']
        
        
    clean_console()
    return X_data_design,y_data,X_data_test
    


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


