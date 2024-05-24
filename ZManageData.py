def clean_console():
    from  os import name,system
    if name == 'nt':  # Pour Windows
        system('cls')
    else:  # Pour Linux et macOS
        system('clear')









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


def Save_label_output(y_pred_labels,len_X_train):
    from csv import writer

    filename = '.\output\ylabel_nn.csv'
    # Écriture de la liste dans le fichier CSV
    with open(filename, mode='w', newline='') as file:
        writer = writer(file, lineterminator='\n')
        writer.writerow(['','prdtypecode'])
        k=0
        for value in y_pred_labels:
            writer.writerow([len_X_train+k,value])
            k+=1



def remove_html_tags(text):
    from bs4 import BeautifulSoup
    return BeautifulSoup(text, "html.parser").get_text()


def save_with_segmentation(data, filename_prefix, segment_size):
    import numpy as np
    from scipy.io import savemat

    num_segments = len(data) // segment_size + 1
    for i in range(num_segments):
        segment = data[i * segment_size:(i + 1) * segment_size]
        mdic = {"data": np.array(segment, dtype=str)}
        segment_filename = f"{filename_prefix}_segment_{i + 1}.mat"
        savemat(segment_filename, mdic)


def load_segmented(filename_prefix):
    data_combined = []
    segment_index = 0
    from scipy.io import loadmat
    from os.path import exists
    from numpy import array
    while True:
        segment_filename = f"{filename_prefix}_segment_{segment_index + 1}.mat"
        if not exists(segment_filename):
            break
        
        segment_data = loadmat(segment_filename)
        segment_data_flat = segment_data['data'].flatten()
        for item in segment_data_flat:
            if isinstance(item, str):
                data_combined.append(item)
            else:
                data_combined.append(str(item))  # Convertir les non-chaînes en chaînes de caractères
        
        segment_index += 1
    
    return array(data_combined)

def remove_html_tags(text):
    from bs4 import BeautifulSoup
    return BeautifulSoup(text, "html.parser").get_text()

def preprocess_text(text, language='french'):
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from unidecode import unidecode
    from string import punctuation
    text = remove_html_tags(text)
    text = text.lower()
    text = unidecode(text)
    tokens = word_tokenize(text, language=language)
    stop_words = set(stopwords.words(language))
    tokens = [word for word in tokens if word not in stop_words and word not in punctuation]
    return ' '.join(tokens)


def Preprocess_dataset(Force_Reload):

    from os.path import exists
    from pandas import read_csv,isna
    from scipy.io import loadmat
    from spacy import load
    from scipy.io import savemat
    from sys import getsizeof
    from numpy import where
    from numpy import array


    from nltk import download

    from nltk.tokenize import word_tokenize
    download("punkt")
    download('stopwords')
    filename_train_design = '.\data\X_train_design.mat'
    filename_train_descrip = '.\data\X_train_descrip'

    filename_test_design = ".\data\X_test_design.mat"
    filename_test_descrip = ".\data\X_test_descrip"

    input_filename_Xtrain = ".\data\X_train_update.csv"
    input_filename_Xtest = ".\data\X_test_update.csv"

    Must_Reload = not exists(filename_train_design) or not exists(filename_train_descrip+'_segment_1.mat')
   
    segment_sz = 10000

    if Must_Reload or Force_Reload:
        X_data_design,X_data_descrip = [],[]
        raw_data,y_data = Get_dataset()
        design = raw_data['designation']
        descrip = raw_data['description']

        nan_mask = descrip.isna()
        nan_indices = where(nan_mask)[0]

        nan_indices_list = nan_indices.tolist()
        for k in range(0,len(raw_data)):
            if k not in nan_indices_list:
                processed_text = preprocess_text(descrip[k], language='french')
                X_data_descrip.append(processed_text)
            else:
                X_data_descrip.append("")

            processed_text = preprocess_text(design[k], language='french')
            X_data_design.append(processed_text)


            # X_data_design.append(raw_to_tokens(design[k],spacy_nlp))
            progress_bar(k + 1,len(raw_data), prefix='Récupération X_train:', suffix='Complété', length=50)
        mdic1 = {"data": X_data_design}
        savemat(filename_train_design,mdic1)


        save_with_segmentation(X_data_descrip,filename_train_descrip,segment_sz)

    else:
        X_data_design = loadmat(filename_train_design)['data']
        X_data_descrip = load_segmented(filename_train_descrip)

        Y_train_filename = ".\data\Y_train_CVw08PX.csv"
        y_data = read_csv(Y_train_filename,index_col=0)


    Must_Reload = not exists(filename_test_design) or not exists(filename_test_descrip+'_segment_1.mat')


    if Must_Reload or Force_Reload:

        X_data_test_descrip,X_data_test_design = [],[]
        raw_data_test = read_csv(input_filename_Xtest,index_col=0)
        design_test = raw_data_test['designation']
        descrip_test = raw_data_test['description']

        nan_mask = descrip_test.isna()
        nan_indices = where(nan_mask)[0]

        nan_indices_list = nan_indices.tolist()
        #la boucle suivante prend bcp de temps
        for k in range(len(raw_data_test)):
            if k not in nan_indices_list:
                processed_text = preprocess_text(descrip[raw_data_test.axes[0][0]+k], language='french')
                X_data_test_descrip.append(processed_text)
                # X_data_test_descrip.append(raw_to_tokens(remove_html_tags(descrip_test[raw_data_test.axes[0][0]+k]),spacy_nlp))
            else:
                X_data_test_descrip.append("")
                
            processed_text = preprocess_text(design[raw_data_test.axes[0][0]+k], language='french')
            X_data_test_design.append(processed_text)
            # X_data_test_design.append(raw_to_tokens(design_test[raw_data_test.axes[0][0]+k],spacy_nlp))
            progress_bar(k + 1,len(raw_data_test), prefix='Récupération X_test: ', suffix='Complété', length=50)
        mdic1 = {"data": X_data_test_design}
        savemat(filename_test_design,mdic1)

        save_with_segmentation(X_data_test_descrip,filename_test_descrip,segment_sz)
    else:
        X_data_test_design = loadmat(filename_test_design)['data']
        X_data_test_descrip = load_segmented(filename_test_descrip)


    clean_console()
    return X_data_design,X_data_descrip,y_data,X_data_test_design,X_data_test_descrip



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
    from ZGlobal_parameter import Error_Map
    try:
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
    except:
        return Error_Map.TYPE_ERROR_INVALID_TYPE.value



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





