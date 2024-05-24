import header

import numpy as np
import pandas as pd
import spacy

import warnings
import time
import pickle
import os

from joblib import dump, load
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
def load_data(fast_coeff : int):
    X_train = pd.read_csv(
        "/Users/welto/Library/CloudStorage/OneDrive-CentraleSupelec/2A/CASA/RakutenPjct/data/X_train_update.csv",
        sep=',',
        usecols=lambda column: column not in [
            'Unnamed: 0',
            'imageid',
            'description'
        ]
    )
    X_test_challenge = pd.read_csv(
        "/Users/welto/Library/CloudStorage/OneDrive-CentraleSupelec/2A/CASA/RakutenPjct/data/X_test_update.csv",
        sep=',',
        usecols=lambda column: column not in [
            'Unnamed: 0',
            'imageid',
            'description'
        ]
    )
    Y_train = pd.read_csv(
        "/Users/welto/Library/CloudStorage/OneDrive-CentraleSupelec/2A/CASA/RakutenPjct/data/Y_train_CVw08PX.csv",
        sep=',',
        usecols=lambda column: column != 'Unnamed: 0'
    )

    X_train = X_train['designation'][:X_train.shape[0]//fast_coeff].tolist()
    X_test_challenge = X_test_challenge['designation'][:X_test_challenge.shape[0]//fast_coeff].tolist()

    Y_train = Y_train['prdtypecode'][:Y_train.shape[0]//fast_coeff].tolist()

    return X_train, X_test_challenge, Y_train
def tokenise_cleaning_data(X_train, X_test_challenge, train_filename, test_challenge_filename):
    if os.path.exists(train_filename) and os.path.exists(test_challenge_filename):
        return load_tokenized_data(train_filename, test_challenge_filename)

    spacy_nlp = spacy.load("fr_core_news_sm")

    X_train_clean = []
    X_test_challenge_clean = []

    a = len(X_train)
    b = len(X_test_challenge)

    header.progress_bar(
        0,
        a,
        prefix='Progress:',
        suffix='Complete',
        length=50
    )

    preprocessing_start_time = time.time()

    for k in range(a):
        X_train_clean.append(
            header.raw_to_tokens(
                X_train[k],
                spacy_nlp
            )
        )
        header.progress_bar(
            k + 1,
            a,
            prefix='X_train_clean :',
            suffix='Complete',
            length=50
        )

    for k in range(b):
        X_test_challenge_clean.append(
            header.raw_to_tokens(
                X_test_challenge[k],
                spacy_nlp
            )
        )
        header.progress_bar(
            k + 1,
            b,
            prefix='X_test_challenge_clean :',
            suffix='Complete',
            length=50
        )

    preprocessing_end_time = time.time()
    preprocessing_time_h, preprocessing_time_min, preprocessing_time_s = header.convert_seconds(
        preprocessing_end_time - preprocessing_start_time)

    print(f"Preprocessed in {int(preprocessing_time_h)}h {int(preprocessing_time_min)}min {int(preprocessing_time_s)}s")

    save_tokenized_data(X_train_clean, X_test_challenge_clean, train_filename, test_challenge_filename)

    return X_train_clean, X_test_challenge_clean
def vectorize_data(X_train_clean, X_test_challenge_clean):
    tfidf = TfidfVectorizer()

    X_train_tfidf = tfidf.fit_transform(X_train_clean)
    X_test_challenge_tfidf = tfidf.transform(X_test_challenge_clean)

    return X_train_tfidf, X_test_challenge_tfidf
def save_tokenized_data(X_train_clean, X_test_challenge_clean, train_filename, test_challenge_filename):
    with open(
            train_filename,
            'wb'
    ) as f:
        pickle.dump(
            X_train_clean,
            f
        )

    with open(
            test_challenge_filename,
            'wb'
    ) as f:
        pickle.dump(
            X_test_challenge_clean,
            f
        )
def load_tokenized_data(train_filename, test_challenge_filename):
    with open(
            train_filename,
            'rb'
    ) as f:
        X_train_clean = pickle.load(f)

    with open(
            test_challenge_filename,
            'rb'
    ) as f:
        X_test_challenge_clean = pickle.load(f)
    return X_train_clean, X_test_challenge_clean
def train_model(X_train_tfidf, Y_train):
    param_grid = {
        'C': 8.916666666666668,
        'gamma': 0.1,
        'kernel': 'rbf'
    }
    svm = SVC(
        C=param_grid['C'],
        gamma=param_grid['gamma'],
        kernel=param_grid['kernel']
    )
    svm.fit(
        X_train_tfidf,
        Y_train
    )
    return svm

def main(fast_coeff : int):
    exec_time_start = time.time()
    warnings.filterwarnings("ignore")

    train_filename = 'X_train_clean.pkl'
    test_challenge_filename = 'X_test_challenge_clean.pkl'

    X_train, X_test_challenge, Y_train = load_data(fast_coeff)

    X_train_clean, X_test_challenge_clean = tokenise_cleaning_data(
        X_train,
        X_test_challenge,
        train_filename,
        test_challenge_filename
    )

    X_train_tfidf, X_test_challenge_tfidf = vectorize_data(
        X_train_clean,
        X_test_challenge_clean
    )

    model = train_model(
        X_train_tfidf,
        Y_train
    )

    Y_pred_challenge = model.predict(X_test_challenge_tfidf)

    header.Save_label_output(Y_pred_challenge, len(X_train_clean))

    exec_time_end = time.time()
    exec_time_h, exec_time_min, exec_time_s = header.convert_seconds(exec_time_end - exec_time_start)
    print(f"Executed in {int(exec_time_h)}h {int(exec_time_min)}min {int(exec_time_s)}s")

if __name__ == "__main__":
    main(fast_coeff=1)


