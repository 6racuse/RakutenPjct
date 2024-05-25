import header

import string
import warnings
import time
import pickle
import os

import numpy as np
import pandas as pd
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

def download_nltk_data():
    if not nltk.data.find('tokenizers/punkt'):
        nltk.download("punkt")
    if not nltk.data.find('corpora/stopwords'):
        nltk.download("stopwords")

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

    X_train = X_train['designation'][:X_train.shape[0] // fast_coeff].tolist()
    X_test_challenge = X_test_challenge['designation'][:X_test_challenge.shape[0] // fast_coeff].tolist()

    Y_train = Y_train['prdtypecode'][:Y_train.shape[0] // fast_coeff].tolist()

    return X_train, X_test_challenge, Y_train

def preprocess_text(text):
    download_nltk_data()

    stop_words = set(
        stopwords.words('french')
    )

    tokens = word_tokenize(
        header.normalize_accent(
            text.lower()
        ),
        language='french'
    )
    tokens = [word for word in tokens if word not in string.punctuation and word not in stop_words]

    return ' '.join(tokens)

def train_model(X_train, Y_train):
    params = {
        'C': 8.071428571428571,
        'gamma': 0.1,
        'kernel': 'rbf'
    }
    pipeline = Pipeline([
        (
            'preprocessor',
            TfidfVectorizer(
                preprocessor=preprocess_text
            )
        ),
        (
            'classifier',
            SVC(
                C=params['C'],
                gamma=params['gamma'],
                kernel=params['kernel']
            )
        )
    ])

    pipeline.fit(
        X_train,
        Y_train
    )

    return pipeline

def main(fast_coeff : int):
    exec_time_start = time.time()
    warnings.filterwarnings("ignore")

    X_train, X_test_challenge, Y_train = load_data(fast_coeff)

    model = train_model(
        X_train,
        Y_train
    )

    Y_pred = model.predict(X_test_challenge)

    header.Save_label_output(
        Y_pred,
        len(X_train)
    )

    exec_time_end = time.time()
    exec_time_h, exec_time_min, exec_time_s = header.convert_seconds(exec_time_end - exec_time_start)

    print(
        f"Executed in {int(exec_time_h)}h {int(exec_time_min)}min {int(exec_time_s)}s"
    )

if __name__ == "__main__":
    main(fast_coeff=1)
