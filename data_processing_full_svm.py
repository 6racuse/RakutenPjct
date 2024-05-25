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

def tokenizer(text):
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

def pipeline_train_model(X_train, Y_train):
    params = {
        'C': 8.071428571428571,
        'gamma': 0.15,
        'kernel': 'rbf'
    }
    pipeline = Pipeline([
        (
            'preprocessor',
            TfidfVectorizer(
                preprocessor=tokenizer
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

    load_time_start = time.time()
    X_train, X_test_challenge, Y_train = load_data(fast_coeff)
    load_time_end = time.time()
    load_time_h, load_time_min, load_time_s, load_time_ms = header.convert_seconds(load_time_end - load_time_start)

    print(
        f"Data loaded in {int(load_time_h)}h {int(load_time_min)}min {int(load_time_s)}s {int(load_time_ms)}ms"
    )

    process_time_start = time.time()
    model = pipeline_train_model(
        X_train,
        Y_train
    )
    process_time_end = time.time()
    process_time_h, process_time_min, process_time_s, process_time_ms = header.convert_seconds(process_time_end - process_time_start)

    print(
        f"Data processed and model trained in {int(process_time_h)}h {int(process_time_min)}min {int(process_time_s)}s {int(process_time_ms)}ms"
    )

    Y_pred = model.predict(X_test_challenge)

    header.Save_label_output(
        Y_pred,
        len(X_train)
    )

    exec_time_end = time.time()
    exec_time_h, exec_time_min, exec_time_s, exec_time_ms = header.convert_seconds(exec_time_end - exec_time_start)

    print(
        f"Executed in {int(exec_time_h)}h {int(exec_time_min)}min {int(exec_time_s)}s {int(exec_time_ms)}ms"
    )

if __name__ == "__main__":
    main(fast_coeff=1)

