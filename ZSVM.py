import numpy as np
import pandas as pd
import spacy
import string

import warnings
import time
import pickle
import os
import nltk

from ZManageData import normalize_accent
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from joblib import dump, load
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

def train_model(X_train_tfidf, Y_train):
    """
        This function trains a Support Vector Machine (SVM) model with the given parameters.

        Args:
            X_train_tfidf (sparse matrix, [n_samples, n_features]): The input data for training. Each row represents a document, and each column represents a feature.
            Y_train (array-like): The target values (class labels) for the training data.

        Returns:
            svm (SVC): The trained SVM model.
        """
    param_grid = {
        'C': [int(k) for k in np.linspace(1, 100, 100)],
        'gamma': [0.01, 0.1, 1],
        'kernel': 'rbf'
    }
    # Cross Validation
    grid_search = GridSearchCV(
        estimator=SVC(),
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=10
    )
    #grid_search.fit(X_train_tfidf, Y_train)
    
    svm = SVC(
        C=8.071428571428571,
        gamma=0.1,
        kernel='rbf'
    )
    
    #Training model
    svm.fit(
        X_train_tfidf,
        Y_train
    )
    return svm
def load_data(fast_coeff : int):
    """
        This function loads the training and testing data from CSV files, and returns the 'designation' column from the training and testing data, and the 'prdtypecode' column from the training data.

        Args:
            fast_coeff (int): A coefficient for reducing the size of the data. The data is divided by this coefficient.

        Returns:
            X_train (list): The 'designation' column from the training data.
            X_test_challenge (list): The 'designation' column from the testing data.
    """
    # Reading CSV files
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
    
    # Converting columns to lists
    X_train = X_train['designation'][:X_train.shape[0]//fast_coeff].tolist()
    X_test_challenge = X_test_challenge['designation'][:X_test_challenge.shape[0]//fast_coeff].tolist()

    Y_train = Y_train['prdtypecode'][:Y_train.shape[0]//fast_coeff].tolist()

    return X_train, X_test_challenge, Y_train
def tokenise_cleaning_data(X_train, X_test):
    """
        This function tokenizes and cleans the input data. It removes punctuation and French stopwords.

        Args:
            X_train (list): The training data to be tokenized and cleaned. It should be a list of strings.
            X_test (list): The testing data to be tokenized and cleaned. It should be a list of strings.

        Returns:
            X_train_clean (list): The tokenized and cleaned training data. Each string is a space-separated string of tokens.
            X_test_clean (list): The tokenized and cleaned testing data. Each string is a space-separated string of tokens.
    """

    nltk.download("punkt")
    nltk.download("stopwords")

    stop_words = set(stopwords.words('french'))

    X_train_clean = []
    X_test_clean = []

    a = len(X_train)
    b = len(X_test)
    
    # Tokenisation, lowering and removal of accents
    for k in range(a):
        tokens = word_tokenize(
            normalize_accent(
                X_train[k].lower()
            ),
            language='french'
        )
        tokens = [word for word in tokens if word not in string.punctuation and word not in stop_words]
        X_train_clean.append(tokens)

    for k in range(b):
        tokens = word_tokenize(
            normalize_accent(
                X_test[k].lower()
            ),
            language='french'
        )
        tokens = [word for word in tokens if word not in string.punctuation and word not in stop_words]
        X_test_clean.append(tokens)

    # Joining the tokens
    X_train_clean = [' '.join(tokens) for tokens in X_train_clean]
    X_test_clean = [' '.join(tokens) for tokens in X_test_clean]

    return X_train_clean, X_test_clean
def vectorize_data(X_train_clean, X_test_challenge_clean):
    """
        This function vectorizes the input data using TF-IDF Vectorizer.

        Args:
            X_train_clean (list): The cleaned training data to be vectorized. Each string is a space-separated string of tokens.
            X_test_challenge_clean (list): The cleaned testing data to be vectorized. Each string is a space-separated string of tokens.

        Returns:
            X_train_tfidf (sparse matrix, [n_samples, n_features]): Transformed training data. Each row represents a document, and each column represents a feature.
            X_test_tfidf (sparse matrix, [n_samples, n_features]): Transformed testing data. Each row represents a document, and each column represents a feature.
    """
    tfidf = TfidfVectorizer()

    X_train_tfidf = tfidf.fit_transform(X_train_clean)
    X_test_tfidf = tfidf.transform(X_test_challenge_clean)

    return X_train_tfidf, X_test_tfidf
