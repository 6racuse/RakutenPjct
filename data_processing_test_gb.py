import header

import numpy as np
import pandas as pd
import spacy

import warnings
import time
import pickle
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder


def load_data(fast_coeff : int, random_state, test_size : float):
    X_train = pd.read_csv(
        "/Users/welto/Library/CloudStorage/OneDrive-CentraleSupelec/2A/CASA/RakutenPjct/data/X_train_update.csv",
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
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_train,
        Y_train,
        test_size=test_size,
        random_state=random_state
    )
    X_train = X_train['designation'][:X_train.shape[0]//fast_coeff].tolist()
    X_test = X_test['designation'][:X_test.shape[0]//fast_coeff].tolist()

    Y_train = Y_train['prdtypecode'][:Y_train.shape[0]//fast_coeff].tolist()
    Y_test = Y_test['prdtypecode'][:Y_test.shape[0]//fast_coeff].tolist()

    label_encoder = LabelEncoder()

    Y_train = label_encoder.fit_transform(Y_train)
    Y_test = label_encoder.transform(Y_test)

    return X_train, X_test, Y_train, Y_test, label_encoder

def tokenise_cleaning_data(X_train, X_test, train_filename, test_filename):
    if os.path.exists(train_filename) and os.path.exists(test_filename):
        return load_tokenized_data(train_filename, test_filename)

    X_train_clean = []
    X_test_clean = []

    a = len(X_train)
    b = len(X_test)

    header.progress_bar(
        0,
        a,
        prefix='Progress:',
        suffix='Complete',
        length=50
    )

    preprocessing_start_time = time.time()

    for k in range(a):
        tokens = word_tokenize(X_train[k], language='french')
        X_train_clean.append(tokens)
        header.progress_bar(
            k + 1,
            a,
            prefix='X_train_raw_designation_clean :',
            suffix='Complete',
            length=50
        )

    for k in range(b):
        tokens = word_tokenize(X_test[k], language='french')
        X_test_clean.append(tokens)
        header.progress_bar(
            k + 1,
            b,
            prefix='X_test_raw_designation_clean :',
            suffix='Complete',
            length=50
        )

    preprocessing_end_time = time.time()
    preprocessing_time_h, preprocessing_time_min, preprocessing_time_s = header.convert_seconds(
        preprocessing_end_time - preprocessing_start_time)

    print(f"Preprocessed in {int(preprocessing_time_h)}h {int(preprocessing_time_min)}min {int(preprocessing_time_s)}s")

    save_tokenized_data(X_train_clean, X_test_clean, train_filename, test_filename)

    return X_train_clean, X_test_clean

def vectorize_data(X_train_clean, X_test_clean):
    tfidf = TfidfVectorizer()

    # Join tokens back into strings
    X_train_strings = [' '.join(tokens) for tokens in X_train_clean]
    X_test_strings = [' '.join(tokens) for tokens in X_test_clean]

    X_train_tfidf = tfidf.fit_transform(X_train_strings)
    X_test_tfidf = tfidf.transform(X_test_strings)

    return X_train_tfidf, X_test_tfidf

def save_tokenized_data(X_train_clean, X_test_clean, train_filename, test_filename):
    with open(
            train_filename,
            'wb'
    ) as f:
        pickle.dump(
            X_train_clean,
            f
        )
    with open(
            test_filename,
            'wb'
    ) as f:
        pickle.dump(
            X_test_clean,
            f
        )
def load_tokenized_data(train_filename, test_filename):
    with open(
            train_filename,
            'rb'
    ) as f:
        X_train_clean = pickle.load(f)
    with open(
            test_filename,
            'rb'
    ) as f:
        X_test_clean = pickle.load(f)
    return X_train_clean, X_test_clean

def train_model(X_train_tfidf, Y_train):
    """param_grid = {
        'n_estimators': 200,
        'learning_rate': 0.1,
        'max_depth': 7
    }"""

    xgb = XGBClassifier()

    xgb.fit(
        X_train_tfidf,
        Y_train
    )

    """best_params = grid_search.best_params_
    print(
        "Best params:",
        best_params
    )"""

    """model = XGBClassifier(**best_params)
    model.fit(
        X_train_tfidf,
        Y_train
    )"""

    return xgb


def evaluate_model(model, X_test_tfidf, Y_test, label_encoder):
    Y_pred_rf = label_encoder.inverse_transform(
        model.predict(X_test_tfidf)
    )

    f1 = f1_score(
        Y_test,
        Y_pred_rf,
        average='macro'
    )

    accuracy = accuracy_score(
        Y_test,
        Y_pred_rf
    )

    print(
        "Classification Report :",
        classification_report(
            Y_test,
            Y_pred_rf
        )
    )

    return f1, accuracy, Y_pred_rf

def main(fast_coeff: int, random_state: int, test_size: float):

    exec_time_start = time.time()
    warnings.filterwarnings("ignore")

    train_filename = 'X_train_clean.pkl'
    test_filename = 'X_test_clean.pkl'

    X_train, X_test, Y_train, Y_test, label_encoder = load_data(
        fast_coeff,
        random_state,
        test_size
    )

    X_train_clean, X_test_clean = tokenise_cleaning_data(
        X_train,
        X_test,
        train_filename,
        test_filename
    )

    X_train_tfidf, X_test_tfidf = vectorize_data(
        X_train_clean,
        X_test_clean
    )

    model = train_model(
        X_train_tfidf,
        Y_train
    )

    Y_test = label_encoder.inverse_transform(Y_test)

    f1, accuracy, Y_pred_rf = evaluate_model(
        model,
        X_test_tfidf,
        Y_test,
        label_encoder
    )

    print("F1 Score:", f1)
    print("Accuracy Score:", accuracy)

    exec_time_end = time.time()
    exec_time_h, exec_time_min, exec_time_s = header.convert_seconds(exec_time_end - exec_time_start)
    print(f"Executed in {int(exec_time_h)}h {int(exec_time_min)}min {int(exec_time_s)}s")

if __name__ == "__main__":
    main(
        fast_coeff=1,
        random_state=53, #53 (random_state qui semble maximiser le f1 score)
        test_size=0.2
    )
