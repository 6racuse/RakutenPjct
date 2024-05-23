import header

import numpy as np
import pandas as pd
import spacy

import warnings
import time

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
    Y_train = pd.read_csv(
        "/Users/welto/Library/CloudStorage/OneDrive-CentraleSupelec/2A/CASA/RakutenPjct/data/Y_train_CVw08PX.csv",
        sep=',',
        usecols=lambda column: column != 'Unnamed: 0'
    )
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_train,
        Y_train,
        test_size=0.2,
        random_state=42
    )
    X_train = X_train['designation'][:X_train.shape[0]//fast_coeff].tolist()
    X_test = X_test['designation'][:X_test.shape[0]//fast_coeff].tolist()

    Y_train = Y_train['prdtypecode'][:Y_train.shape[0]//fast_coeff].tolist()
    Y_test = Y_test['prdtypecode'][:Y_test.shape[0]//fast_coeff].tolist()

    return X_train, X_test, Y_train, Y_test
def tokenise_cleaning_data(X_train, X_test):
    spacy_nlp = spacy.load("fr_core_news_sm")

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
        X_train_clean.append(
            header.raw_to_tokens(
                X_train[k],
                spacy_nlp
            )
        )
        header.progress_bar(
            k + 1,
            a,
            prefix='X_train_raw_designation_clean :',
            suffix='Complete',
            length=50
        )

    for k in range(b):
        X_test_clean.append(
            header.raw_to_tokens(
                X_test[k],
                spacy_nlp
            )
        )
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

    return X_train_clean, X_test_clean
def vectorize_data(X_train_clean, X_test_clean):
    tfidf = TfidfVectorizer()

    X_train_tfidf = tfidf.fit_transform(X_train_clean)
    X_test_tfidf = tfidf.transform(X_test_clean)

    return X_train_tfidf, X_test_tfidf
def train_model(X_train_tfidf, Y_train):
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf']
    }
    grid_search = GridSearchCV(
        SVC(),
        param_grid,
        refit=True,
        verbose=10,
        cv=5,
        scoring='f1_macro'
    )
    grid_search.fit(
        X_train_tfidf,
        Y_train
    )
    return grid_search
def evaluate_model(model, X_test_tfidf, Y_test):
    Y_pred = model.predict(X_test_tfidf)

    f1 = f1_score(
        Y_test,
        Y_pred,
        average='macro'
    )
    accuracy = accuracy_score(
        Y_test,
        Y_pred
    )
    return f1, accuracy
def main():
    exec_time_start = time.time()
    warnings.filterwarnings("ignore")

    X_train, X_test, Y_train, Y_test = load_data(100)

    X_train_clean, X_test_clean = tokenise_cleaning_data(
        X_train,
        X_test
    )

    X_train_tfidf, X_test_tfidf = vectorize_data(
        X_train_clean,
        X_test_clean
    )

    model = train_model(
        X_train_tfidf,
        Y_train
    )

    f1, accuracy = evaluate_model(
        model,
        X_test_tfidf,
        Y_test
    )

    print(
        "f1 score:",
        f1
    )
    print(
        "accuracy score:",
        accuracy
    )

    exec_time_end = time.time()
    exec_time_h, exec_time_min, exec_time_s = header.convert_seconds(exec_time_end - exec_time_start)
    print(f"Executed in {int(exec_time_h)}h {int(exec_time_min)}min {int(exec_time_s)}s")


if __name__ == "__main__":
    main()
