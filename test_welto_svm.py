import BankModel
import header
import ProcessRakuten

import numpy as np
import pandas as pd
import spacy

import warnings
import time

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

exec_time_start = time.time()

# Ignore warnings
warnings.filterwarnings("ignore")

# Importation des données
X_train = pd.read_csv(
    "/Users/welto/Library/CloudStorage/OneDrive-CentraleSupelec/2A/CASA/RakutenPjct/data/X_train_update.csv",
    sep=','
)

Y_train = pd.read_csv(
    "/Users/welto/Library/CloudStorage/OneDrive-CentraleSupelec/2A/CASA/RakutenPjct/data/Y_train_CVw08PX.csv",
    sep=','
)

# Preprocessing
fast_coeff = 1

# Suppression des colonnes inutiles
X_train = X_train.drop(
    labels=['Unnamed: 0', 'imageid', 'description']
    , axis=1
)
Y_train = Y_train.drop(
    labels='Unnamed: 0',
    axis=1
)

# Séparation train-test
X_train, X_test = train_test_split(
    X_train,
    test_size=0.2,
    random_state=70
)
Y_train, Y_test = train_test_split(
    Y_train,
    test_size=0.2,
    random_state=70
)

# Tokenisation et cleaning
X_train_raw_designation = X_train['designation'][:X_train.shape[0]//fast_coeff].tolist()
X_test_raw_designation = X_test['designation'][:X_test.shape[0]//fast_coeff].tolist()
Y_train = Y_train['prdtypecode'][:Y_train.shape[0]//fast_coeff].tolist()
Y_test = Y_test['prdtypecode'][:Y_test.shape[0]//fast_coeff].tolist()


spacy_nlp = spacy.load("fr_core_news_sm")

X_train_raw_designation_clean = []
X_test_raw_designation_clean = []

a = len(X_train_raw_designation)
b = len(X_test_raw_designation)

header.progress_bar(
    0,
    a,
    prefix='Progress:',
    suffix='Complete',
    length=50
)

preprocessing_start_time = time.time()

for k in range(a):
    X_train_raw_designation_clean.append(
        header.raw_to_tokens(
            X_train_raw_designation[k],
            spacy_nlp
        )
    )
    header.progress_bar(
        k + 1,
        a,
        prefix='X_train_raw_designation_clean:',
        suffix='Complete',
        length=50
    )

for k in range(b):
    X_test_raw_designation_clean.append(
        header.raw_to_tokens(
            X_test_raw_designation[k],
            spacy_nlp
        )
    )
    header.progress_bar(
        k + 1,
        b,
        prefix='X_test_raw_designation_clean:',
        suffix='Complete',
        length=50
    )

preprocessing_end_time = time.time()
preprocessing_time_h, preprocessing_time_min, preprocessing_time_s = header.convert_seconds(preprocessing_end_time - preprocessing_start_time)

print(f"Preprocessed in {int(preprocessing_time_h)}h {int(preprocessing_time_min)}min {int(preprocessing_time_s)}s")

# Vectorisation
tfidf = TfidfVectorizer()

X_train_tfidf = tfidf.fit_transform(X_train_raw_designation_clean)
X_test_tfidf = tfidf.transform(X_test_raw_designation_clean)

print(X_train_tfidf.shape, X_test_tfidf.shape)

# Entraînement
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

training_start_time = time.time()
grid_search.fit(X_train_tfidf, Y_train)

training_end_time = time.time()
training_time_h, training_time_min, training_time_s = header.convert_seconds(training_end_time - training_start_time)

print(f"Model trained in {int(training_time_h)}h {int(training_time_min)}min {int(training_time_s)}s")

Y_pred_svm = grid_search.predict(X_test_tfidf)

# Évaluation du modèle
print(
    "f1 score :",
    f1_score(
        Y_test,
        Y_pred_svm,
        average='macro'
    )
)
print(
    "accuracy score :",
    accuracy_score(
        Y_test,
        Y_pred_svm
    )
)

exec_time_end = time.time()
exec_time_h, exec_time_min, exec_time_s = header.convert_seconds(exec_time_end - exec_time_start)

print(f"Executed in {int(exec_time_h)}h {int(exec_time_min)}min {int(exec_time_s)}s")