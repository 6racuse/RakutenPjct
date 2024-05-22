import BankModel
import header
import ProcessRakuten

import numpy as np
import pandas as pd
import spacy

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

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
    random_state=42
)
Y_train, Y_test = train_test_split(
    Y_train,
    test_size=0.2,
    random_state=42
)

# Tokenisation et cleaning
X_train_raw_designation = X_train['designation'].tolist()
X_test_raw_designation = X_test['designation'].tolist()

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

# Vectorisation
tfidf = TfidfVectorizer()

X_train_tfidf = tfidf.fit_transform(X_train_raw_designation_clean)
X_test_tfidf = tfidf.transform(X_test_raw_designation_clean)

# Entraînement
fast_coeff = 1

svm = SVC()

svm.fit(
    X_train_tfidf[:X_train_tfidf.shape[0]//fast_coeff],
    Y_train[:len(Y_train)//fast_coeff]
)

Y_pred_svm = svm.predict(X_test_tfidf[:X_test_tfidf.shape[0]//fast_coeff])

# Évaluation du modèle
print(
    "f1 score :",
    f1_score(
        Y_test[:X_test_tfidf.shape[0]//fast_coeff],
        Y_pred_svm,
        average='macro'
    )
)
print(
    "accuracy score :",
    accuracy_score(
        Y_test[:X_test_tfidf.shape[0]//fast_coeff],
        Y_pred_svm
    )
)