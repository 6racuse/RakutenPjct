from ZManageData import progress_bar, normalize_accent
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import Zmain
from ZManageData import Preprocess_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import time
from nltk.corpus import stopwords
import string

def plot_f1_scores(X_train, X_test, y_train, y_test):
    f1_scores = []
    neighbors = list(range(1, 50))

    for n in neighbors:
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        f1_scores.append(f1)

    plt.plot(neighbors, f1_scores)
    plt.xlabel('Number of Neighbors')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Number of Neighbors')
    plt.show()


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

    return X_train, X_test, Y_train, Y_test
def tokenise_cleaning_data(X_train, X_test):

    nltk.download("punkt")
    nltk.download("stopwords")

    stop_words = set(stopwords.words('french'))

    X_train_clean = []
    X_test_clean = []

    a = len(X_train)
    b = len(X_test)

    progress_bar(
        0,
        a,
        prefix='Progress:',
        suffix='Complete',
        length=50
    )

    preprocessing_start_time = time.time()

    for k in range(a):
        tokens = word_tokenize(
            normalize_accent(
                X_train[k].lower()
            ),
            language='french'
        )
        tokens = [word for word in tokens if word not in string.punctuation and word not in stop_words]
        X_train_clean.append(tokens)
        progress_bar(
            k + 1,
            a,
            prefix='X_train_raw_designation_clean :',
            suffix='Complete',
            length=50
        )

    for k in range(b):
        tokens = word_tokenize(
            normalize_accent(
                X_test[k].lower()
            ),
            language='french'
        )
        tokens = [word for word in tokens if word not in string.punctuation and word not in stop_words]
        X_test_clean.append(tokens)
        progress_bar(
            k + 1,
            b,
            prefix='X_test_raw_designation_clean :',
            suffix='Complete',
            length=50
        )

    X_train_clean = [' '.join(tokens) for tokens in X_train_clean]
    X_test_clean = [' '.join(tokens) for tokens in X_test_clean]

    preprocessing_end_time = time.time()
    preprocessing_time_h, preprocessing_time_min, preprocessing_time_s, preprocessing_time_ms = convert_seconds(
        preprocessing_end_time - preprocessing_start_time
    )

    print(f"Preprocessed in {int(preprocessing_time_h)}h {int(preprocessing_time_min)}min {int(preprocessing_time_s)}s {int(preprocessing_time_ms)}ms")

    return X_train_clean, X_test_clean

def vectorize_data(X_train_clean, X_test_clean):
    tfidf = TfidfVectorizer()

    X_train_tfidf = tfidf.fit_transform(X_train_clean)
    X_test_tfidf = tfidf.transform(X_test_clean)

    return X_train_tfidf, X_test_tfidf

X_train, X_test, y_data, y_test = load_data(
    fast_coeff=1,
    random_state=42,
    test_size=0.2
)

X_train, X_test = tokenise_cleaning_data(X_train, X_test)
X_train_tfidf, X_test_tfidf = vectorize_data(X_train, X_test)
plot_f1_scores(X_train, X_test, y_data, y_test)