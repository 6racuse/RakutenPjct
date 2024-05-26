from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import Zmain
from ZManageData import Preprocess_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

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


X_train, y_data, X_test = Preprocess_dataset()
X_train, X_test, y_data, y_test = train_test_split(
    X_train,
    y_data,
    random_state=42,
    shuffle=True,
    test_size=0.2
)
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

plot_f1_scores(X_train, X_test, y_data, y_test)