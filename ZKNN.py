from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def train_knn(X_train, y_train):
    BestKNN = KNeighborsClassifier(n_neighbors=49)
    BestKNN.fit(X_train, y_train)

    return BestKNN