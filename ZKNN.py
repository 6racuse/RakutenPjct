from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def train_knn(X_train, y_train):
    """
        Train a K-Nearest Neighbors (KNN) classifier on the provided training data.

        Args:
            X_train (sparse matrix): The feature matrix for training.
            y_train (array-like): The target labels for training.

        Returns:
            KNeighborsClassifier: The trained KNN classifier with the best found parameters.
    """
    params = {'n_neighbors': range(2, 20)}
    n_folds = 10
    cv = KFold(n_splits=n_folds, shuffle=False)

    grid_search = GridSearchCV(
        estimator=KNeighborsClassifier(),
        param_grid=params,
        return_train_score=True,
        cv=cv,
    ).fit(X_train, y_train)

    best_estimator = grid_search.best_estimator_

    BestKNN = KNeighborsClassifier(n_neighbors=49)
    BestKNN.fit(X_train, y_train)

    return BestKNN