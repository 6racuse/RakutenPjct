from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def train_knn(X_train, y_train):
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