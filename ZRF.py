from sklearn.tree import _tree
import matplotlib.pyplot as plt
import joblib

def train_rf(X_train,Y_train):
    from sklearn.ensemble import RandomForestClassifier
    best_param = {'n_estimators': 600}
    rf_model = RandomForestClassifier(n_estimators=best_param['n_estimators'], verbose=10, n_jobs=-1)
    rf_model.fit(X_train, Y_train)
    return rf_model
    

    
def Global_get_best_params(X_train,Y_train):
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np

    X_train_transposed = X_train.T  
    classifier = RandomForestClassifier()

    param_grid = {
        'n_estimators': [10, 50, 100],  
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]  
    }

    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_transposed, Y_train)
    return grid_search.best_params





def get_tree_structure(tree, feature_names):
    node_count = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold

    def recurse(node):
        if children_left[node] == _tree.TREE_LEAF:
            return {'value': tree.value[node]}
        else:
            feature_name = feature_names[feature[node]]
            left = recurse(children_left[node])
            right = recurse(children_right[node])
            return {
                'name': f'Feature {feature[node]} <= {threshold[node]}',
                'children': [left, right]
            }

    return recurse(0)

def plot_dendrogram(forest, feature_names):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("Random Forest Dendrogram")
    ax.set_xlabel("Tree Index")
    ax.set_ylabel("Depth")

    for i, tree in enumerate(forest.estimators_):
        tree_structure = get_tree_structure(tree.tree_, feature_names)
        plot_tree_structure(ax, tree_structure, depth=0, tree_index=i)

    plt.tight_layout()
    plt.show()

def plot_tree_structure(ax, node, depth, tree_index):
    if 'name' in node:
        ax.text(tree_index, depth, node['name'], fontsize=8)
        if 'children' in node:
            for child in node['children']:
                plot_tree_structure(ax, child, depth + 1, tree_index)

def setupmain_plot():
        
    random_forest_model = joblib.load("./models/rf_model.joblib")

    feature_names = list(range(len(random_forest_model.feature_importances_)))

    plot_dendrogram(random_forest_model, feature_names)

#setupmain_plot()