from sklearn.metrics import f1_score
import numpy as np
from collections import Counter

def weighted_vote_prediction(y_test, y_pred_nn, y_pred_svm, y_pred_knn):

    f1_nn = f1_score(
        y_test,
        y_pred_nn,
        average='macro'
    )
    f1_svm = f1_score(
        y_test,
        y_pred_svm,
        average='macro'
    )
    f1_knn = f1_score(
        y_test,
        y_pred_knn,
        average='macro'
    )

    total = f1_nn + f1_svm + f1_knn
    weight_nn = f1_nn / total
    weight_svm = f1_svm / total
    weight_knn = f1_knn / total

    y_pred_final = []
    for nn, svm, knn in zip(y_pred_nn, y_pred_svm, y_pred_knn):
        votes = Counter({nn: weight_nn, svm: weight_svm, knn: weight_knn})

        y_pred_final.append(
            votes.most_common(1)[0][0]
        )

    return y_pred_final