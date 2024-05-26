from collections import Counter

def weighted_vote_prediction(y_pred_nn, y_pred_svm, y_pred_knn):

    f1_nn = 0.808
    f1_svm = 0.8256
    f1_knn = 0.7113

    total = f1_nn + f1_svm + f1_knn

    y_pred_final = []
    for nn, svm, knn in zip(y_pred_nn, y_pred_svm, y_pred_knn):
        votes = Counter({
            nn: f1_nn,
            svm: f1_svm,
            knn: f1_knn
        })

        y_pred_final.append(
            votes.most_common(1)[0][0]
        )

    return y_pred_final