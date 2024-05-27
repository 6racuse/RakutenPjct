from collections import Counter

def weighted_vote_prediction(y_pred_nn, y_pred_svm, y_pred_rf):

    f1_nn = 0.808
    f1_svm = 0.8256
    f1_rf = 0.7920
    
    y_pred_final = []
<<<<<<< HEAD
    for nn, svm, rf in zip(y_pred_nn, y_pred_svm,y_pred_rf):
=======
    for nn, svm, knn, rf in zip(y_pred_nn, y_pred_svm,y_pred_rf):
>>>>>>> 3e871f03b3e7596dad48deeb20a37afa6ab93cb0
        votes = Counter({
            nn: f1_nn,
            svm: f1_svm,
            rf: f1_rf
        })

        y_pred_final.append(
            votes.most_common(1)[0][0]
        )

    return y_pred_final