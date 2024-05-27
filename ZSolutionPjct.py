from collections import Counter

def weighted_vote_prediction(y_pred_nn, y_pred_svm, y_pred_rf):
    """
        This function performs a weighted vote prediction based on the predictions of three models: Neural Network (NN), Support Vector Machine (SVM), and Random Forest (RF).
        The weights are determined by the F1 scores of the respective models.

        Args:
            y_pred_nn (array-like): The predictions made by the NN model.
            y_pred_svm (array-like): The predictions made by the SVM model.
            y_pred_rf (array-like): The predictions made by the RF model.

        Returns:
            y_pred_final (list): The final predictions made by performing a weighted vote of the three models' predictions.
    """

    # F1 scores of the models
    f1_nn = 0.808
    f1_svm = 0.8256
    f1_rf = 0.7920
    
    y_pred_final = []

    for nn, svm, rf in zip(y_pred_nn, y_pred_svm,y_pred_rf):
        # Count the votes of the models, weighted by their F1 scores
        votes = Counter({
            nn: f1_nn,
            svm: f1_svm,
            rf: f1_rf
        })

        # Append the prediction with the highest weighted vote to the final predictions
        y_pred_final.append(
            votes.most_common(1)[0][0]
        )

    return y_pred_final
