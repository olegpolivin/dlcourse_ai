import warnings


def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    # Implemented below
    predicted_positive = sum(prediction)
    tp = sum(prediction*ground_truth)
    fp = predicted_positive - tp

    precision = tp/predicted_positive
    recall = tp/sum(ground_truth)

    # Check that precision/recall are not zero
    if recall > 0 and precision > 0:
        f1 = 2/(1/recall + 1/precision)
    else:
        warnings.warn('Either precision or recall equal zero. Cannot calculate proper F1-measure')
        f1 = 0
    accuracy = sum(prediction == ground_truth) / len(ground_truth)
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    # Done below
    accuracy = sum(prediction == ground_truth) / len(ground_truth)
    return accuracy
